"""Session bot - one XMPP bot per session."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from src.core.session_runtime import SessionRuntime
from src.core.session_runtime.api import (
    EventSinkPort,
    OutboundMessage,
    ProcessingChanged,
    SessionEvent,
    SessionPort,
)
from src.core.session_runtime.ports import (
    AttachmentPromptPort,
    HistoryPort,
    MessageStorePort,
    RalphLoopStorePort,
    RunnerFactoryPort,
    SessionState,
    SessionStorePort,
)
from src.bots.session.inbound import (
    extract_attachment_urls,
    extract_bob_images,
    extract_switch_meta,
    normalize_leading_at,
    strip_urls_from_body,
)
from src.bots.session.xhtml import build_xhtml_message
from src.bots.session.typing import TypingIndicator
from src.commands import CommandHandler
from src.db import MessageRepository, RalphLoopRepository, SessionRepository
from src.lifecycle.sessions import create_session as lifecycle_create_session
from src.helpers import (
    append_to_history,
    log_activity,
)
from src.runners import Runner, create_runner
from src.runners.debate.config import DebateConfig
from src.runners.pi.config import PiConfig
from src.attachments import Attachment, AttachmentStore
from src.utils import SWITCH_META_NS, BaseXMPPBot, build_message_meta

if TYPE_CHECKING:
    import sqlite3

    from src.db import Session
    from src.manager import SessionManager


class SessionBot(BaseXMPPBot):
    """XMPP bot for a single session."""

    def __init__(
        self,
        session_name: str,
        jid: str,
        password: str,
        db: "sqlite3.Connection",
        working_dir: str,
        output_dir: Path,
        xmpp_recipient: str,
        xmpp_domain: str,
        xmpp_server: str,
        ejabberd_ctl: str,
        manager: "SessionManager | None" = None,
    ):
        super().__init__(jid, password, recipient=xmpp_recipient)
        self.session_name = session_name
        self.log = logging.getLogger(f"session.{self.session_name}")
        self.db = db
        self.sessions = SessionRepository(db)
        self.messages = MessageRepository(db)
        self.ralph_loops = RalphLoopRepository(db)
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.xmpp_recipient = xmpp_recipient
        self.xmpp_domain = xmpp_domain
        self.xmpp_server = xmpp_server
        self.ejabberd_ctl = ejabberd_ctl
        self.manager = manager
        self.runner: Runner | None = None
        self.room_jid: str | None = None
        self.room_nick: str = self.session_name
        self.startup_error: str | None = None
        self.processing = False
        self.shutting_down = False
        self._typing = TypingIndicator(
            send_typing=self.send_typing,
            is_active=lambda: self.processing,
            is_shutting_down=lambda: self.shutting_down,
        )

        # Best-effort escape hatch: when /cancel is used against a local vLLM-backed
        # model, we may need to hard-abort the GPU-side generation.
        self._last_vllm_abort_ts = 0.0
        self._vllm_abort_task: asyncio.Task | None = None

        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_attempt: int = 0

        self._runtime = self._build_runtime()
        # Public session interface for commands/other adapters.
        self.session: SessionPort = self._runtime
        self.attachment_store = AttachmentStore()
        self.commands = CommandHandler(self)

        self.register_plugin("xep_0045")

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

    # -------------------------------------------------------------------------
    # Runtime wiring
    # -------------------------------------------------------------------------

    class _EventSinkAdapter(EventSinkPort):
        def __init__(self, bot: "SessionBot"):
            self._bot = bot

        async def emit(self, event: SessionEvent) -> None:
            if isinstance(event, ProcessingChanged):
                self._bot.processing = event.active
                if event.active:
                    self._bot._typing.start()
                else:
                    self._bot._typing.stop()
                return

            if isinstance(event, OutboundMessage):
                self._bot.send_reply(
                    event.text,
                    meta_type=event.meta_type,
                    meta_tool=event.meta_tool,
                    meta_attrs=event.meta_attrs,
                    meta_payload=event.meta_payload,
                )
                # Tool/progress messages often clear typing indicators.
                if event.meta_type == "tool":
                    self._bot._typing.maybe_send(min_interval_s=5.0)
                return

    class _SessionsAdapter(SessionStorePort):
        def __init__(self, repo: SessionRepository):
            self._repo = repo

        def get(self, name: str) -> SessionState | None:
            s = self._repo.get(name)
            if not s:
                return None
            return SessionState(
                name=s.name,
                active_engine=s.active_engine,
                claude_session_id=s.claude_session_id,
                pi_session_id=s.pi_session_id,
                model_id=s.model_id,
                reasoning_mode=s.reasoning_mode,
            )

        async def update_last_active(self, name: str) -> None:
            await self._repo.update_last_active(name)

        async def update_claude_session_id(self, name: str, session_id: str) -> None:
            await self._repo.update_claude_session_id(name, session_id)

        async def update_pi_session_id(self, name: str, session_id: str) -> None:
            await self._repo.update_pi_session_id(name, session_id)

    class _MessagesAdapter(MessageStorePort):
        def __init__(self, repo: MessageRepository):
            self._repo = repo

        async def add(self, session_name: str, role: str, content: str, engine: str) -> None:
            await self._repo.add(session_name, role, content, engine)

    class _RunnerFactoryAdapter(RunnerFactoryPort):
        def create(
            self,
            engine: str,
            *,
            working_dir: str,
            output_dir: Path,
            session_name: str,
            pi_config: PiConfig | None = None,
            debate_config: DebateConfig | None = None,
        ) -> Runner:
            return create_runner(
                engine,
                working_dir=working_dir,
                output_dir=output_dir,
                session_name=session_name,
                pi_config=pi_config,
                debate_config=debate_config,
            )

    class _HistoryAdapter(HistoryPort):
        def append_to_history(
            self, message: str, working_dir: str, claude_session_id: str | None
        ) -> None:
            append_to_history(message, working_dir, claude_session_id)

        def log_activity(self, message: str, *, session: str, source: str) -> None:
            log_activity(message, session=session, source=source)

    class _PromptAdapter(AttachmentPromptPort):
        def augment_prompt(
            self, body: str, attachments: list[Attachment] | None
        ) -> str:
            if not attachments:
                return (body or "").strip()
            lines: list[str] = [(body or "").strip(), "", "User attached image(s):"]
            for a in attachments:
                lines.append(f"- {a.local_path}")
            return "\n".join(lines).strip()

    class _RalphLoopsAdapter(RalphLoopStorePort):
        def __init__(self, repo: RalphLoopRepository):
            self._repo = repo

        async def create(
            self,
            session_name: str,
            prompt: str,
            max_iterations: int,
            completion_promise: str | None,
            wait_seconds: float,
        ) -> int:
            return await self._repo.create(
                session_name,
                prompt,
                max_iterations=max_iterations,
                completion_promise=completion_promise,
                wait_seconds=float(wait_seconds or 0.0),
            )

        async def update_progress(
            self,
            loop_id: int,
            current_iteration: int,
            total_cost: float,
            status: str = "running",
        ) -> None:
            await self._repo.update_progress(loop_id, current_iteration, total_cost, status)

    def _build_runtime(self) -> SessionRuntime:
        return SessionRuntime(
            session_name=self.session_name,
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            sessions=self._SessionsAdapter(self.sessions),
            messages=self._MessagesAdapter(self.messages),
            events=self._EventSinkAdapter(self),
            runner_factory=self._RunnerFactoryAdapter(),
            history=self._HistoryAdapter(),
            prompt=self._PromptAdapter(),
            ralph_loops=self._RalphLoopsAdapter(self.ralph_loops),
            infer_meta_tool_from_summary=self._infer_meta_tool_from_summary,
        )

    # -------------------------------------------------------------------------
    # XMPP lifecycle
    # -------------------------------------------------------------------------

    async def on_start(self, event):
        await self.guard(self._on_start(event), context="session.on_start")

    async def _on_start(self, event):
        self.send_presence()
        try:
            await asyncio.wait_for(self.get_roster(), timeout=15)
            await asyncio.wait_for(
                self["xep_0280"].enable(), timeout=15  # type: ignore[attr-defined,union-attr]
            )
        except asyncio.TimeoutError:
            self.startup_error = "XMPP startup timed out (roster/carbons)"
            self.log.error("Startup timed out during roster/carbons")
            self.disconnect()
            return
        self._load_room_settings()
        if self.room_jid:
            joined = await self._join_collaboration_room()
            if not joined:
                if not self.startup_error:
                    self.startup_error = f"failed to join room {self.room_jid}"
                self.log.error("Startup aborted: %s", self.startup_error)
                self.disconnect()
                self.set_connected(False)
                return
        self.log.info("Connected")
        self.set_connected(True)
        self._reconnect_attempt = 0

    def on_disconnected(self, event):
        self.set_connected(False)
        if self.shutting_down:
            self.log.info("Disconnected during shutdown; not reconnecting")
            return
        if self._reconnect_task and not self._reconnect_task.done():
            self.log.debug("Reconnect already in progress; skipping duplicate")
            return
        self._reconnect_task = asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        _MAX_ATTEMPTS = 10
        _BASE_DELAY = 5
        _MAX_DELAY = 60
        while self._reconnect_attempt < _MAX_ATTEMPTS:
            if self.shutting_down:
                return
            self._reconnect_attempt += 1
            delay = min(_BASE_DELAY * (2 ** (self._reconnect_attempt - 1)), _MAX_DELAY)
            self.log.warning(
                "Reconnecting (attempt %d/%d) in %ds...",
                self._reconnect_attempt, _MAX_ATTEMPTS, delay,
            )
            await asyncio.sleep(delay)
            if self.shutting_down:
                return
            try:
                self.connect()
            except Exception:
                self.log.warning("Reconnect connect() failed", exc_info=True)
                continue
            # connect() succeeded at the transport level — wait for
            # _on_start to confirm (which resets _reconnect_attempt)
            # or for on_disconnected to re-enter this path.
            return
        self.log.error("Giving up reconnect after %d attempts", _MAX_ATTEMPTS)

    def send_reply(
        self,
        text: str,
        recipient: str | None = None,
        *,
        meta_type: str | None = None,
        meta_tool: str | None = None,
        meta_attrs: dict[str, str] | None = None,
        meta_payload: object | None = None,
    ):
        """Send message, splitting if needed."""
        if self.shutting_down:
            return

        # Some runners occasionally include trailing blank lines; trim only
        # terminal newlines so clients do not render padded empty space.
        text = text.rstrip("\r\n")

        # Many XMPP clients and bridges display/truncate long messages.
        # Keep a conservative default and allow override for power users.
        try:
            max_len = int(os.getenv("SWITCH_XMPP_MESSAGE_MAX_LEN", "3500"))
        except ValueError:
            max_len = 3500
        max_len = max(500, min(max_len, 100000))

        def _safe_send(m) -> bool:
            try:
                m.send()
                return True
            except Exception:
                self.log.warning("XMPP send failed", exc_info=True)
                return False

        target = recipient or self._default_reply_recipient()
        is_room_target = bool(self.room_jid and target == self.room_jid)
        if len(text) <= max_len:
            msg = self.make_message(
                mto=target,
                mbody=text,
                mtype="groupchat" if is_room_target else "chat",
            )
            if not is_room_target:
                msg["chat_state"] = "active"

            rich = build_xhtml_message(text)
            if rich is not None:
                msg.xml.append(rich)

            if meta_type:
                meta = build_message_meta(
                    meta_type,
                    meta_tool=meta_tool,
                    meta_attrs=meta_attrs,
                    meta_payload=meta_payload,
                )
                msg.xml.append(meta)

            _safe_send(msg)
            return

        parts = self._split_message(text, max_len)
        total = len(parts)
        for i, part in enumerate(parts, 1):
            header = f"[{i}/{total}]\n" if i > 1 else ""
            footer = f"\n[{i}/{total}]" if i < total else ""
            body = header + part + footer if total > 1 else part
            if is_room_target:
                msg = self.make_message(mto=target, mbody=body, mtype="groupchat")
            else:
                msg = self.make_message(mto=target, mbody=body, mtype="chat")
                msg["chat_state"] = "active" if i == total else "composing"

            rich = build_xhtml_message(body)
            if rich is not None:
                msg.xml.append(rich)

            if meta_type:
                meta = build_message_meta(
                    meta_type,
                    meta_tool=meta_tool,
                    meta_attrs=meta_attrs,
                    meta_payload=meta_payload,
                )
                msg.xml.append(meta)

            if not _safe_send(msg):
                break

    def _default_reply_recipient(self) -> str:
        return self.room_jid or self.xmpp_recipient

    def send_typing(self, recipient: str | None = None):
        target = recipient or self._default_reply_recipient()
        if self.room_jid and target == self.room_jid:
            return
        super().send_typing(recipient=target)

    def _load_room_settings(self) -> None:
        session = self.sessions.get(self.session_name)
        self.room_jid = (session.room_jid or "").split("/", 1)[0] if session else None
        self.room_nick = self.session_name

    async def _join_collaboration_room(self) -> bool:
        room = (self.room_jid or "").strip()
        if not room:
            return True
        try:
            muc = cast(Any, self["xep_0045"])
            await muc.join_muc(room, self.room_nick)  # type: ignore[attr-defined]
            participants = self.sessions.list_collaborators(self.session_name)
            for participant in participants:
                if participant == str(self.boundjid.bare):
                    continue
                try:
                    muc.invite(room, participant)  # type: ignore[attr-defined]
                except Exception:
                    continue
            return True
        except Exception:
            self.startup_error = f"failed to join collaboration room {room}"
            self.log.exception("Failed to join collaboration room: %s", room)
            return False

    @staticmethod
    def _infer_meta_tool_from_summary(summary: str) -> str | None:
        """Best-effort mapping from tool summary text to meta.tool."""
        # Pi tool summaries look like: "[tool:bash ...]".
        if summary.startswith("[tool:"):
            end = summary.find("]")
            head = summary[:end] if end != -1 else summary
            inner = head[len("[tool:") :]
            tool = inner.split(maxsplit=1)[0].strip()
            return tool or None

        # Pi tool-result summaries look like: "[tool-result:bash ...]".
        if summary.startswith("[tool-result:"):
            end = summary.find("]")
            head = summary[:end] if end != -1 else summary
            inner = head[len("[tool-result:") :]
            tool = inner.split(maxsplit=1)[0].strip()
            return tool or None

        # Claude tool summaries look like: "[Bash: ...]" / "[Read: ...]".
        if summary.startswith("[") and ":" in summary:
            name = summary[1:].split(":", 1)[0].strip()
            return name.lower() if name else None

        return None

    # -------------------------------------------------------------------------
    # Cancellation / shutdown
    # -------------------------------------------------------------------------

    def cancel_operations(self, *, notify: bool = False) -> bool:
        """Best-effort cancellation of in-flight work.

        Returns True if there was something to cancel.
        """
        cancelled_any = False

        if self._runtime.cancel_operations(notify=notify):
            cancelled_any = True

        # Best-effort: also cancel any ad-hoc runner paths.
        if self.runner and self.processing:
            cancelled_any = True
            self.runner.cancel()

        # If we're using Helga vLLM, cancellation at the runner layer doesn't
        # always stop the underlying vLLM generation.
        # For those cases, do a best-effort hard abort.
        if cancelled_any:
            self._maybe_hard_abort_vllm()

        return cancelled_any

    def _maybe_hard_abort_vllm(self) -> None:
        """Best-effort: restart Helga vLLM to abort the active generation.

        This is intentionally conservative:
        - Only triggers for Pi sessions using vLLM-backed models
        - Only triggers when the selected model is our vLLM-backed GLM provider
        - Cooldown + single in-flight task to avoid restart storms
        """

        enabled = os.getenv("SWITCH_VLLM_HARD_CANCEL", "1").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return

        session = self.sessions.get(self.session_name)
        if not session:
            return
        engine = (session.active_engine or "").strip().lower()
        if engine not in {"pi", "debate"}:
            return
        model_id = (session.model_id or "").strip()
        if not model_id.startswith("glm_vllm/"):
            return

        # Avoid spamming restarts if multiple cancellation paths fire.
        try:
            cooldown_s = float(os.getenv("SWITCH_VLLM_HARD_CANCEL_COOLDOWN_S", "10"))
        except ValueError:
            cooldown_s = 10.0
        now = time.monotonic()
        if now - self._last_vllm_abort_ts < max(0.0, cooldown_s):
            return
        self._last_vllm_abort_ts = now

        if self._vllm_abort_task and not self._vllm_abort_task.done():
            return

        self._vllm_abort_task = self.spawn_guarded(
            self._hard_abort_vllm(), context="session.vllm.hard_abort"
        )

    async def _hard_abort_vllm(self) -> None:
        """Restart the vLLM container on Helga and wait for readiness."""

        host = os.getenv("SWITCH_VLLM_SSH_HOST", "chkn_gpus").strip() or "chkn_gpus"
        container = (
            os.getenv(
                "SWITCH_VLLM_DOCKER_CONTAINER", "glm47_switch_128k"
            ).strip()
            or "glm47_switch_128k"
        )
        base_url = os.getenv(
            "SWITCH_VLLM_HEALTH_URL", "http://127.0.0.1:8027/v1/models"
        )

        try:
            timeout_s = float(os.getenv("SWITCH_VLLM_HARD_CANCEL_TIMEOUT_S", "90"))
        except ValueError:
            timeout_s = 90.0

        remote_cmd = (
            "set -euo pipefail; "
            f"docker restart {container} >/dev/null; "
            "for i in $(seq 1 120); do "
            f"curl -fsS {base_url} >/dev/null && exit 0; "
            "sleep 1; "
            "done; "
            "echo vllm_not_ready >&2; exit 1"
        )

        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            remote_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            with suppress(Exception):
                proc.kill()
            self.log.warning(
                "vLLM hard abort timed out host=%s container=%s health=%s timeout_s=%s",
                host,
                container,
                base_url,
                timeout_s,
            )
            return

        if proc.returncode != 0:
            out = (stdout or b"").decode("utf-8", errors="replace").strip()
            err = (stderr or b"").decode("utf-8", errors="replace").strip()
            self.log.warning(
                "vLLM hard abort failed (rc=%s) host=%s container=%s stdout=%s stderr=%s",
                proc.returncode,
                host,
                container,
                out[-1000:],
                err[-1000:],
            )

    async def hard_kill(self) -> None:
        """Hard-kill this session.

        - Cancels in-flight work
        - Prevents reconnect
        - Deletes XMPP account, kills tmux tail, marks session closed
        """
        if self.shutting_down:
            return

        self.shutting_down = True

        self._runtime.shutdown()

        # Stop any in-flight work and drop queued messages.
        self.cancel_operations(notify=False)

        self.send_reply("Session closed. Goodbye!")

        # Give any final messages a brief chance to flush.
        await asyncio.sleep(0.25)

        try:
            manager = self.manager
            if manager is None:
                raise RuntimeError("Session manager unavailable")
            await manager.kill_session(self.session_name, send_goodbye=False)
        finally:
            with suppress(Exception):
                self.disconnect()

    def _split_message(self, text: str, max_len: int) -> list[str]:
        """Split text into chunks respecting paragraph boundaries."""
        parts = []
        current = ""
        for para in text.split("\n\n"):
            if len(current) + len(para) + 2 <= max_len:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    parts.append(current)
                if len(para) > max_len:
                    # Split long paragraphs by line
                    current = ""
                    for line in para.split("\n"):
                        if len(current) + len(line) + 1 <= max_len:
                            current = f"{current}\n{line}" if current else line
                        else:
                            if current:
                                parts.append(current)
                            while len(line) > max_len:
                                parts.append(line[:max_len])
                                line = line[max_len:]
                            current = line
                else:
                    current = para
        if current:
            parts.append(current)
        return parts

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_message(self, msg):
        await self.guard(
            self._handle_session_message(msg), context="session.on_message"
        )

    async def _handle_session_message(self, msg):
        msg_type = msg["type"]
        if msg_type not in ("chat", "normal", "groupchat"):
            return

        if self.shutting_down:
            return

        sender = str(msg["from"].bare)
        dispatcher_jid = self._current_dispatcher_jid()

        if msg_type == "groupchat":
            room = self.room_jid
            if not room or sender != room:
                return
            nick = str(msg["from"].resource or "")
            if nick and nick == self.room_nick:
                return
            is_scheduled = False
        else:
            if self.room_jid and sender != dispatcher_jid:
                # Collaboration sessions only accept user chat inside the room.
                return
            is_scheduled = sender == dispatcher_jid

        trusted_peer = self._is_trusted_peer_session_sender(sender)
        if (
            msg_type != "groupchat"
            and sender not in (self.xmpp_recipient, dispatcher_jid)
            and not trusted_peer
        ):
            return

        meta_type, meta_attrs, meta_payload = extract_switch_meta(
            msg, meta_ns=SWITCH_META_NS
        )

        # Allow meta-only messages (e.g., button-based question replies).
        body = (msg["body"] or "").strip()
        attachments: list[Attachment] = []
        urls = extract_attachment_urls(msg, body)

        if urls:
            attachments.extend(
                await self.attachment_store.download_images(self.session_name, urls)
            )
            if attachments:
                body = strip_urls_from_body(body, urls)

        bob_images = extract_bob_images(msg)
        if bob_images:
            attachments.extend(
                self.attachment_store.store_images_from_bytes(
                    self.session_name, bob_images
                )
            )

        if attachments:
            self._send_attachment_meta(attachments)

        if meta_type == "question-reply":
            request_id = (meta_attrs or {}).get("request_id")
            answer_obj: object | None = None
            if isinstance(meta_payload, dict):
                if "answers" in meta_payload:
                    answer_obj = meta_payload.get("answers")
                elif "text" in meta_payload:
                    answer_obj = meta_payload.get("text")
            if answer_obj is None:
                answer_obj = body
            if self.answer_pending_question(answer_obj, request_id=request_id):
                self.log.info("Answered pending question via meta reply")
            return

        if not body and attachments:
            body = "Please analyze the attached image(s)."
        if not body:
            return

        body = normalize_leading_at(body)

        self.log.info(f"Message{'[scheduled]' if is_scheduled else ''}: {body[:50]}...")

        # Commands only from user
        if not is_scheduled and await self.commands.handle(body):
            return

        # Shell commands — only session owner or collaborators may execute.
        if body.startswith("!"):
            if not self._is_shell_authorized(msg):
                self.send_reply("Shell commands are restricted to the session owner and collaborators.")
                return
            await self.run_shell_command(body[1:].strip())
            return

        # Check for pending question answers first
        if self.answer_pending_question(body):
            self.log.info(f"Answered pending question with: {body[:50]}...")
            return

        # If Ralph is running, inject the message instead of queuing.
        if not is_scheduled and self._runtime.inject_ralph_prompt(body):
            self.log.info(f"Injected into Ralph: {body[:50]}...")
            return

        # Scheduled messages are best-effort; drop them if we're already running.
        if is_scheduled and self.processing:
            return

        # If we're busy, allow +spawn to fork work instead of queuing it.
        if (
            self.processing
            and (not is_scheduled)
            and body.startswith("+")
            and self.manager
        ):
            await self.spawn_sibling_session(body[1:].strip())
            return

        queued_before = self.processing or (self._runtime.pending_count() > 0)
        await self._runtime.enqueue(
            body,
            attachments,
            trigger_response=True,
            scheduled=is_scheduled,
            wait=False,
        )
        if queued_before and not is_scheduled:
            self.send_reply(f"Queued ({self._runtime.pending_count()} pending)")
        return

    def _current_dispatcher_jid(self) -> str:
        session = self.sessions.get(self.session_name)
        if session and session.dispatcher_jid:
            return session.dispatcher_jid.split("/", 1)[0]
        return f"oc@{self.xmpp_domain}"

    def _is_trusted_peer_session_sender(self, sender_jid: str) -> bool:
        sender_bare = (sender_jid or "").split("/", 1)[0]
        if not sender_bare:
            return False

        peer = self.sessions.get_by_jid(sender_bare)
        if not peer or peer.status != "active" or peer.name == self.session_name:
            return False

        current = self.sessions.get(self.session_name)
        if not current:
            return False

        current_dispatcher = (current.dispatcher_jid or "").split("/", 1)[0]
        peer_dispatcher = (peer.dispatcher_jid or "").split("/", 1)[0]

        if current_dispatcher and peer_dispatcher:
            return current_dispatcher == peer_dispatcher

        return not current_dispatcher and not peer_dispatcher

    def _send_attachment_meta(self, attachments: list[Attachment]) -> None:
        payload = {
            "version": 1,
            "engine": "switch",
            "attachments": [
                {
                    "id": a.id,
                    "kind": a.kind,
                    "mime": a.mime,
                    "filename": a.filename,
                    "local_path": a.local_path,
                    "public_url": a.public_url,
                    "size_bytes": a.size_bytes,
                    "sha256": a.sha256,
                    "original_url": a.original_url,
                }
                for a in attachments
            ],
        }
        self.send_reply(
            f"[Received {len(attachments)} image(s)]",
            meta_type="attachment",
            meta_tool="attachment",
            meta_attrs={
                "version": "1",
                "engine": "switch",
                "count": str(len(attachments)),
            },
            meta_payload=payload,
        )

    # -------------------------------------------------------------------------
    # Shell commands
    # -------------------------------------------------------------------------

    def _is_shell_authorized(self, msg) -> bool:
        """Check if the message sender is the session owner or a collaborator."""
        session = self.sessions.get(self.session_name)
        if not session:
            return False

        owner_bare = (session.owner_jid or "").split("/", 1)[0]
        msg_type = msg["type"]

        if msg_type == "groupchat":
            # In MUC, we can't reliably get the real JID from the nick.
            # Shell commands are too dangerous to allow without identity verification.
            return False

        sender_bare = str(msg["from"].bare)
        if owner_bare and sender_bare == owner_bare:
            return True

        collaborators = self.sessions.list_collaborators(self.session_name)
        return sender_bare in collaborators

    async def run_shell_command(self, cmd: str):
        """Run shell command, send output, inform agent."""
        if not cmd:
            self.send_reply("Usage: !<command> (e.g., !pwd, !ls, !git status)")
            return

        self.log.info(f"Shell command: {cmd}")
        self.send_typing()

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.working_dir,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            self.log.warning("Shell command timed out after 30s, killing: %s", cmd)
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
            self.send_reply(f"$ {cmd}\n(timed out after 30s — process killed)")
            context_msg = (
                f"[Shell command `{cmd}` timed out after 30s and was killed]"
            )
            await self.process_message(context_msg, trigger_response=False)
            return
        output = stdout.decode("utf-8", errors="replace").strip() or "(no output)"

        display = output[:4000] + "\n... (truncated)" if len(output) > 4000 else output
        self.send_reply(
            f"$ {cmd}\n{display}",
            meta_type="tool-result",
            meta_tool="bash",
        )

        context_msg = (
            f"[I ran a shell command: `{cmd}`]\n\nOutput:\n```\n{output[:8000]}\n```"
        )
        await self.process_message(context_msg, trigger_response=False)

    async def peek_output(self, num_lines: int = 30):
        """Show recent output without adding to context."""
        output_file = self.output_dir / f"{self.session_name}.log"
        if not output_file.exists():
            self.send_reply("No output captured yet.")
            return

        lines = self._read_tail(output_file, max(num_lines, 100))
        if not lines:
            self.send_reply("Output file empty.")
            return

        status = "RUNNING" if self.processing else "IDLE"
        output = f"[{status}] Last {len(lines)} lines:\n" + "\n".join(lines)
        if len(output) > 3500:
            output = "... (truncated)\n" + output[-3500:]
        self.send_reply(output)

    def _read_tail(self, path: Path, num_lines: int) -> list[str]:
        """Read last N lines from file."""
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return []

            buffer = b""
            chunk_size = 4096
            while len(buffer.splitlines()) <= num_lines and f.tell() > 0:
                read_size = min(chunk_size, f.tell())
                f.seek(-read_size, os.SEEK_CUR)
                buffer = f.read(read_size) + buffer
                f.seek(-read_size, os.SEEK_CUR)

            lines = buffer.splitlines()
            if lines and f.tell() > 0:
                lines = lines[1:]  # Skip partial first line
            return [
                line.decode("utf-8", errors="replace") for line in lines[-num_lines:]
            ]

    # -------------------------------------------------------------------------
    # Message processing
    # -------------------------------------------------------------------------

    async def process_message(
        self,
        body: str,
        trigger_response: bool = True,
        *,
        attachments: list[Attachment] | None = None,
    ) -> None:
        """Enqueue a message for serialized processing."""
        await self._runtime.enqueue(
            body,
            list(attachments or []),
            trigger_response=trigger_response,
            scheduled=False,
            wait=True,
        )

    def answer_pending_question(
        self, answer: object, *, request_id: str | None = None
    ) -> bool:
        """Answer a pending question via XMPP meta reply."""
        return self._runtime.answer_question(answer, request_id=request_id)

    # -------------------------------------------------------------------------
    # Sibling sessions
    # -------------------------------------------------------------------------

    async def spawn_sibling_session(self, first_message: str):
        """Spawn sibling session while this one is busy."""
        if not first_message:
            return
        if not self.manager:
            self.send_reply("Session manager unavailable.")
            return

        self.send_reply("Spawning sibling session...")

        parent = self.sessions.get(self.session_name)
        engine = parent.active_engine if parent else "pi"
        model_id = parent.model_id if parent else None
        owner_jid = parent.owner_jid if parent else None
        collaborators: list[str] | None = None
        if parent and parent.room_jid:
            existing = self.sessions.list_collaborators(self.session_name)
            collaborators = [jid for jid in existing if jid != owner_jid]

        created_name = await lifecycle_create_session(
            self.manager,
            first_message,
            engine=engine,
            model_id=model_id,
            label=None,
            name_hint=f"{self.session_name}-sib",
            announce="Sibling session '{name}' (spawned from {parent}). Processing: {preview}...",
            announce_vars={"parent": self.session_name},
            dispatcher_jid=None,
            owner_jid=owner_jid,
            collaborators=collaborators,
        )
        if not created_name:
            self.send_reply("Failed to create sibling session")
            return
