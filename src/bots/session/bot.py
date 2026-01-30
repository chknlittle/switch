"""Session bot - one XMPP bot per OpenCode/Claude session."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from src.bots.ralph_mixin import RalphMixin
from src.core.session_runtime import SessionRuntime
from src.core.session_runtime.ports import (
    AttachmentPromptPort,
    HistoryPort,
    MessageStorePort,
    ReplyPort,
    RunnerFactoryPort,
    SessionState,
    SessionStorePort,
)
from src.bots.session.inbound import (
    extract_attachment_urls,
    extract_switch_meta,
    normalize_leading_at,
    strip_urls_from_body,
)
from src.bots.session.typing import TypingIndicator
from src.commands import CommandHandler
from src.db import MessageRepository, RalphLoopRepository, SessionRepository
from src.engines import get_engine_spec
from src.lifecycle.sessions import create_session as lifecycle_create_session
from src.helpers import (
    append_to_history,
    log_activity,
)
from src.runners import Question, Runner, create_runner
from src.runners.opencode.config import OpenCodeConfig
from src.attachments import Attachment, AttachmentStore
from src.utils import SWITCH_META_NS, BaseXMPPBot, build_message_meta

if TYPE_CHECKING:
    import sqlite3

    from src.db import Session
    from src.manager import SessionManager


@dataclass(frozen=True)
class EngineHandler:
    name: str
    run: Callable[[str, "Session", list[Attachment] | None], Awaitable[None]]
    reset: Callable[[], None]
    supports_reasoning: bool


class SessionBot(RalphMixin, BaseXMPPBot):
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
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.xmpp_recipient = xmpp_recipient
        self.xmpp_domain = xmpp_domain
        self.xmpp_server = xmpp_server
        self.ejabberd_ctl = ejabberd_ctl
        self.manager = manager
        self.runner: Runner | None = None
        self.processing = False
        self.shutting_down = False
        self._typing = TypingIndicator(
            send_typing=self.send_typing,
            is_active=lambda: self.processing,
            is_shutting_down=lambda: self.shutting_down,
        )

        self._runtime = self._build_runtime()

        # Legacy attributes kept for compatibility with older helper paths.
        # New message processing uses SessionRuntime.
        self._run_task: asyncio.Future | None = None
        self._pending_question_answers: dict[str, asyncio.Future] = {}
        self.attachment_store = AttachmentStore()
        self.init_ralph(RalphLoopRepository(self.db))
        self.commands = CommandHandler(self)
        self._engine_handlers = self._build_engine_handlers()

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

    # -------------------------------------------------------------------------
    # Runtime wiring
    # -------------------------------------------------------------------------

    class _ReplyAdapter(ReplyPort):
        def __init__(self, bot: "SessionBot"):
            self._bot = bot

        def send_reply(
            self,
            text: str,
            *,
            meta_type: str | None = None,
            meta_tool: str | None = None,
            meta_attrs: dict[str, str] | None = None,
            meta_payload: object | None = None,
        ) -> None:
            self._bot.send_reply(
                text,
                meta_type=meta_type,
                meta_tool=meta_tool,
                meta_attrs=meta_attrs,
                meta_payload=meta_payload,
            )

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
                opencode_session_id=s.opencode_session_id,
                opencode_agent=s.opencode_agent,
                model_id=s.model_id,
                reasoning_mode=s.reasoning_mode,
            )

        def update_last_active(self, name: str) -> None:
            self._repo.update_last_active(name)

        def update_claude_session_id(self, name: str, session_id: str) -> None:
            self._repo.update_claude_session_id(name, session_id)

        def update_opencode_session_id(self, name: str, session_id: str) -> None:
            self._repo.update_opencode_session_id(name, session_id)

    class _MessagesAdapter(MessageStorePort):
        def __init__(self, repo: MessageRepository):
            self._repo = repo

        def add(self, session_name: str, role: str, content: str, engine: str) -> None:
            self._repo.add(session_name, role, content, engine)

    class _RunnerFactoryAdapter(RunnerFactoryPort):
        def create(
            self,
            engine: str,
            *,
            working_dir: str,
            output_dir: Path,
            session_name: str,
            opencode_config: OpenCodeConfig | None = None,
        ) -> Runner:
            return create_runner(
                engine,
                working_dir=working_dir,
                output_dir=output_dir,
                session_name=session_name,
                opencode_config=opencode_config,
            )

    class _HistoryAdapter(HistoryPort):
        def append_to_history(self, message: str, working_dir: str, claude_session_id: str | None) -> None:
            append_to_history(message, working_dir, claude_session_id)

        def log_activity(self, message: str, *, session: str, source: str) -> None:
            log_activity(message, session=session, source=source)

    class _PromptAdapter(AttachmentPromptPort):
        def augment_prompt(self, body: str, attachments: list[Attachment] | None) -> str:
            if not attachments:
                return (body or "").strip()
            lines: list[str] = [(body or "").strip(), "", "User attached image(s):"]
            for a in attachments:
                lines.append(f"- {a.local_path}")
            return "\n".join(lines).strip()

    def _build_runtime(self) -> SessionRuntime:
        return SessionRuntime(
            session_name=self.session_name,
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            sessions=self._SessionsAdapter(self.sessions),
            messages=self._MessagesAdapter(self.messages),
            reply=self._ReplyAdapter(self),
            typing=self._typing,
            runner_factory=self._RunnerFactoryAdapter(),
            history=self._HistoryAdapter(),
            prompt=self._PromptAdapter(),
            infer_meta_tool_from_summary=self._infer_meta_tool_from_summary,
            on_processing_changed=lambda active: setattr(self, "processing", active),
        )

    # -------------------------------------------------------------------------
    # XMPP lifecycle
    # -------------------------------------------------------------------------

    async def on_start(self, event):
        await self.guard(self._on_start(event), context="session.on_start")

    async def _on_start(self, event):
        self.send_presence()
        await self.get_roster()
        await self["xep_0280"].enable()  # type: ignore[attr-defined,union-attr]
        self.log.info("Connected")
        self.set_connected(True)

    def on_disconnected(self, event):
        self.set_connected(False)
        if self.shutting_down:
            self.log.info("Disconnected during shutdown; not reconnecting")
            return
        self.log.warning("Disconnected, reconnecting...")
        asyncio.ensure_future(self._reconnect())

    def _build_engine_handlers(self) -> dict[str, EngineHandler]:
        claude_spec = get_engine_spec("claude")
        opencode_spec = get_engine_spec("opencode")
        return {
            "claude": EngineHandler(
                name="claude",
                run=self._run_claude,
                reset=lambda: self.sessions.reset_claude_session(self.session_name),
                supports_reasoning=claude_spec.supports_reasoning
                if claude_spec
                else False,
            ),
            "opencode": EngineHandler(
                name="opencode",
                run=self._run_opencode,
                reset=lambda: self.sessions.reset_opencode_session(self.session_name),
                supports_reasoning=opencode_spec.supports_reasoning
                if opencode_spec
                else False,
            ),
        }

    def engine_handler_for(self, engine: str) -> EngineHandler | None:
        return self._engine_handlers.get(engine)

    async def _reconnect(self):
        await asyncio.sleep(5)
        self.connect()

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

        max_len = 100000
        target = recipient or self.xmpp_recipient
        if len(text) <= max_len:
            msg = self.make_message(mto=target, mbody=text, mtype="chat")
            msg["chat_state"] = "active"

            if meta_type:
                meta = build_message_meta(
                    meta_type,
                    meta_tool=meta_tool,
                    meta_attrs=meta_attrs,
                    meta_payload=meta_payload,
                )
                msg.xml.append(meta)

            msg.send()
            return

        parts = self._split_message(text, max_len)
        total = len(parts)
        for i, part in enumerate(parts, 1):
            header = f"[{i}/{total}]\n" if i > 1 else ""
            footer = f"\n[{i}/{total}]" if i < total else ""
            body = header + part + footer if total > 1 else part
            msg = self.make_message(mto=target, mbody=body, mtype="chat")
            msg["chat_state"] = "active" if i == total else "composing"

            if meta_type:
                meta = build_message_meta(
                    meta_type,
                    meta_tool=meta_tool,
                    meta_attrs=meta_attrs,
                    meta_payload=meta_payload,
                )
                msg.xml.append(meta)

            msg.send()

    @staticmethod
    def _infer_meta_tool_from_summary(summary: str) -> str | None:
        """Best-effort mapping from tool summary text to meta.tool."""
        # OpenCode tool summaries look like: "[tool:bash ...]".
        if summary.startswith("[tool:"):
            end = summary.find("]")
            head = summary[:end] if end != -1 else summary
            inner = head[len("[tool:") :]
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

        if self.ralph_loop:
            cancelled_any = True
            self.ralph_loop.cancel()

        # Best-effort: also cancel any ad-hoc runner paths.
        if self.runner and self.processing:
            cancelled_any = True
            self.runner.cancel()

        return cancelled_any

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
        await self.guard(self._handle_session_message(msg), context="session.on_message")

    async def _handle_session_message(self, msg):
        if msg["type"] not in ("chat", "normal"):
            return

        if self.shutting_down:
            return

        sender = str(msg["from"].bare)
        dispatcher_jid = f"oc@{self.xmpp_domain}"
        if sender not in (self.xmpp_recipient, dispatcher_jid):
            return

        meta_type, meta_attrs, meta_payload = extract_switch_meta(msg, meta_ns=SWITCH_META_NS)

        # Allow meta-only messages (e.g., button-based question replies).
        body = (msg["body"] or "").strip()
        is_scheduled = sender == dispatcher_jid

        attachments: list[Attachment] = []
        urls = extract_attachment_urls(msg, body)
        if urls:
            attachments = await self.attachment_store.download_images(self.session_name, urls)
            if attachments:
                self._send_attachment_meta(attachments)
                body = strip_urls_from_body(body, urls)

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

        # Shell commands
        if body.startswith("!"):
            await self.run_shell_command(body[1:].strip())
            return

        # Check for pending question answers first
        if self.answer_pending_question(body):
            self.log.info(f"Answered pending question with: {body[:50]}...")
            return

        # Scheduled messages are best-effort; drop them if we're already running.
        if is_scheduled and self.processing:
            return

        # If we're busy, allow +spawn to fork work instead of queuing it.
        if self.processing and (not is_scheduled) and body.startswith("+") and self.manager:
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
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode("utf-8", errors="replace").strip() or "(no output)"

        display = output[:4000] + "\n... (truncated)" if len(output) > 4000 else output
        self.send_reply(
            f"$ {cmd}\n{display}",
            meta_type="tool-result",
            meta_tool="bash",
        )

        context_msg = f"[I ran a shell command: `{cmd}`]\n\nOutput:\n```\n{output[:8000]}\n```"
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

    def _augment_prompt_with_attachments(
        self, body: str, attachments: list[Attachment] | None
    ) -> str:
        if not attachments:
            return body
        lines: list[str] = [body.strip(), "", "User attached image(s):"]
        for a in attachments:
            lines.append(f"- {a.local_path}")
        return "\n".join(lines).strip()

    async def _process_one_message_now(
        self,
        body: str,
        *,
        trigger_response: bool,
        attachments: list[Attachment] | None,
    ) -> None:
        """Process exactly one message (no queueing, no typing/processing flags)."""
        session = self.sessions.get(self.session_name)
        if not session:
            self.send_reply("Session not found in database.")
            return

        self.sessions.update_last_active(self.session_name)
        body_for_history = self._augment_prompt_with_attachments(body, attachments)
        append_to_history(body_for_history, self.working_dir, session.claude_session_id)
        log_activity(body, session=self.session_name, source="xmpp")
        self.messages.add(self.session_name, "user", body_for_history, session.active_engine)

        if not trigger_response:
            return

        handler = self.engine_handler_for(session.active_engine)
        if not handler:
            self.send_reply(f"Unknown engine '{session.active_engine}'.")
            return

        # ensure_future accepts any awaitable and gives us a cancellable future.
        self._run_task = asyncio.ensure_future(handler.run(body, session, attachments))
        try:
            await self._run_task
        finally:
            self._run_task = None

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

    async def run_opencode_one_shot(
        self,
        prompt: str,
        model_id: str,
        working_dir: str | None = None,
    ) -> None:
        """Run a single OpenCode prompt with a temporary model override."""
        session = self.sessions.get(self.session_name)
        if not session:
            self.send_reply("Session not found in database.")
            return

        original_engine = session.active_engine
        original_model = session.model_id
        original_working_dir = self.working_dir

        self.sessions.update_engine(self.session_name, "opencode")
        self.sessions.update_model(self.session_name, model_id)
        if working_dir:
            self.working_dir = working_dir
        try:
            await self.process_message(prompt)
        finally:
            self.sessions.update_engine(self.session_name, original_engine)
            self.sessions.update_model(self.session_name, original_model)
            self.working_dir = original_working_dir

    async def run_opencode_capture(
        self,
        prompt: str,
        model_id: str,
        working_dir: str | None = None,
        agent: str | None = None,
    ) -> str:
        """Run OpenCode and capture the final response text.

        Args:
            prompt: The prompt to send
            model_id: Model to use
            working_dir: Optional working directory override
            agent: Optional agent override (use "summary" for tool-less extraction)
        """
        session = self.sessions.get(self.session_name)
        if not session:
            self.send_reply("Session not found in database.")
            return ""

        original_working_dir = self.working_dir
        if working_dir:
            self.working_dir = working_dir

        self.processing = True
        question_callback = self._create_question_callback()
        self.runner = create_runner(
            "opencode",
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            session_name=self.session_name,
            opencode_config=OpenCodeConfig(
                model=model_id,
                reasoning_mode=session.reasoning_mode,
                agent=agent or session.opencode_agent,
                question_callback=question_callback,
            ),
        )

        response_text = ""

        try:
            async for event_type, content in self.runner.run(
                prompt, session.opencode_session_id
            ):
                if event_type == "session_id" and isinstance(content, str):
                    self.sessions.update_opencode_session_id(self.session_name, content)
                elif event_type == "text" and isinstance(content, str):
                    response_text += content
                elif event_type == "error":
                    self.send_reply(f"Error: {content}")
                    break
        finally:
            self.processing = False
            self.working_dir = original_working_dir

        return response_text.strip()

    async def _run_claude(
        self, body: str, session, attachments: list[Attachment] | None
    ):
        """Run Claude and handle events."""
        body = self._augment_prompt_with_attachments(body, attachments)
        self.runner = create_runner(
            "claude",
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            session_name=self.session_name,
        )
        response_parts: list[str] = []
        tool_summaries: list[str] = []
        last_progress_at = 0

        async for event_type, content in self.runner.run(
            body, session.claude_session_id
        ):
            if event_type == "session_id":
                session_id = content if isinstance(content, str) else None
                if session_id:
                    self.sessions.update_claude_session_id(self.session_name, session_id)
            elif event_type == "text":
                text = content if isinstance(content, str) else None
                if text is not None:
                    response_parts = [text]
            elif event_type == "tool":
                summary = content if isinstance(content, str) else None
                if summary is None:
                    continue
                tool_summaries.append(summary)
                if len(tool_summaries) - last_progress_at >= 8:
                    last_progress_at = len(tool_summaries)
                    self.send_reply(
                        f"... {' '.join(tool_summaries[-3:])}",
                        meta_type="tool",
                        meta_tool=self._infer_meta_tool_from_summary(tool_summaries[-1]),
                    )
                    # Progress messages often clear typing indicators; reassert.
                    self._typing.maybe_send(min_interval_s=5.0)
            elif event_type == "result":
                self._send_result(
                    tool_summaries, response_parts, content, session.active_engine
                )
            elif event_type == "error":
                self._typing.stop()
                self.send_reply(f"Error: {content}")
            elif event_type == "cancelled":
                self._typing.stop()
                self.send_reply("Cancelled.")

    async def _run_opencode(
        self, body: str, session, attachments: list[Attachment] | None
    ):
        """Run OpenCode and handle events."""
        body = self._augment_prompt_with_attachments(body, attachments)
        # Set up question callback for handling AI questions
        question_callback = self._create_question_callback()

        self.runner = create_runner(
            "opencode",
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            session_name=self.session_name,
            opencode_config=OpenCodeConfig(
                model=session.model_id,
                reasoning_mode=session.reasoning_mode,
                agent=session.opencode_agent,
                question_callback=question_callback,
            ),
        )
        response_parts: list[str] = []
        tool_summaries: list[str] = []
        accumulated = ""
        last_progress_at = 0
        event_count = 0

        self.log.info(f"Starting OpenCode run for: {body[:50]}...")
        try:
            async for event_type, content in self.runner.run(body, session.opencode_session_id):
                if self.shutting_down:
                    return
                event_count += 1
                self.log.debug(f"OpenCode event #{event_count}: {event_type}")

                if event_type == "session_id":
                    session_id = content if isinstance(content, str) else None
                    if session_id:
                        self.sessions.update_opencode_session_id(
                            self.session_name, session_id
                        )
                elif event_type == "text" and isinstance(content, str):
                    accumulated += content
                    response_parts = [accumulated]
                elif event_type == "tool" and isinstance(content, str):
                    tool_summaries.append(content)
                    # Emit progress a bit more eagerly for bash (users tend to
                    # care about command execution), and at least once early so
                    # short runs don't look stalled.
                    is_bash = content.startswith("[tool:bash")
                    if is_bash or len(tool_summaries) == 1:
                        last_progress_at = len(tool_summaries)
                        self.send_reply(
                            f"... {content}",
                            meta_type="tool",
                            meta_tool=self._infer_meta_tool_from_summary(content),
                        )
                        # Tool/progress messages often clear typing indicators.
                        self._typing.maybe_send(min_interval_s=5.0)
                    elif len(tool_summaries) - last_progress_at >= 8:
                        last_progress_at = len(tool_summaries)
                        self.send_reply(
                            f"... {' '.join(tool_summaries[-3:])}",
                            meta_type="tool",
                            meta_tool=self._infer_meta_tool_from_summary(tool_summaries[-1]),
                        )
                        self._typing.maybe_send(min_interval_s=5.0)
                elif event_type == "question" and isinstance(content, Question):
                    # Question is being handled by callback, just log it
                    self.log.info(f"Question asked: {content.request_id}")
                elif event_type == "result" and isinstance(content, dict):
                    if not response_parts:
                        text = content.get("text")
                        if isinstance(text, str) and text:
                            response_parts.append(text)
                    self._send_result(
                        tool_summaries, response_parts, content, session.active_engine
                    )
                elif event_type == "error":
                    self.log.warning(f"OpenCode error event: {content}")
                    self._typing.stop()
                    self.send_reply(f"Error: {content}")
                elif event_type == "cancelled":
                    self._typing.stop()
                    self.send_reply("Cancelled.")

            self.log.info(f"OpenCode run completed after {event_count} events")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.log.exception(f"Exception in _run_opencode loop: {e}")
            self.send_reply(f"Error during OpenCode run: {e}")

    def _create_question_callback(self):
        """Create a callback for handling AI questions via XMPP."""

        async def question_callback(question: Question) -> list[list[str]]:
            """Handle a question from the AI by asking the user via XMPP."""
            question_text = self._format_question(question)
            self.send_reply(
                question_text,
                meta_type="question",
                meta_tool="question",
                meta_attrs={
                    "version": "1",
                    "engine": "opencode",
                    "request_id": question.request_id,
                    "question_count": str(len(question.questions or [])),
                },
                meta_payload={
                    "version": 1,
                    "engine": "opencode",
                    "request_id": question.request_id,
                    "questions": question.questions,
                },
            )

            # Create a future to wait for the answer
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending_question_answers[question.request_id] = future

            try:
                # Wait for user response with timeout
                answer = await asyncio.wait_for(future, timeout=300)  # 5 min timeout
                parsed = self._parse_question_answer(question, answer)
                return parsed
            except asyncio.TimeoutError:
                self.send_reply("[Question timed out - proceeding without answer]")
                raise
            finally:
                self._pending_question_answers.pop(question.request_id, None)

        return question_callback

    def _parse_question_answer(self, question: Question, answer: object) -> list[list[str]]:
        """Parse a user answer into OpenCode's expected answers shape."""
        if isinstance(answer, list):
            # Allow structured answers: [["Option A"], ["Option B"]]
            return answer  # type: ignore[return-value]

        text = str(answer or "").strip()
        qs = question.questions or []
        if not qs:
            return []

        # Split into one segment per question when possible.
        segments: list[str] = []
        if "\n" in text:
            segments = [s.strip() for s in text.splitlines() if s.strip()]
        if not segments and ";" in text and len(qs) > 1:
            segments = [s.strip() for s in text.split(";") if s.strip()]
        if not segments:
            segments = [text]

        answers: list[list[str]] = []
        for idx, q in enumerate(qs):
            seg = segments[idx] if idx < len(segments) else (segments[0] if segments else "")
            options = q.get("options") if isinstance(q, dict) else None
            if not isinstance(options, list) or not options:
                answers.append([seg] if seg else [])
                continue

            labels: list[str] = []
            for opt in options:
                if isinstance(opt, dict):
                    lab = str(opt.get("label", "") or "").strip()
                    if lab:
                        labels.append(lab)

            chosen: list[str] = []

            # If the whole segment matches a label (including spaces), accept it.
            seg_norm = seg.strip().lower()
            direct = next((lab for lab in labels if lab.lower() == seg_norm), None)
            if direct:
                answers.append([direct])
                continue

            for tok in re.split(r"[\s,]+", seg.strip()):
                if not tok:
                    continue
                if tok.isdigit():
                    n = int(tok)
                    if 1 <= n <= len(labels):
                        chosen.append(labels[n - 1])
                    continue
                match = next((lab for lab in labels if lab.lower() == tok.lower()), None)
                if match:
                    chosen.append(match)

            # De-dupe while preserving order.
            seen: set[str] = set()
            chosen = [x for x in chosen if not (x in seen or seen.add(x))]
            answers.append(chosen)

        return answers

    def _format_question(self, question: Question) -> str:
        """Format a Question object for display to user."""
        parts: list[str] = []
        parts.append("[Question]")
        for q_idx, q in enumerate(question.questions or [], 1):
            if not isinstance(q, dict):
                continue
            header = str(q.get("header", "") or "").strip()
            text = str(q.get("question", "") or "").strip()
            options = q.get("options", [])

            if header:
                parts.append(f"{q_idx}) {header}")
            elif len(question.questions or []) > 1:
                parts.append(f"{q_idx})")
            if text:
                parts.append(text)

            if isinstance(options, list) and options:
                parts.append("Options:")
                for i, opt in enumerate(options, 1):
                    if not isinstance(opt, dict):
                        continue
                    label = str(opt.get("label", f"Option {i}") or f"Option {i}").strip()
                    desc = str(opt.get("description", "") or "").strip()
                    parts.append(f"  {i}) {label}" + (f" - {desc}" if desc else ""))

        parts.append("Reply with option number(s) (e.g., '1' or '1,2') or label text.")
        return "\n".join([p for p in parts if p])

    def answer_pending_question(self, answer: object, *, request_id: str | None = None) -> bool:
        """Answer a pending question (OpenCode server questions)."""
        if self._runtime.answer_question(answer, request_id=request_id):
            return True

        # Back-compat: allow older ad-hoc question flows.
        if not self._pending_question_answers:
            return False
        rid = request_id or list(self._pending_question_answers.keys())[-1]
        future = self._pending_question_answers.get(rid)
        if future and not future.done():
            future.set_result(answer)
            return True
        return False

    def _send_result(
        self,
        tool_summaries: list[str],
        response_parts: list[str],
        stats: object,
        engine: str,
    ):
        """Format and send final result."""
        # Ensure we don't keep sending composing after the final message.
        self._typing.stop()
        parts = []

        show_tools = os.getenv("SWITCH_SEND_TOOL_SUMMARIES", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        show_stats = os.getenv("SWITCH_SEND_RUN_STATS", "1").lower() not in (
            "0",
            "false",
            "no",
        )

        if show_tools and tool_summaries:
            tools = " ".join(tool_summaries[:5])
            if len(tool_summaries) > 5:
                tools += f" +{len(tool_summaries) - 5}"
            parts.append(tools)
        if response_parts:
            parts.append(response_parts[-1])

        meta_type = None
        meta_attrs: dict[str, str] | None = None
        stats_text: str | None = None

        if isinstance(stats, str):
            stats_text = stats
        elif isinstance(stats, dict):
            # Convert all values to strings for XML attributes.
            meta_type = "run-stats"
            meta_attrs = {}
            for k, v in stats.items():
                if v is None:
                    continue
                meta_attrs[str(k)] = str(v)
            stats_text = stats.get("summary") if isinstance(stats.get("summary"), str) else None

        # Deliberately do not append the run-stats summary to the message body.
        # We send stats via the `urn:switch:message-meta` extension so clients
        # can render it without duplicating a plaintext footer.

        self.send_reply(
            "\n\n".join([p for p in parts if p]),
            meta_type=meta_type,
            meta_attrs=meta_attrs,
        )
        self.messages.add(
            self.session_name,
            "assistant",
            response_parts[-1] if response_parts else "",
            engine,
        )

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
        engine = parent.active_engine if parent else "opencode"
        agent = parent.opencode_agent if parent else "bridge"
        model_id = parent.model_id if parent else None

        created_name = await lifecycle_create_session(
            self.manager,
            first_message,
            engine=engine,
            opencode_agent=agent,
            model_id=model_id,
            label=None,
            name_hint=f"{self.session_name}-sib",
            announce="Sibling session '{name}' (spawned from {parent}). Processing: {preview}...",
            announce_vars={"parent": self.session_name},
            dispatcher_jid=None,
        )
        if not created_name:
            self.send_reply("Failed to create sibling session")
            return
