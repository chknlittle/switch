"""Session bot - one XMPP bot per OpenCode/Claude session."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from src.bots.ralph_mixin import RalphMixin
from src.commands import CommandHandler
from src.db import MessageRepository, RalphLoopRepository, SessionRepository
from src.engines import get_engine_spec
from src.lifecycle.sessions import create_session as lifecycle_create_session
from src.helpers import (
    append_to_history,
    log_activity,
)
from src.runners import ClaudeRunner, OpenCodeResult, OpenCodeRunner, Question
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
        self.runner: OpenCodeRunner | ClaudeRunner | None = None
        self._run_task: asyncio.Future | None = None
        self.processing = False
        self.shutting_down = False
        self._typing_task: asyncio.Task | None = None
        self._typing_last_sent = 0.0
        self.message_queue: asyncio.Queue[tuple[str, list[Attachment]]] = asyncio.Queue()
        # When set, queued messages should be dropped and the queue drain loop
        # should stop after the current in-flight work unwinds.
        self._cancel_queue_drain = False
        # request_id -> Future(answer) where answer is either a raw user string
        # or a structured OpenCode-style answers array.
        self._pending_question_answers: dict[str, asyncio.Future] = {}
        self.attachment_store = AttachmentStore()
        self.log = logging.getLogger(f"session.{session_name}")
        self.init_ralph(RalphLoopRepository(db))
        self.commands = CommandHandler(self)
        self._engine_handlers = self._build_engine_handlers()

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

    # -------------------------------------------------------------------------
    # XMPP lifecycle
    # -------------------------------------------------------------------------

    async def on_start(self, event):
        try:
            self.send_presence()
            await self.get_roster()
            await self["xep_0280"].enable()  # type: ignore[attr-defined,union-attr]
            self.log.info("Connected")
            self.set_connected(True)
        except Exception:
            self.log.exception("Session start error")
            self.set_connected(False)

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

        # Always stop queue churn: drop any queued messages and prevent the
        # drain loop from continuing once the current run unwinds.
        self._cancel_queue_drain = True
        if not self.message_queue.empty():
            cancelled_any = True
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if self.ralph_loop:
            cancelled_any = True
            self.ralph_loop.cancel()

        if self.runner and self.processing:
            cancelled_any = True
            try:
                self.runner.cancel()
            except Exception:
                pass

        if self._run_task and not self._run_task.done():
            cancelled_any = True
            self._run_task.cancel()

        if notify and cancelled_any:
            self.send_reply("Cancelling current work...")

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

        # Stop any in-flight work and drop queued messages.
        self.cancel_operations(notify=False)
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Give the ack message (sent before hard_kill) a brief chance to flush.
        await asyncio.sleep(0.25)

        try:
            if self.manager:
                await self.manager.kill_session(self.session_name)
            else:
                # Fallback: local cleanup if manager is unavailable.
                from src.helpers import delete_xmpp_account, kill_tmux_session

                username = self.boundjid.user
                delete_xmpp_account(username, self.ejabberd_ctl, self.xmpp_domain, self.log)
                kill_tmux_session(self.session_name)
                self.sessions.close(self.session_name)
        finally:
            try:
                self.disconnect()
            except Exception:
                pass

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

    def _extract_switch_meta(self, msg) -> tuple[str | None, dict[str, str] | None, object | None]:
        """Extract Switch message meta extension (best-effort)."""
        try:
            for child in getattr(msg, "xml", []) or []:
                if getattr(child, "tag", None) != f"{{{SWITCH_META_NS}}}meta":
                    continue
                attrs = dict(getattr(child, "attrib", {}) or {})
                meta_type = attrs.get("type")

                payload_obj: object | None = None
                payload = child.find(f"{{{SWITCH_META_NS}}}payload")
                if payload is not None and (payload.get("format") or "").lower() == "json":
                    raw = (payload.text or "").strip()
                    if raw:
                        payload_obj = json.loads(raw)

                return meta_type, attrs, payload_obj
        except Exception:
            return None, None, None
        return None, None, None

    async def on_message(self, msg):
        try:
            await self._handle_session_message(msg)
        except asyncio.CancelledError:
            # Cancels are expected (e.g., /cancel, /kill).
            return
        except Exception:
            self.log.exception("Session message error")
            self.send_reply("Error handling message")

    async def _handle_session_message(self, msg):
        if msg["type"] not in ("chat", "normal"):
            return

        if self.shutting_down:
            return

        sender = str(msg["from"].bare)
        dispatcher_jid = f"oc@{self.xmpp_domain}"
        if sender not in (self.xmpp_recipient, dispatcher_jid):
            return

        meta_type, meta_attrs, meta_payload = self._extract_switch_meta(msg)

        # Allow meta-only messages (e.g., button-based question replies).
        body = (msg["body"] or "").strip()
        is_scheduled = sender == dispatcher_jid

        attachments: list[Attachment] = []
        try:
            urls = self._extract_attachment_urls(msg, body)
            if urls:
                attachments = await self.attachment_store.download_images(
                    self.session_name, urls
                )
                if attachments:
                    self._send_attachment_meta(attachments)
                    body = self._strip_urls_from_body(body, urls)
        except Exception:
            attachments = []

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

        if body.startswith("@"):
            body = "/" + body[1:]

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

        # Busy handling - queue messages instead of rejecting
        if self.processing:
            if is_scheduled:
                return
            if body.startswith("+") and self.manager:
                await self.spawn_sibling_session(body[1:].strip())
                return
            # Queue the message for later processing
            await self.message_queue.put((body, attachments))
            queue_size = self.message_queue.qsize()
            self.send_reply(f"Queued ({queue_size} pending)")
            return

        await self._process_with_queue(body, attachments)

    _URL_RE = re.compile(r"https?://[^\s<>\]\)\}]+", re.IGNORECASE)

    def _extract_attachment_urls(self, msg, body: str) -> list[str]:
        urls: list[str] = []

        # jabber:x:oob and similar: scan all descendants for <url> elements.
        try:
            for el in getattr(msg, "xml", []) or []:
                for child in list(el.iter()):
                    tag = getattr(child, "tag", "")
                    if tag.endswith("}url") or tag == "url":
                        text = (getattr(child, "text", None) or "").strip()
                        if text.startswith("http"):
                            urls.append(text)
        except Exception:
            pass

        for m in self._URL_RE.finditer(body or ""):
            raw = m.group(0)
            url = raw.rstrip(".,;:!?")
            if url.startswith("http"):
                urls.append(url)

        seen: set[str] = set()
        out: list[str] = []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return out

    def _strip_urls_from_body(self, body: str, urls: list[str]) -> str:
        if not body:
            return body
        out = body
        for u in urls:
            out = out.replace(u, "")
        return " ".join(out.split()).strip()

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

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.working_dir,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace").strip() or "(no output)"

            display = (
                output[:4000] + "\n... (truncated)" if len(output) > 4000 else output
            )
            self.send_reply(
                f"$ {cmd}\n{display}",
                meta_type="tool-result",
                meta_tool="bash",
            )

            context_msg = f"[I ran a shell command: `{cmd}`]\n\nOutput:\n```\n{output[:8000]}\n```"
            await self.process_message(context_msg, trigger_response=False)

        except asyncio.TimeoutError:
            self.send_reply(f"$ {cmd}\n(timed out after 30s)")
        except Exception as e:
            self.send_reply(f"$ {cmd}\nError: {e}")

    async def peek_output(self, num_lines: int = 30):
        """Show recent output without adding to context."""
        output_file = self.output_dir / f"{self.session_name}.log"
        if not output_file.exists():
            self.send_reply("No output captured yet.")
            return

        try:
            lines = self._read_tail(output_file, max(num_lines, 100))
            if not lines:
                self.send_reply("Output file empty.")
                return

            status = "RUNNING" if self.processing else "IDLE"
            output = f"[{status}] Last {len(lines)} lines:\n" + "\n".join(lines)
            if len(output) > 3500:
                output = "... (truncated)\n" + output[-3500:]
            self.send_reply(output)
        except Exception as e:
            self.send_reply(f"Error reading output: {e}")

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

    async def _process_with_queue(self, body: str, attachments: list[Attachment] | None):
        """Process a message and then drain any queued messages."""
        # New user work starts a new drain window.
        self._cancel_queue_drain = False
        await self.process_message(body, attachments=attachments)

        # If a /cancel happened while this message was in-flight, don't churn
        # through any queued messages.
        if self._cancel_queue_drain:
            self._cancel_queue_drain = False
            return

        # Process any queued messages
        while not self.message_queue.empty():
            try:
                if self._cancel_queue_drain:
                    break
                next_body, next_attachments = self.message_queue.get_nowait()

                # /cancel may have fired after we dequeued but before we start.
                if self._cancel_queue_drain:
                    break

                queue_remaining = self.message_queue.qsize()
                if queue_remaining > 0:
                    self.send_reply(
                        f"Processing queued message ({queue_remaining} remaining)..."
                    )
                else:
                    self.send_reply("Processing queued message...")
                await self.process_message(next_body, attachments=next_attachments)

                if self._cancel_queue_drain:
                    break
            except asyncio.QueueEmpty:
                break

        # Reset after drain loop ends.
        if self._cancel_queue_drain:
            self._cancel_queue_drain = False

    def _augment_prompt_with_attachments(
        self, body: str, attachments: list[Attachment] | None
    ) -> str:
        if not attachments:
            return body
        lines: list[str] = [body.strip(), "", "User attached image(s):"]
        for a in attachments:
            lines.append(f"- {a.local_path}")
        return "\n".join(lines).strip()

    async def process_message(
        self,
        body: str,
        trigger_response: bool = True,
        *,
        attachments: list[Attachment] | None = None,
    ):
        """Send message to agent and relay response."""
        if self.shutting_down:
            return
        self.processing = True
        if trigger_response:
            self._start_typing_loop()

        try:
            session = self.sessions.get(self.session_name)
            if not session:
                self.send_reply("Session not found in database.")
                return

            self.sessions.update_last_active(self.session_name)
            body_for_history = self._augment_prompt_with_attachments(body, attachments)
            append_to_history(body_for_history, self.working_dir, session.claude_session_id)
            log_activity(body, session=self.session_name, source="xmpp")
            self.messages.add(
                self.session_name, "user", body_for_history, session.active_engine
            )

            if not trigger_response:
                return

            handler = self.engine_handler_for(session.active_engine)
            if not handler:
                self.send_reply(f"Unknown engine '{session.active_engine}'.")
                return
            # ensure_future accepts any awaitable and gives us a cancellable future.
            self._run_task = asyncio.ensure_future(
                handler.run(body, session, attachments)
            )
            await self._run_task

        except asyncio.CancelledError:
            self.log.info("Message processing cancelled")
            return
        except Exception as e:
            self.log.exception("Error")
            self.send_reply(f"Error: {e}")
        finally:
            self._run_task = None
            self.processing = False
            self._stop_typing_loop()

    def _maybe_send_typing(self, *, min_interval_s: float = 5.0) -> None:
        """Best-effort composing keepalive (rate-limited)."""
        if self.shutting_down:
            return
        now = time.monotonic()
        if now - self._typing_last_sent < min_interval_s:
            return
        self._typing_last_sent = now
        self.send_typing()

    async def _typing_loop(self, *, interval_s: float = 15.0) -> None:
        """Periodically refresh composing while a run is in-flight."""
        try:
            while self.processing and not self.shutting_down:
                # Many clients clear "typing" after ~10-30s unless refreshed.
                self._maybe_send_typing(min_interval_s=0.0)
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

    def _start_typing_loop(self) -> None:
        # If a prior loop is still running, keep it.
        if self._typing_task and not self._typing_task.done():
            self._maybe_send_typing(min_interval_s=0.0)
            return

        # Send immediately, then keepalive in the background.
        self._maybe_send_typing(min_interval_s=0.0)
        self._typing_task = asyncio.create_task(self._typing_loop())

    def _stop_typing_loop(self) -> None:
        task = self._typing_task
        self._typing_task = None
        if task and not task.done():
            task.cancel()

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
        self.runner = OpenCodeRunner(
            self.working_dir,
            self.output_dir,
            self.session_name,
            model=model_id,
            reasoning_mode=session.reasoning_mode,
            agent=agent or session.opencode_agent,
            question_callback=question_callback,
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
        self.runner = ClaudeRunner(self.working_dir, self.output_dir, self.session_name)
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
                    self._maybe_send_typing(min_interval_s=5.0)
            elif event_type == "result":
                self._send_result(
                    tool_summaries, response_parts, content, session.active_engine
                )
            elif event_type == "error":
                self._stop_typing_loop()
                self.send_reply(f"Error: {content}")
            elif event_type == "cancelled":
                self._stop_typing_loop()
                self.send_reply("Cancelled.")

    async def _run_opencode(
        self, body: str, session, attachments: list[Attachment] | None
    ):
        """Run OpenCode and handle events."""
        body = self._augment_prompt_with_attachments(body, attachments)
        # Set up question callback for handling AI questions
        question_callback = self._create_question_callback()

        self.runner = OpenCodeRunner(
            self.working_dir,
            self.output_dir,
            self.session_name,
            model=session.model_id,
            reasoning_mode=session.reasoning_mode,
            agent=session.opencode_agent,
            question_callback=question_callback,
        )
        response_parts: list[str] = []
        tool_summaries: list[str] = []
        accumulated = ""
        last_progress_at = 0
        event_count = 0

        self.log.info(f"Starting OpenCode run for: {body[:50]}...")
        try:
            async for event_type, content in self.runner.run(
                body, session.opencode_session_id, attachments=None
            ):
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
                        self._maybe_send_typing(min_interval_s=5.0)
                    elif len(tool_summaries) - last_progress_at >= 8:
                        last_progress_at = len(tool_summaries)
                        self.send_reply(
                            f"... {' '.join(tool_summaries[-3:])}",
                            meta_type="tool",
                            meta_tool=self._infer_meta_tool_from_summary(tool_summaries[-1]),
                        )
                        self._maybe_send_typing(min_interval_s=5.0)
                elif event_type == "question" and isinstance(content, Question):
                    # Question is being handled by callback, just log it
                    self.log.info(f"Question asked: {content.request_id}")
                elif event_type == "result" and isinstance(content, OpenCodeResult):
                    if not response_parts and content.text:
                        response_parts.append(content.text)
                    model_short = (
                        session.model_id.split("/")[-1] if session.model_id else "?"
                    )
                    stats = {
                        "engine": "opencode",
                        "model": model_short,
                        "tokens_in": content.tokens_in,
                        "tokens_out": content.tokens_out,
                        "tokens_reasoning": content.tokens_reasoning,
                        "tokens_cache_read": content.tokens_cache_read,
                        "tokens_cache_write": content.tokens_cache_write,
                        "cost_usd": f"{content.cost:.3f}",
                        "duration_s": f"{content.duration_s:.1f}",
                        "summary": (
                            f"[{model_short} {content.tokens_in}/{content.tokens_out} tok"
                            f" r{content.tokens_reasoning} c{content.tokens_cache_read}/{content.tokens_cache_write}"
                            f" ${content.cost:.3f} {content.duration_s:.1f}s]"
                        ),
                    }
                    self._send_result(
                        tool_summaries, response_parts, stats, session.active_engine
                    )
                elif event_type == "error":
                    self.log.warning(f"OpenCode error event: {content}")
                    self._stop_typing_loop()
                    self.send_reply(f"Error: {content}")
                elif event_type == "cancelled":
                    self._stop_typing_loop()
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
        """Answer a pending question. Called from message handler."""
        if not self._pending_question_answers:
            return False

        # Default to the most recent pending question.
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
        self._stop_typing_loop()
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
