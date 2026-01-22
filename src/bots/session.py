"""Session bot - one XMPP bot per OpenCode/Claude session."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from src.bots.ralph_mixin import RalphMixin
from src.commands import CommandHandler
from src.db import MessageRepository, RalphLoopRepository, SessionRepository
from src.helpers import (
    add_roster_subscription,
    append_to_history,
    create_tmux_session,
    log_activity,
    register_unique_account,
)
from src.runners import ClaudeRunner, OpenCodeResult, OpenCodeRunner
from src.utils import BaseXMPPBot

if TYPE_CHECKING:
    import sqlite3

    from src.manager import SessionManager


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
        self.processing = False
        self.log = logging.getLogger(f"session.{session_name}")
        self.init_ralph(RalphLoopRepository(db))
        self.commands = CommandHandler(self)

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
            await self["xep_0280"].enable()  # type: ignore[attr-defined]
            self.log.info("Connected")
            self._connected = True
        except Exception:
            self.log.exception("Session start error")
            self._connected = False

    def is_connected(self) -> bool:
        return getattr(self, "_connected", False)

    def on_disconnected(self, event):
        self.log.warning("Disconnected, reconnecting...")
        asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        await asyncio.sleep(5)
        self.connect()

    async def _self_destruct(self):
        await asyncio.sleep(1)
        self.disconnect()

    # -------------------------------------------------------------------------
    # Message sending
    # -------------------------------------------------------------------------

    def send_reply(self, text: str, recipient: str | None = None, max_len: int = 3500):
        """Send message, splitting if needed."""
        target = recipient or self.xmpp_recipient
        if len(text) <= max_len:
            msg = self.make_message(mto=target, mbody=text, mtype="chat")
            msg["chat_state"] = "active"
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
            msg.send()

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
        try:
            await self._handle_message(msg)
        except Exception:
            self.log.exception("Session message error")
            self.send_reply("Error handling message")

    async def _handle_message(self, msg):
        if msg["type"] not in ("chat", "normal") or not msg["body"]:
            return

        sender = str(msg["from"].bare)
        dispatcher_jid = f"oc@{self.xmpp_domain}"
        if sender not in (self.xmpp_recipient, dispatcher_jid):
            return

        body = msg["body"].strip()
        is_scheduled = sender == dispatcher_jid

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

        # Busy handling
        if self.processing:
            if is_scheduled:
                return
            if body.startswith("+") and self.manager:
                await self.spawn_sibling_session(body[1:].strip())
                return
            self.send_reply("Still processing... (use +message to spawn sibling session)")
            return

        await self.process_message(body)

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

            display = output[:4000] + "\n... (truncated)" if len(output) > 4000 else output
            self.send_reply(f"$ {cmd}\n{display}")

            context_msg = (
                f"[I ran a shell command: `{cmd}`]\n\nOutput:\n```\n{output[:8000]}\n```\n"
                "\n(Just acknowledge briefly - I may ask about this next.)"
            )
            await self.process_message(context_msg)

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
                output = output[:3500] + "\n... (truncated)"
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
            return [line.decode("utf-8", errors="replace") for line in lines[-num_lines:]]

    # -------------------------------------------------------------------------
    # Message processing
    # -------------------------------------------------------------------------

    async def process_message(self, body: str):
        """Send message to agent and relay response."""
        self.processing = True
        self.send_typing()

        try:
            session = self.sessions.get(self.session_name)
            if not session:
                self.send_reply("Session not found in database.")
                return

            self.sessions.update_last_active(self.session_name)
            append_to_history(body, self.working_dir, session.claude_session_id)
            log_activity(body, session=self.session_name, source="xmpp")
            self.messages.add(self.session_name, "user", body, session.active_engine)

            if session.active_engine == "claude":
                await self._run_claude(body, session)
            else:
                await self._run_opencode(body, session)

        except Exception as e:
            self.log.exception("Error")
            self.send_reply(f"Error: {e}")
        finally:
            self.processing = False

    async def _run_claude(self, body: str, session):
        """Run Claude and handle events."""
        self.runner = ClaudeRunner(self.working_dir, self.output_dir, self.session_name)
        response_parts: list[str] = []
        tool_summaries: list[str] = []
        last_progress_at = 0

        async for event_type, content in self.runner.run(body, session.claude_session_id):
            if event_type == "session_id":
                self.sessions.update_claude_session_id(self.session_name, content)
            elif event_type == "text":
                response_parts = [content]
            elif event_type == "tool":
                tool_summaries.append(content)
                if len(tool_summaries) - last_progress_at >= 8:
                    last_progress_at = len(tool_summaries)
                    self.send_reply(f"... {' '.join(tool_summaries[-3:])}")
            elif event_type == "result":
                self._send_result(tool_summaries, response_parts, content, session.active_engine)
            elif event_type == "error":
                self.send_reply(f"Error: {content}")
            elif event_type == "cancelled":
                self.send_reply("Cancelled.")

    async def _run_opencode(self, body: str, session):
        """Run OpenCode and handle events."""
        self.runner = OpenCodeRunner(
            self.working_dir, self.output_dir, self.session_name,
            model=session.model_id,
            reasoning_mode=session.reasoning_mode,
            agent=session.opencode_agent,
        )
        response_parts: list[str] = []
        tool_summaries: list[str] = []
        accumulated = ""
        last_progress_at = 0

        async for event_type, content in self.runner.run(body, session.opencode_session_id):
            if event_type == "session_id":
                self.sessions.update_opencode_session_id(self.session_name, content)
            elif event_type == "text" and isinstance(content, str):
                accumulated += content
                response_parts = [accumulated]
            elif event_type == "tool" and isinstance(content, str):
                tool_summaries.append(content)
                if len(tool_summaries) - last_progress_at >= 8:
                    last_progress_at = len(tool_summaries)
                    self.send_reply(f"... {' '.join(tool_summaries[-3:])}")
            elif event_type == "result" and isinstance(content, OpenCodeResult):
                stats = (
                    f"[{content.tokens_in}/{content.tokens_out} tok"
                    f" r{content.tokens_reasoning} c{content.tokens_cache_read}/{content.tokens_cache_write}"
                    f" ${content.cost:.3f} {content.duration_s:.1f}s]"
                )
                self._send_result(tool_summaries, response_parts, stats, session.active_engine)
            elif event_type == "error":
                self.send_reply(f"Error: {content}")
            elif event_type == "cancelled":
                self.send_reply("Cancelled.")

    def _send_result(
        self,
        tool_summaries: list[str],
        response_parts: list[str],
        stats: str,
        engine: str,
    ):
        """Format and send final result."""
        parts = []
        if tool_summaries:
            tools = " ".join(tool_summaries[:5])
            if len(tool_summaries) > 5:
                tools += f" +{len(tool_summaries) - 5}"
            parts.append(tools)
        if response_parts:
            parts.append(response_parts[-1])
        parts.append(stats)

        self.send_reply("\n\n".join(parts))
        self.messages.add(
            self.session_name, "assistant",
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

        account = register_unique_account(
            f"{self.session_name}-sib",
            self.db, self.ejabberd_ctl, self.xmpp_domain, self.log,
        )
        if not account:
            self.send_reply("Failed to create sibling session")
            return

        name, password, jid = account
        recipient_user = self.xmpp_recipient.split("@")[0]
        add_roster_subscription(name, self.xmpp_recipient, "Clients", self.ejabberd_ctl, self.xmpp_domain)
        add_roster_subscription(recipient_user, jid, "Sessions", self.ejabberd_ctl, self.xmpp_domain)
        create_tmux_session(name, self.working_dir)

        self.sessions.create(name=name, xmpp_jid=jid, xmpp_password=password, tmux_name=name)

        bot = await self.manager.start_session_bot(name, jid, password)
        if bot:
            for _ in range(50):
                if bot.is_connected():
                    break
                await asyncio.sleep(0.1)
            bot.send_reply(f"Sibling session '{name}' (spawned from {self.session_name})")
            await bot.process_message(first_message)
        else:
            self.send_reply("Failed to start sibling session")
