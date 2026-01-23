"""Dispatcher bot - creates new session bots on demand."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Coroutine

from src.db import SessionRepository
from src.engines import opencode_model_for_agent
from src.helpers import (
    add_roster_subscription,
    create_tmux_session,
    register_unique_account,
    slugify,
)
from src.utils import BaseXMPPBot

if TYPE_CHECKING:
    import sqlite3

    from src.manager import SessionManager


class DispatcherBot(BaseXMPPBot):
    """Dispatcher bot that creates new session bots.

    Each dispatcher is tied to a specific engine/agent:
    - cc: Claude Code
    - oc: OpenCode with GLM 4.7 (bridge agent)
    - oc-gpt: OpenCode with GPT 5.2 (bridge-gpt agent)
    """

    def __init__(
        self,
        jid: str,
        password: str,
        db: "sqlite3.Connection",
        working_dir: str,
        xmpp_recipient: str,
        xmpp_domain: str,
        ejabberd_ctl: str,
        manager: "SessionManager | None" = None,
        *,
        engine: str = "opencode",
        opencode_agent: str | None = "bridge",
        label: str = "GLM 4.7",
    ):
        super().__init__(jid, password)
        self.db = db
        self.sessions = SessionRepository(db)
        self.working_dir = working_dir
        self.xmpp_recipient = xmpp_recipient
        self.xmpp_domain = xmpp_domain
        self.ejabberd_ctl = ejabberd_ctl
        self.manager: SessionManager | None = manager
        self.engine = engine
        self.opencode_agent = opencode_agent
        self.label = label

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

        self._commands: dict[str, Callable[[str], Coroutine]] = {
            "/list": self._cmd_list,
            "/kill": self._cmd_kill,
            "/recent": self._cmd_recent,
            "/help": self._cmd_help,
        }

    async def on_start(self, event):
        self.send_presence()
        await self.get_roster()
        self.log = logging.getLogger("dispatcher")
        self.log.info("Dispatcher connected")

    def on_disconnected(self, event):
        self.log.warning("Dispatcher disconnected, reconnecting...")
        asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        await asyncio.sleep(5)
        self.connect()

    async def on_message(self, msg):
        try:
            if msg["type"] not in ("chat", "normal") or not msg["body"]:
                return

            sender = str(msg["from"].bare)
            dispatcher_bare = str(self.boundjid.bare)
            sender_user = sender.split("@")[0]

            if not (
                sender == self.xmpp_recipient
                or sender == dispatcher_bare
                or sender_user.startswith("switch-loopback-")
            ):
                return

            is_loopback = sender_user.startswith("switch-loopback-")
            reply_to = sender if is_loopback else self.xmpp_recipient

            body = msg["body"].strip()
            if body.startswith("@"):
                body = "/" + body[1:]

            self.log.info(f"Dispatcher received: {body[:50]}...")

            if body.startswith("/"):
                if is_loopback:
                    self.send_reply(
                        "Loopback only supports session creation.", recipient=reply_to
                    )
                    return
                await self._dispatch_command(body)
                return

            await self.create_session(body)
            if is_loopback:
                self.send_reply(f"Dispatcher received: {body}", recipient=reply_to)

        except Exception as exc:
            self.log.exception("Dispatcher error")
            self.send_reply(f"Error: {exc}", recipient=self.xmpp_recipient)

    async def _dispatch_command(self, body: str) -> None:
        """Dispatch command to handler."""
        parts = body.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handler = self._commands.get(cmd)
        if handler:
            await handler(arg)
        else:
            self.send_reply(f"Unknown: {cmd}. Try /help", recipient=self.xmpp_recipient)

    async def _cmd_list(self, _arg: str) -> None:
        sessions = self.sessions.list_recent(15)
        if sessions:
            lines = ["Sessions:"] + [f"  {s.name}@{self.xmpp_domain}" for s in sessions]
            self.send_reply("\n".join(lines), recipient=self.xmpp_recipient)
        else:
            self.send_reply("No sessions yet.", recipient=self.xmpp_recipient)

    async def _cmd_kill(self, arg: str) -> None:
        if not arg:
            self.send_reply(
                "Usage: /kill <session-name>", recipient=self.xmpp_recipient
            )
            return
        if not self.manager:
            self.send_reply(
                "Session manager unavailable.", recipient=self.xmpp_recipient
            )
            return
        await self.manager.kill_session(arg)
        self.send_reply(f"Killed: {arg}", recipient=self.xmpp_recipient)

    async def _cmd_recent(self, _arg: str) -> None:
        sessions = self.sessions.list_recent(10)
        if sessions:
            lines = ["Recent:"]
            for s in sessions:
                last = s.last_active[5:16] if s.last_active else "?"
                lines.append(f"  {s.name} [{s.status}] {last}")
            self.send_reply("\n".join(lines), recipient=self.xmpp_recipient)
        else:
            self.send_reply("No sessions yet.", recipient=self.xmpp_recipient)

    async def _cmd_help(self, _arg: str) -> None:
        self.send_reply(
            f"Send any message to start a new {self.label} session.\n\n"
            "Orchestrators:\n"
            "  cc@ - Claude Code\n"
            "  oc@ - OpenCode (GLM 4.7)\n"
            "  oc-gpt@ - OpenCode (GPT 5.2)\n\n"
            "Commands:\n"
            "  /list - show sessions\n"
            "  /recent - recent with status\n"
            "  /kill <name> - end session\n"
            "  /help - this message",
            recipient=self.xmpp_recipient,
        )

    async def create_session(self, first_message: str):
        """Create a new session."""
        self.send_typing()
        message = first_message.strip()

        account = register_unique_account(
            slugify(message or first_message),
            self.db,
            self.ejabberd_ctl,
            self.xmpp_domain,
            self.log,
        )
        if not account:
            self.send_reply(f"Failed to create session", recipient=self.xmpp_recipient)
            return

        name, password, jid = account
        self.send_reply(
            f"Creating: {name} ({self.label})...", recipient=self.xmpp_recipient
        )

        recipient_user = self.xmpp_recipient.split("@")[0]
        add_roster_subscription(
            name, self.xmpp_recipient, "Clients", self.ejabberd_ctl, self.xmpp_domain
        )
        add_roster_subscription(
            recipient_user, jid, "Sessions", self.ejabberd_ctl, self.xmpp_domain
        )
        create_tmux_session(name, self.working_dir)

        model_id = opencode_model_for_agent(self.opencode_agent)
        self.sessions.create(
            name=name,
            xmpp_jid=jid,
            xmpp_password=password,
            tmux_name=name,
            model_id=model_id,
            opencode_agent=self.opencode_agent or "bridge",
            active_engine=self.engine,
        )

        if not self.manager:
            self.send_reply(
                "Session manager unavailable.", recipient=self.xmpp_recipient
            )
            return

        bot = await self.manager.start_session_bot(name, jid, password)
        if bot:
            for _ in range(50):
                if bot.is_connected():
                    break
                await asyncio.sleep(0.1)
            bot.send_reply(
                f"Session '{name}' ({self.label}). Processing: {message[:50]}..."
            )
            await bot.process_message(message or first_message)
        else:
            self.send_reply(
                f"Failed to start bot for {name}", recipient=self.xmpp_recipient
            )
