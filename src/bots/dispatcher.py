"""Dispatcher bot - creates new session bots on demand."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Coroutine

from src.db import SessionRepository
from src.lifecycle.sessions import create_session as lifecycle_create_session
from src.ralph import parse_ralph_command
from src.runners import create_runner
from src.runners.opencode.config import OpenCodeConfig
from src.utils import BaseXMPPBot

if TYPE_CHECKING:
    import sqlite3

    from src.manager import SessionManager


class DispatcherBot(BaseXMPPBot):
    """Dispatcher bot that creates new session bots.

    Each dispatcher is tied to a specific engine/agent:
    - cc: Claude Code
    - oc: OpenCode with GLM 4.7 Heretic (bridge agent)
    - oc-gpt: OpenCode with GPT 5.2 (bridge-gpt agent)
    - oc-glm-zen: OpenCode with GLM 4.7 via Zen (bridge-zen agent)
    - oc-gpt-or: OpenCode with GPT 5.2 via OpenRouter (bridge-gpt-or agent)
    - oc-kimi-coding: OpenCode with Kimi K2.5 via Kimi for Coding (bridge-kimi-coding agent)
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
        label: str = "GLM 4.7 Heretic",
    ):
        super().__init__(jid, password)
        # Initialize logger early because Slixmpp can deliver stanzas before
        # session_start fires (race on connect), and we log inside on_message.
        self.log = logging.getLogger(f"dispatcher.{jid}")
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
            "/commit": self._cmd_commit,
            "/c": self._cmd_commit,
            "/ralph": self._cmd_ralph,
            "/help": self._cmd_help,
        }

    async def on_start(self, event):
        self.send_presence()
        await self.get_roster()
        self.log.info("Dispatcher connected")
        self.set_connected(True)

    def on_disconnected(self, event):
        self.log.warning("Dispatcher disconnected, reconnecting...")
        self.set_connected(False)
        asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        await asyncio.sleep(5)
        self.connect()

    async def on_message(self, msg):
        await self.guard(
            self._handle_dispatcher_message(msg),
            recipient=self.xmpp_recipient,
            context="dispatcher.on_message",
        )

    async def _handle_dispatcher_message(self, msg):
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

    async def _cmd_commit(self, arg: str) -> None:
        if not arg:
            self.send_reply(
                "Usage: /commit <repo> or /commit <host>:<repo>",
                recipient=self.xmpp_recipient,
            )
            return

        arg = arg.strip()

        # Check for host:path syntax (e.g., helga:moonshot-v2)
        if ":" in arg and not arg.startswith("/"):
            host, repo = arg.split(":", 1)
            repo_path = f"~/{repo}" if not repo.startswith("/") else repo

            self.send_reply(
                f"Committing {repo} on {host}...", recipient=self.xmpp_recipient
            )

            prompt = (
                f"This project is on remote host '{host}'. "
                f"Use `ssh {host} 'cd {repo_path} && <command>'` for all git operations. "
                f"Please check git status, commit any changes with a good message, and push."
            )
            working_dir = str(Path.home())  # run locally, SSH handles remote
        else:
            # Local repo
            repo_path = Path.home() / arg
            if not (repo_path / ".git").exists():
                self.send_reply(
                    f"Not a git repo: {repo_path}", recipient=self.xmpp_recipient
                )
                return

            self.send_reply(f"Committing {arg}...", recipient=self.xmpp_recipient)
            prompt = f"please commit and push the working changes in {repo_path}"
            working_dir = str(repo_path)

        runner = create_runner(
            "opencode",
            working_dir=working_dir,
            output_dir=Path(self.working_dir) / "output",
            opencode_config=OpenCodeConfig(
                model="glm_vllm/glm-4.7-flash-heretic.Q8_0.gguf",
                agent="bridge",
            ),
        )

        result_text = ""
        async for event_type, data in runner.run(prompt):
            if event_type == "result" and isinstance(data, dict):
                text = data.get("text")
                if isinstance(text, str):
                    result_text = text
            elif event_type == "error":
                self.send_reply(f"Error: {data}", recipient=self.xmpp_recipient)
                return

        # Strip echoed prompt from response if present
        if result_text.startswith(prompt):
            result_text = result_text[len(prompt) :].lstrip()

        if result_text:
            self.send_reply(result_text.strip(), recipient=self.xmpp_recipient)
        else:
            self.send_reply("Done (no output)", recipient=self.xmpp_recipient)

    async def _cmd_help(self, _arg: str) -> None:
        self.send_reply(
            f"Send any message to start a new {self.label} session.\n\n"
            "Orchestrators:\n"
            "  cc@ - Claude Code\n"
            "  oc@ - OpenCode (GLM 4.7 Heretic)\n"
            "  oc-gpt@ - OpenCode (GPT 5.2)\n"
            "  oc-glm-zen@ - OpenCode (GLM 4.7 Zen)\n"
            "  oc-gpt-or@ - OpenCode (GPT 5.2 OpenRouter)\n\n"
            "Commands:\n"
            "  /list - show sessions\n"
            "  /recent - recent with status\n"
            "  /kill <name> - end session\n"
            "  /commit [host:]<repo> - commit and push\n"
            "  /ralph <args> - create a session and start a Ralph loop\n"
            "  /help - this message",
            recipient=self.xmpp_recipient,
        )

    async def _cmd_ralph(self, arg: str) -> None:
        """Create a new session and run a /ralph loop inside it.

        We support running /ralph from the dispatcher because users often want to
        kick off long loops from their "home" contact, not an already-open session.
        """
        raw_arg = arg.strip()
        if not raw_arg:
            self.send_reply(
                "Usage: /ralph <prompt/args>\n"
                "Example: /ralph 10 Refactor auth --wait 5 --done 'All tests pass'\n"
                "Swarm:   /ralph Refactor auth --max 10 --swarm 5",
                recipient=self.xmpp_recipient,
            )
            return

        parsed = parse_ralph_command(f"/ralph {raw_arg}")
        if parsed is None:
            self.send_reply(
                "Usage: /ralph <prompt/args>\n"
                "Example: /ralph 10 Refactor auth --wait 5 --done 'All tests pass'\n"
                "Swarm:   /ralph Refactor auth --max 10 --swarm 5",
                recipient=self.xmpp_recipient,
            )
            return

        swarm = int(parsed.get("swarm") or 1)
        forward_args = (parsed.get("forward_args") or raw_arg).strip()

        MAX_SWARM = 50
        if swarm > MAX_SWARM:
            swarm = MAX_SWARM
            self.send_reply(
                f"Clamped --swarm to {MAX_SWARM} for safety.",
                recipient=self.xmpp_recipient,
            )

        if not self.manager:
            self.send_reply(
                "Session manager unavailable.", recipient=self.xmpp_recipient
            )
            return

        # Use a stable, short name hint so repeated loops become ralph, ralph-2, ...
        # Create the session first, then invoke the /ralph command via the session
        # command handler. (Directly enqueuing "/ralph ..." as a normal message
        # would send it to the model instead of starting the Ralph loop.)

        async def _start_one() -> str | None:
            created_name = await lifecycle_create_session(
                self.manager,
                "",
                engine=self.engine,
                opencode_agent=self.opencode_agent,
                label=self.label,
                name_hint="ralph",
                announce="Ralph session '{name}' ({label}). Starting loop...",
                dispatcher_jid=str(self.boundjid.bare),
            )
            if not created_name:
                return None
            bot = self.manager.session_bots.get(created_name)
            if not bot:
                return None
            await bot.commands.handle(f"/ralph {forward_args}")
            return created_name

        if swarm <= 1:
            created = await _start_one()
            if not created:
                self.send_reply(
                    "Failed to create Ralph session", recipient=self.xmpp_recipient
                )
                return
            self.send_reply(
                f"Started Ralph in {created}@{self.xmpp_domain}",
                recipient=self.xmpp_recipient,
            )
            return

        names: list[str] = []
        for _ in range(swarm):
            created = await _start_one()
            if created:
                names.append(created)

        if not names:
            self.send_reply(
                "Failed to create Ralph swarm sessions", recipient=self.xmpp_recipient
            )
            return

        lines = [
            f"Started Ralph swarm x{len(names)}:",
            *[f"  {n}@{self.xmpp_domain}" for n in names],
        ]
        self.send_reply("\n".join(lines), recipient=self.xmpp_recipient)

    async def create_session(self, first_message: str):
        """Create a new session."""
        self.send_typing()
        message = first_message.strip()

        if not self.manager:
            self.send_reply(
                "Session manager unavailable.", recipient=self.xmpp_recipient
            )
            return

        created_name = await lifecycle_create_session(
            self.manager,
            message or first_message,
            engine=self.engine,
            opencode_agent=self.opencode_agent,
            label=self.label,
            on_reserved=lambda n: self.send_reply(
                f"Creating: {n} ({self.label})...", recipient=self.xmpp_recipient
            ),
            dispatcher_jid=str(self.boundjid.bare),
        )
        if not created_name:
            self.send_reply("Failed to create session", recipient=self.xmpp_recipient)
            return
