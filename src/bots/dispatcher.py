"""Dispatcher bot - creates new session bots on demand."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Coroutine, cast

from src.db import SessionRepository
from src.engines import PI_MODEL_DEFAULT
from src.lifecycle.sessions import create_session as lifecycle_create_session
from src.ralph import parse_ralph_command
from src.runners import create_runner
from src.runners.pi.config import PiConfig
from src.utils import BaseXMPPBot

if TYPE_CHECKING:
    import sqlite3

    from src.manager import SessionManager


class DispatcherBot(BaseXMPPBot):
    """Dispatcher bot that creates new session bots.

    Each dispatcher is tied to a specific engine:
    - cc/claude: Claude Code
    - pi: Pi (Qwen/GLM via local inference)
    - debate: Multi-model debate
    """

    _WITH_FLAG_RE = re.compile(
        r"(?:^|\s)(?:--|—|–|−)with(?:=|\s|$)|(?:^|\s)(?:-|—|–|−)w(?:\s|$)"
    )

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
        engine: str = "pi",
        model_id: str | None = None,
        label: str = "Pi",
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
        self.model_id = model_id
        self.label = label

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

        self._commands: dict[str, Callable[[str, str], Coroutine]] = {
            "/list": self._cmd_list,
            "/kill": self._cmd_kill,
            "/recent": self._cmd_recent,
            "/commit": self._cmd_commit,
            "/c": self._cmd_commit,
            "/ralph": self._cmd_ralph,
            "/new": self._cmd_new,
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
        recipient = str(msg["from"].bare) if msg["type"] in ("chat", "normal") else None
        await self.guard(
            self._handle_dispatcher_message(msg),
            recipient=recipient,
            context="dispatcher.on_message",
        )

    async def _handle_dispatcher_message(self, msg):
        if msg["type"] not in ("chat", "normal") or not msg["body"]:
            return

        sender = str(msg["from"].bare)
        dispatcher_bare = str(self.boundjid.bare)
        owner_bare = (self.xmpp_recipient or "").split("/", 1)[0]
        sender_user = sender.split("@")[0]

        if not sender:
            return

        is_loopback = sender_user.startswith("switch-loopback-")
        is_dispatcher_self = sender == dispatcher_bare
        is_owner_sender = sender == owner_bare

        if not (is_owner_sender or is_dispatcher_self or is_loopback):
            self.send_reply(
                "Dispatcher access is owner-only. Ask the owner to start a shared session with /new --with ...",
                recipient=sender,
            )
            return

        if is_dispatcher_self:
            # Backward compatibility for local helper scripts that authenticate
            # as the dispatcher account itself.
            owner_jid = owner_bare
            reply_to = owner_bare
        elif is_loopback:
            owner_jid = owner_bare
            reply_to = sender
        else:
            owner_jid = sender
            reply_to = sender

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
            await self._dispatch_command(body, reply_to)
            return

        if self._WITH_FLAG_RE.search(body):
            parsed = self._parse_new_args(body)
            if not parsed:
                self.send_reply(
                    "Usage: /new --with <jid[,jid]> <prompt>", recipient=reply_to
                )
                return
            participants, prompt = parsed
            participants = [p for p in participants if p != reply_to]
            if participants:
                await self.create_session(
                    prompt,
                    reply_to=reply_to,
                    owner_jid=owner_jid,
                    collaborators=participants,
                )
                return

        await self.create_session(body, reply_to=reply_to, owner_jid=owner_jid)
        if is_loopback:
            self.send_reply(f"Dispatcher received: {body}", recipient=reply_to)

    async def _dispatch_command(self, body: str, reply_to: str) -> None:
        """Dispatch command to handler."""
        parts = body.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handler = self._commands.get(cmd)
        if handler:
            await handler(arg, reply_to)
        else:
            self.send_reply(f"Unknown: {cmd}. Try /help", recipient=reply_to)

    async def _cmd_list(self, _arg: str, reply_to: str) -> None:
        sessions = self.sessions.list_recent_for_owner(reply_to, 15)
        if sessions:
            lines = ["Sessions:"] + [f"  {s.name}@{self.xmpp_domain}" for s in sessions]
            self.send_reply("\n".join(lines), recipient=reply_to)
        else:
            self.send_reply("No sessions yet.", recipient=reply_to)

    async def _cmd_kill(self, arg: str, reply_to: str) -> None:
        if not arg:
            self.send_reply("Usage: /kill <session-name>", recipient=reply_to)
            return
        if not self.manager:
            self.send_reply("Session manager unavailable.", recipient=reply_to)
            return
        session = self.sessions.get(arg)
        owner = (session.owner_jid or "").split("/", 1)[0] if session else ""
        if not session or owner != reply_to:
            self.send_reply(f"Session not found: {arg}", recipient=reply_to)
            return
        await self.manager.kill_session(arg)
        self.send_reply(f"Killed: {arg}", recipient=reply_to)

    async def _cmd_recent(self, _arg: str, reply_to: str) -> None:
        sessions = self.sessions.list_recent_for_owner(reply_to, 10)
        if sessions:
            lines = ["Recent:"]
            for s in sessions:
                last = s.last_active[5:16] if s.last_active else "?"
                lines.append(f"  {s.name} [{s.status}] {last}")
            self.send_reply("\n".join(lines), recipient=reply_to)
        else:
            self.send_reply("No sessions yet.", recipient=reply_to)

    async def _cmd_commit(self, arg: str, reply_to: str) -> None:
        if not arg:
            self.send_reply(
                "Usage: /commit <repo> or /commit <host>:<repo>",
                recipient=reply_to,
            )
            return

        arg = arg.strip()

        # Check for host:path syntax (e.g., helga:moonshot-v2)
        if ":" in arg and not arg.startswith("/"):
            host, repo = arg.split(":", 1)
            repo_path = f"~/{repo}" if not repo.startswith("/") else repo

            self.send_reply(
                f"Committing {repo} on {host}...", recipient=reply_to
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
                    f"Not a git repo: {repo_path}", recipient=reply_to
                )
                return

            self.send_reply(f"Committing {arg}...", recipient=reply_to)
            prompt = f"please commit and push the working changes in {repo_path}"
            working_dir = str(repo_path)

        runner = create_runner(
            "pi",
            working_dir=working_dir,
            output_dir=Path(self.working_dir) / "output",
            pi_config=PiConfig(
                model=PI_MODEL_DEFAULT or None,
            ),
        )

        result_text = ""
        async for event_type, data in runner.run(prompt):
            if event_type == "result" and isinstance(data, dict):
                text = data.get("text")
                if isinstance(text, str):
                    result_text = text
            elif event_type == "error":
                self.send_reply(f"Error: {data}", recipient=reply_to)
                return

        # Strip echoed prompt from response if present
        if result_text.startswith(prompt):
            result_text = result_text[len(prompt) :].lstrip()

        if result_text:
            self.send_reply(result_text.strip(), recipient=reply_to)
        else:
            self.send_reply("Done (no output)", recipient=reply_to)

    async def _cmd_help(self, _arg: str, reply_to: str) -> None:
        self.send_reply(
            f"Send any message to start a new {self.label} session.\n\n"
            "Dispatchers are configured by Switch and may vary per deployment.\n\n"
            "Commands:\n"
            "  /list - show sessions\n"
            "  /recent - recent with status\n"
            "  /kill <name> - end session\n"
            "  /new --with <jid[,jid]> <prompt> - shared room session\n"
            "  /commit [host:]<repo> - commit and push\n"
            "  /ralph <args> - create a session and start a Ralph loop\n"
            "  /help - this message",
            recipient=reply_to,
        )

    @staticmethod
    def _normalize_participant(value: str, default_domain: str) -> str | None:
        token = (value or "").strip().strip(",")
        if not token:
            return None
        bare = token.split("/", 1)[0]
        if "@" not in bare:
            bare = f"{bare}@{default_domain}"
        return bare.lower()

    def _parse_new_args(self, arg: str) -> tuple[list[str], str] | None:
        arg = (
            arg.replace("—", "-")
            .replace("–", "-")
            .replace("−", "-")
            .strip()
        )
        try:
            tokens = shlex.split(arg)
        except ValueError:
            return None

        participants: list[str] = []
        prompt_tokens: list[str] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token in {"--with", "-with", "-w"}:
                idx += 1
                if idx >= len(tokens):
                    return None
                raw_list = tokens[idx]
                for part in raw_list.split(","):
                    norm = self._normalize_participant(part, self.xmpp_domain)
                    if norm:
                        participants.append(norm)
                idx += 1
                continue

            if token.startswith("--with=") or token.startswith("-with="):
                raw_list = token.split("=", 1)[1]
                for part in raw_list.split(","):
                    norm = self._normalize_participant(part, self.xmpp_domain)
                    if norm:
                        participants.append(norm)
                idx += 1
                continue

            prompt_tokens.append(token)
            idx += 1

        prompt = " ".join(prompt_tokens).strip()
        if not prompt:
            return None

        deduped: list[str] = []
        seen: set[str] = set()
        for p in participants:
            if p in seen:
                continue
            seen.add(p)
            deduped.append(p)
        return deduped, prompt

    async def _cmd_new(self, arg: str, reply_to: str) -> None:
        parsed = self._parse_new_args(arg)
        if not parsed:
            self.send_reply(
                "Usage: /new --with <jid[,jid]> <prompt>", recipient=reply_to
            )
            return
        participants, prompt = parsed
        participants = [p for p in participants if p != reply_to]
        if not participants:
            self.send_reply(
                "Usage: /new --with <jid[,jid]> <prompt>", recipient=reply_to
            )
            return
        await self.create_session(
            prompt,
            reply_to=reply_to,
            owner_jid=reply_to,
            collaborators=participants,
        )

    async def _cmd_ralph(self, arg: str, reply_to: str) -> None:
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
                recipient=reply_to,
            )
            return

        parsed = parse_ralph_command(f"/ralph {raw_arg}")
        if parsed is None:
            self.send_reply(
                "Usage: /ralph <prompt/args>\n"
                "Example: /ralph 10 Refactor auth --wait 5 --done 'All tests pass'\n"
                "Swarm:   /ralph Refactor auth --max 10 --swarm 5",
                recipient=reply_to,
            )
            return

        swarm = int(parsed.get("swarm") or 1)
        forward_args = (parsed.get("forward_args") or raw_arg).strip()

        MAX_SWARM = 50
        if swarm > MAX_SWARM:
            swarm = MAX_SWARM
            self.send_reply(
                f"Clamped --swarm to {MAX_SWARM} for safety.",
                recipient=reply_to,
            )

        if not self.manager:
            self.send_reply(
                "Session manager unavailable.", recipient=reply_to
            )
            return
        manager = cast("SessionManager", self.manager)

        # Use a stable, short name hint so repeated loops become ralph, ralph-2, ...
        # Create the session first, then invoke the /ralph command via the session
        # command handler. (Directly enqueuing "/ralph ..." as a normal message
        # would send it to the model instead of starting the Ralph loop.)

        async def _start_one() -> str | None:
            created_name = await lifecycle_create_session(
                manager,
                "",
                engine=self.engine,
                model_id=self.model_id,
                label=self.label,
                name_hint="ralph",
                announce="Ralph session '{name}' ({label}). Starting loop...",
                dispatcher_jid=str(self.boundjid.bare),
                owner_jid=reply_to,
            )
            if not created_name:
                return None
            bot = manager.session_bots.get(created_name)
            if not bot:
                return None
            await bot.commands.handle(f"/ralph {forward_args}")
            return created_name

        if swarm <= 1:
            created = await _start_one()
            if not created:
                self.send_reply(
                    "Failed to create Ralph session", recipient=reply_to
                )
                return
            self.send_reply(
                f"Started Ralph in {created}@{self.xmpp_domain}",
                recipient=reply_to,
            )
            return

        names: list[str] = []
        for _ in range(swarm):
            created = await _start_one()
            if created:
                names.append(created)

        if not names:
            self.send_reply(
                "Failed to create Ralph swarm sessions", recipient=reply_to
            )
            return

        lines = [
            f"Started Ralph swarm x{len(names)}:",
            *[f"  {n}@{self.xmpp_domain}" for n in names],
        ]
        self.send_reply("\n".join(lines), recipient=reply_to)

    async def create_session(
        self,
        first_message: str,
        *,
        reply_to: str,
        owner_jid: str,
        collaborators: list[str] | None = None,
    ):
        """Create a new session."""
        self.send_typing()
        message = first_message.strip()

        if not self.manager:
            self.send_reply("Session manager unavailable.", recipient=reply_to)
            return

        created_name = await lifecycle_create_session(
            self.manager,
            message or first_message,
            engine=self.engine,
            model_id=self.model_id,
            label=self.label,
            on_reserved=lambda n: self.send_reply(
                f"Creating: {n} ({self.label})...", recipient=reply_to
            ),
            dispatcher_jid=str(self.boundjid.bare),
            owner_jid=owner_jid,
            collaborators=collaborators,
        )
        if not created_name:
            self.send_reply("Failed to create session", recipient=reply_to)
            return
        created = self.sessions.get(created_name)
        if created and created.room_jid:
            self.send_reply(
                f"Collab room ready: {created.room_jid}",
                recipient=reply_to,
            )
