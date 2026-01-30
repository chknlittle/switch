"""Session manager - manages all session bots and dispatchers."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from pathlib import Path

from src.bots import DirectoryBot, DispatcherBot, SessionBot
from src.db import SessionRepository
from src.helpers import create_tmux_session
from src.lifecycle.sessions import kill_session as lifecycle_kill_session
from src.lifecycle.sessions import create_session as lifecycle_create_session

log = logging.getLogger("manager")


class SessionManager:
    """Manages all session bots and dispatchers."""

    def __init__(
        self,
        db: sqlite3.Connection,
        working_dir: str,
        output_dir: Path,
        xmpp_server: str,
        xmpp_domain: str,
        xmpp_recipient: str,
        ejabberd_ctl: str,
        dispatchers_config: dict,
    ):
        self.db = db
        self.sessions = SessionRepository(db)
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.xmpp_server = xmpp_server
        self.xmpp_domain = xmpp_domain
        self.xmpp_recipient = xmpp_recipient
        self.ejabberd_ctl = ejabberd_ctl
        self.dispatchers_config = dispatchers_config
        self.session_bots: dict[str, SessionBot] = {}
        self.dispatchers: dict[str, DispatcherBot] = {}
        self.directory: DirectoryBot | None = None

    async def start_directory_service(
        self, *, jid: str, password: str, pubsub_service_jid: str
    ):
        """Start the Switch directory service bot."""
        directory = DirectoryBot(
            jid,
            password,
            db=self.db,
            xmpp_domain=self.xmpp_domain,
            dispatchers_config=self.dispatchers_config,
            pubsub_service_jid=pubsub_service_jid,
        )
        directory.connect_to_server(self.xmpp_server)
        self.directory = directory
        log.info(f"Started directory service: {jid} (pubsub={pubsub_service_jid})")

    def notify_directory_sessions_changed(self, dispatcher_jid: str | None = None) -> None:
        """Best-effort pubsub ping so clients refresh hierarchy."""
        if not self.directory:
            return
        try:
            self.directory.notify_sessions_changed(dispatcher_jid=dispatcher_jid)
        except Exception:
            pass

    async def start_session_bot(self, name: str, jid: str, password: str) -> SessionBot:
        """Start a session bot."""
        bot = SessionBot(
            name,
            jid,
            password,
            self.db,
            self.working_dir,
            self.output_dir,
            self.xmpp_recipient,
            self.xmpp_domain,
            self.xmpp_server,
            self.ejabberd_ctl,
            manager=self,
        )
        bot.connect_to_server(self.xmpp_server)
        self.session_bots[name] = bot
        return bot

    async def create_session(self, message: str):
        """Create a new session from dispatcher message."""
        created = await lifecycle_create_session(
            self,
            message,
            engine="opencode",
            opencode_agent="bridge",
            label="OpenCode",
            dispatcher_jid=None,
        )
        if not created:
            log.error("Failed to create session")
            return

    async def kill_session(self, name: str) -> bool:
        """Kill a session and cleanup."""
        return await lifecycle_kill_session(self, name)

    async def start_dispatchers(self):
        """Start all dispatcher bots."""
        for name, cfg in self.dispatchers_config.items():
            dispatcher = DispatcherBot(
                cfg["jid"],
                cfg["password"],
                self.db,
                self.working_dir,
                self.xmpp_recipient,
                self.xmpp_domain,
                self.ejabberd_ctl,
                manager=self,
                engine=cfg["engine"],
                opencode_agent=cfg["agent"],
                label=cfg["label"],
            )
            dispatcher.connect_to_server(self.xmpp_server)
            self.dispatchers[name] = dispatcher
            log.info(f"Started dispatcher: {name} ({cfg['jid']})")

    async def restore_sessions(self):
        """Restore existing sessions from DB."""
        # Restoring every historical "active" session can create hundreds of bots,
        # which is slow and can destabilize the process. Prefer restoring only the
        # most recently used sessions; older sessions can be re-opened on-demand.
        limit = int(os.getenv("SWITCH_RESTORE_ACTIVE_LIMIT", "50"))
        try:
            active = self.sessions.list_active_recent(limit=limit)
        except Exception:
            active = self.sessions.list_active()
        for session in active:
            # Ensure the session has a tmux pane tailing its log.
            create_tmux_session(session.name, self.working_dir)
            await self.start_session_bot(
                session.name, session.xmpp_jid, session.xmpp_password
            )
        log.info(f"Started {len(active)} existing session(s)")
