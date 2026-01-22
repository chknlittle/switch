#!/usr/bin/env python3
"""Session manager - manages all session bots and dispatchers."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

from src.db import SessionRepository
from src.dispatcher import DispatcherBot
from src.helpers import (
    add_roster_subscription,
    create_tmux_session,
    delete_xmpp_account,
    kill_tmux_session,
    register_unique_account,
    slugify,
)
from src.session_bot import SessionBot

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
        self.loop = asyncio.get_event_loop()

    async def start_session_bot(self, name: str, jid: str, password: str):
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
        base_name = slugify(message)
        account = register_unique_account(
            base_name, self.db, self.ejabberd_ctl, self.xmpp_domain, log
        )
        if not account:
            log.error(f"Failed to create session {base_name}")
            return

        name, password, jid = account

        recipient_user = self.xmpp_recipient.split("@")[0]
        add_roster_subscription(
            name, self.xmpp_recipient, "Clients", self.ejabberd_ctl, self.xmpp_domain
        )
        add_roster_subscription(
            recipient_user, jid, "Sessions", self.ejabberd_ctl, self.xmpp_domain
        )

        create_tmux_session(name, self.working_dir)

        self.sessions.create(
            name=name,
            xmpp_jid=jid,
            xmpp_password=password,
            tmux_name=name,
        )

        bot = await self.start_session_bot(name, jid, password)
        if bot:
            for _ in range(50):
                if bot.is_connected():
                    break
                await asyncio.sleep(0.1)

            bot.send_reply(f"Session '{name}' created. Message here to continue.")
            await bot.process_message(message)
        else:
            log.error(f"Failed to start bot for {name}")

    async def kill_session(self, name: str) -> bool:
        """Kill a session and cleanup."""
        session = self.sessions.get(name)
        if not session:
            return False

        if session.status == "closed":
            return True

        username = session.xmpp_jid.split("@")[0]
        delete_xmpp_account(username, self.ejabberd_ctl, self.xmpp_domain, log)
        kill_tmux_session(name)

        self.sessions.close(name)
        if name in self.session_bots:
            del self.session_bots[name]
        return True

    async def start_dispatchers(self):
        """Start all dispatcher bots (cc, oc, oc-gpt)."""
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
        """Restore existing sessions from DB on startup."""
        active_sessions = self.sessions.list_active()
        for session in active_sessions:
            await self.start_session_bot(
                session.name, session.xmpp_jid, session.xmpp_password
            )
        log.info(f"Started {len(active_sessions)} existing session(s)")
