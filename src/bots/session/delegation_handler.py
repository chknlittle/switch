"""Delegation startup context for session bots (no auto-delegation from chat)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

    from src.manager import SessionManager


class DelegationHandlerMixin:
    """Mixin for SessionBot: dispatcher list and startup context for agents."""

    log: logging.Logger
    session_name: str
    db: sqlite3.Connection
    manager: SessionManager | None
    xmpp_server: str

    def _build_delegation_startup_context(self) -> str:
        session = self.sessions.get(self.session_name)
        if session and session.dispatcher_jid and self.manager:
            dispatcher_jid = session.dispatcher_jid.split("/", 1)[0]
            for cfg in (self.manager.dispatchers_config or {}).values():
                if not isinstance(cfg, dict):
                    continue
                jid = str(cfg.get("jid") or "").split("/", 1)[0]
                if jid != dispatcher_jid:
                    continue
                if cfg.get("delegation_context") is False:
                    return ""
                break

        dispatchers = self._available_delegate_dispatchers()
        if not dispatchers:
            return ""

        names = ", ".join(sorted(dispatchers.keys()))
        return (
            "[Switch delegation context]\n"
            f"Available dispatchers right now: {names}.\n"
            "When the user asks to delegate to another model, run Switch scripts yourself "
            "(scripts/ask-agent.py or scripts/spawn-session.py with --dispatcher <name>). "
            "Switch does not auto-delegate from chat messages. "
            "Use /dispatchers to refresh this list if needed."
        )

    def _available_delegate_dispatchers(self) -> dict[str, dict]:
        if not self.manager:
            return {}

        out: dict[str, dict] = {}
        for name, cfg in (self.manager.dispatchers_config or {}).items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("disabled") is True:
                continue
            jid = str(cfg.get("jid") or "").strip()
            password = str(cfg.get("password") or "").strip()
            if not jid or not password:
                continue
            out[str(name)] = cfg
        return out
