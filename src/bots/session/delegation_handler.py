"""Conversational delegation: parse user intent and delegate to dispatchers."""

from __future__ import annotations

import logging
import os
import re
import secrets
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from src.delegation import delegate_once, parse_intent, resolve_dispatcher_name

if TYPE_CHECKING:
    import sqlite3

    from src.manager import SessionManager


class DelegationHandlerMixin:
    """Mixin for SessionBot: dispatcher list, startup context, conversational delegate."""

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
            "When the user asks to ask/delegate to another model, use one of these names. "
            "Use /dispatchers to refresh this list if needed. "
            "If the user mentions unfamiliar terms, check Switch session history for relevant context before responding."
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

    async def _maybe_handle_conversational_delegation(self, body: str) -> bool:
        dispatchers = self._available_delegate_dispatchers()
        unknown_target = self._extract_unknown_delegation_target(body, dispatchers)
        if unknown_target:
            known = ", ".join(sorted(dispatchers.keys())) or "none"
            self.send_reply(
                f"I couldn't find dispatcher '{unknown_target}'. Available: {known}. "
                "Try /dispatchers for the full list."
            )
            return True

        intent = parse_intent(body, dispatchers=dispatchers)
        if not intent:
            return False

        cfg = dispatchers.get(intent.dispatcher_name)
        if not cfg:
            self.send_reply(
                f"Delegation failed: unknown dispatcher '{intent.dispatcher_name}'."
            )
            return True

        dispatcher_jid = str(cfg.get("jid") or "").strip()
        dispatcher_password = str(cfg.get("password") or "").strip()
        if not dispatcher_jid or not dispatcher_password:
            self.send_reply(
                f"Delegation failed: dispatcher '{intent.dispatcher_name}' is not fully configured."
            )
            return True

        token = f"switch-delegate-{secrets.token_hex(6)}"
        try:
            self.delegations.create(
                token=token,
                parent_session=self.session_name,
                dispatcher_name=intent.dispatcher_name,
                dispatcher_jid=dispatcher_jid,
                prompt=intent.prompt,
            )
            self.delegations.mark_running(token)
        except Exception:
            self.log.exception("Failed to persist delegation task")

        self.send_reply(
            f"Delegating to {intent.dispatcher_name}...",
            meta_type="delegation",
            meta_tool="delegate",
            meta_attrs={
                "version": "1",
                "state": "running",
                "dispatcher": intent.dispatcher_name,
                "token": token,
            },
        )

        timeout_s = float(os.getenv("SWITCH_DELEGATE_TIMEOUT_S", "180") or "180")
        poll_s = float(os.getenv("SWITCH_DELEGATE_POLL_INTERVAL_S", "1.0") or "1.0")

        async def _send_via_current_session(envelope: str) -> None:
            self.send_message(
                mto=cast(Any, dispatcher_jid), mbody=envelope, mtype="chat"
            )

        try:
            result = await delegate_once(
                self.db,
                server=self.xmpp_server,
                dispatcher_jid=dispatcher_jid,
                dispatcher_password=dispatcher_password,
                prompt=intent.prompt,
                parent_session=self.session_name,
                token=token,
                timeout_s=timeout_s,
                poll_interval_s=poll_s,
                send_func=_send_via_current_session,
                on_spawned=lambda s, m: self.delegations.mark_spawned(
                    token,
                    delegated_session=s,
                    delegated_user_message_id=m,
                ),
            )
            with suppress(Exception):
                self.delegations.mark_completed(
                    token,
                    delegated_reply_message_id=result.assistant_message_id,
                )
            self.send_reply(
                f"[Delegated via {intent.dispatcher_name} ({result.session_name})]\n\n{result.content}",
                meta_type="delegation",
                meta_tool="delegate",
                meta_attrs={
                    "version": "1",
                    "state": "completed",
                    "dispatcher": intent.dispatcher_name,
                    "token": token,
                    "delegated_session": result.session_name,
                },
            )
        except TimeoutError as e:
            with suppress(Exception):
                self.delegations.mark_failed(token, error=str(e), status="timed_out")
            self.send_reply(f"Delegation timed out: {e}")
        except Exception as e:
            with suppress(Exception):
                self.delegations.mark_failed(token, error=str(e), status="failed")
            self.send_reply(f"Delegation failed: {type(e).__name__}: {e}")

        return True

    def _extract_unknown_delegation_target(
        self, body: str, dispatchers: dict[str, dict]
    ) -> str | None:
        text = (body or "").strip()
        if not text:
            return None

        known = set(dispatchers.keys())
        if not known:
            return None

        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = re.sub(
            r"^(?:(?:ok(?:ay)?|alright|all\s+right|hey|yo|well|so|right|hmm|um|uh)[,\s]+)+",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()

        patterns = [
            r"^(?:please\s+)?(?:can\s+you\s+)?(?:ask|query|consult)\s+(?P<target>[a-z0-9_-]+)\s+",
            r"^(?:please\s+)?(?:can\s+you\s+)?delegate(?:\s+(?:this|that|it))?(?:\s+to)?\s+(?P<target>[a-z0-9_-]+)\b",
            r"^(?:please\s+)?(?:can\s+you\s+)?get\s+(?:a\s+)?second\s+opinion\s+from\s+(?P<target>[a-z0-9_-]+)\b",
        ]
        for pat in patterns:
            m = re.match(pat, normalized, flags=re.IGNORECASE)
            if not m:
                continue
            target = (m.groupdict().get("target") or "").strip()
            if not target:
                continue
            if resolve_dispatcher_name(target, known) is None:
                return target
        return None
