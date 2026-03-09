"""Internal delegation helpers for Switch sessions.

Supports conversational delegation requests and DB-backed loopback waits.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.utils import BaseXMPPBot, get_xmpp_config, run_ejabberdctl


@dataclass(frozen=True)
class DelegationIntent:
    dispatcher_name: str
    prompt: str
    trigger: str


@dataclass(frozen=True)
class DelegationResult:
    token: str
    session_name: str
    user_message_id: int
    assistant_message_id: int
    content: str


@dataclass(frozen=True)
class IntentRule:
    pattern: re.Pattern[str]
    trigger: str
    requires_target: bool = False


_ALIAS_TO_DISPATCHER: dict[str, str] = {
    "codex": "oc-codex",
    "gemini": "oc-gemini",
    "gpt": "oc-gpt",
    "chatgpt": "oc-gpt",
    "claude": "cc",
    "heretic": "qwen",
    "glm": "qwen",
    "kimi": "oc-kimi-coding",
    "zen": "oc-glm-zen",
}

_LEADING_FILLERS_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all\s+right|hey|yo|well|so|right|hmm|um|uh)[,\s]+)+",
    re.IGNORECASE,
)

_INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        pattern=re.compile(
            r"^(?:please\s+)?(?:can\s+you\s+)?(?:ask|query|consult)\s+(?P<target>[a-z0-9_-]+)\s+(?P<prompt>.+)$",
            re.IGNORECASE,
        ),
        trigger="ask-target",
        requires_target=True,
    ),
    IntentRule(
        pattern=re.compile(
            r"^(?:please\s+)?(?:can\s+you\s+)?delegate(?:\s+(?:this|that|it))?(?:\s+to)?\s+(?P<target>[a-z0-9_-]+)\s*[:,-]?\s*(?P<prompt>.+)$",
            re.IGNORECASE,
        ),
        trigger="delegate-target",
        requires_target=True,
    ),
    IntentRule(
        pattern=re.compile(
            r"^(?:please\s+)?(?:can\s+you\s+)?delegate(?:\s+(?:this|that|it))?\s*(?:on|about|for|:)?\s*(?P<prompt>.+)$",
            re.IGNORECASE,
        ),
        trigger="delegate-generic",
    ),
    IntentRule(
        pattern=re.compile(
            r"^(?:please\s+)?(?:can\s+you\s+)?get\s+(?:a\s+)?second\s+opinion(?:\s+from\s+(?P<target>[a-z0-9_-]+))?\s*(?:on|about|for|:)?\s*(?P<prompt>.+)$",
            re.IGNORECASE,
        ),
        trigger="second-opinion",
    ),
)


class _DispatchSendBot(BaseXMPPBot):
    def __init__(self, jid: str, password: str, target_jid: str, message: str):
        super().__init__(jid, password, recipient=target_jid)
        self.message = message
        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("failed_auth", self.on_failed_auth)

    def on_failed_auth(self, event):
        self.disconnect(wait=False)

    async def on_start(self, event):
        self.send_presence()
        await self.get_roster()
        self.send_reply(self.message)
        self.disconnect(wait=True)


def _default_dispatcher_name() -> str:
    return (os.getenv("SWITCH_DEFAULT_DISPATCHER") or "oc-gpt").strip() or "oc-gpt"


def _default_delegate_dispatcher() -> str:
    return (
        os.getenv("SWITCH_DELEGATE_DEFAULT_DISPATCHER")
        or os.getenv("SWITCH_DEFAULT_DISPATCHER")
        or "oc-codex"
    ).strip() or "oc-codex"


def resolve_dispatcher_name(raw: str | None, known: set[str]) -> str | None:
    token = (raw or "").strip().lower()
    if not token:
        return None

    if token in known:
        return token

    mapped = _ALIAS_TO_DISPATCHER.get(token)
    if mapped and mapped in known:
        return mapped

    return None


def _normalize_intent_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return _LEADING_FILLERS_RE.sub("", normalized).strip()


def _clean_prompt(raw_prompt: str) -> str:
    return re.sub(r"^to\s+", "", raw_prompt.strip(), flags=re.IGNORECASE).strip()


def _resolve_dispatcher_with_fallback(target_raw: str, known: set[str]) -> str | None:
    dispatcher_name = resolve_dispatcher_name(target_raw, known)
    if target_raw and not dispatcher_name:
        # The user asked for a specific target, but it's not recognized.
        return None
    if dispatcher_name:
        return dispatcher_name

    fallback = _default_delegate_dispatcher()
    dispatcher_name = resolve_dispatcher_name(fallback, known)
    if dispatcher_name:
        return dispatcher_name

    return resolve_dispatcher_name(_default_dispatcher_name(), known)


def parse_intent(body: str, *, dispatchers: dict[str, dict]) -> DelegationIntent | None:
    text = (body or "").strip()
    if not text:
        return None

    known = set(dispatchers.keys())
    if not known:
        return None

    normalized = _normalize_intent_text(text)

    for rule in _INTENT_RULES:
        m = rule.pattern.match(normalized)
        if not m:
            continue

        prompt = _clean_prompt(m.groupdict().get("prompt") or "")
        if not prompt:
            return None

        target_raw = (m.groupdict().get("target") or "").strip()
        if rule.requires_target and not target_raw:
            continue
        dispatcher_name = _resolve_dispatcher_with_fallback(target_raw, known)
        if target_raw and not dispatcher_name:
            continue
        if not dispatcher_name:
            return None

        return DelegationIntent(
            dispatcher_name=dispatcher_name,
            prompt=prompt,
            trigger=rule.trigger,
        )

    return None


def build_envelope(*, token: str, prompt: str, parent_session: str) -> str:
    return (
        f"[delegate_id:{token}]\n"
        "You are being consulted by another Switch session. "
        "Provide your answer directly and concisely.\n"
        f"Parent session: {parent_session}\n\n"
        f"{prompt}"
    )


def get_latest_message_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS max_id FROM session_messages"
    ).fetchone()
    if not row:
        return 0
    value = row["max_id"]
    return int(value) if isinstance(value, (int, float)) else 0


def find_spawned_session_for_token(
    conn: sqlite3.Connection,
    *,
    dispatcher_jid: str,
    token: str,
    min_message_id: int,
) -> tuple[str, int] | None:
    row = conn.execute(
        """
        SELECT m.session_name, m.id
        FROM session_messages AS m
        JOIN sessions AS s ON s.name = m.session_name
        WHERE m.role = 'user'
          AND m.id > ?
          AND instr(m.content, ?) > 0
          AND s.status = 'active'
          AND COALESCE(s.dispatcher_jid, '') = ?
        ORDER BY m.id DESC
        LIMIT 1
        """,
        (min_message_id, token, dispatcher_jid),
    ).fetchone()
    if not row:
        return None
    return str(row["session_name"]), int(row["id"])


def find_assistant_reply(
    conn: sqlite3.Connection, *, session_name: str, after_id: int
) -> tuple[str, int] | None:
    row = conn.execute(
        """
        SELECT content, id
        FROM session_messages
        WHERE session_name = ?
          AND role = 'assistant'
          AND id > ?
        ORDER BY id ASC
        LIMIT 1
        """,
        (session_name, after_id),
    ).fetchone()
    if not row:
        return None
    return str(row["content"] or ""), int(row["id"])


async def send_dispatcher_message(
    *,
    server: str,
    dispatcher_jid: str,
    dispatcher_password: str,
    body: str,
) -> None:
    del dispatcher_password  # Kept for API compatibility.

    cfg = get_xmpp_config()
    domain = cfg["domain"]
    ejabberd_ctl = cfg["ejabberd_ctl"]

    username = f"switch-loopback-{secrets.token_hex(3)}"
    password = secrets.token_urlsafe(12)
    ok, output = run_ejabberdctl(ejabberd_ctl, "register", username, domain, password)
    if not ok:
        raise RuntimeError(f"failed to register delegate sender: {output}")

    jid = f"{username}@{domain}"
    bot = _DispatchSendBot(jid, password, dispatcher_jid, body)
    try:
        bot.connect_to_server(server)
        await asyncio.wait_for(bot.disconnected, timeout=12)
    finally:
        run_ejabberdctl(ejabberd_ctl, "unregister", username, domain)


async def delegate_once(
    conn: sqlite3.Connection,
    *,
    server: str,
    dispatcher_jid: str,
    dispatcher_password: str,
    prompt: str,
    parent_session: str,
    token: str | None = None,
    timeout_s: float = 180.0,
    poll_interval_s: float = 1.0,
    on_spawned: Callable[[str, int], None] | None = None,
    send_func: Callable[[str], Awaitable[None]] | None = None,
) -> DelegationResult:
    token = (token or f"switch-delegate-{secrets.token_hex(6)}").strip()
    envelope = build_envelope(token=token, prompt=prompt, parent_session=parent_session)

    min_message_id = get_latest_message_id(conn)
    if send_func is not None:
        await send_func(envelope)
    else:
        await send_dispatcher_message(
            server=server,
            dispatcher_jid=dispatcher_jid,
            dispatcher_password=dispatcher_password,
            body=envelope,
        )

    deadline = time.monotonic() + max(5.0, float(timeout_s or 0.0))
    poll_interval = max(0.1, float(poll_interval_s or 1.0))
    session_name: str | None = None
    user_message_id: int | None = None

    while time.monotonic() < deadline:
        conn.commit()
        session_ref = find_spawned_session_for_token(
            conn,
            dispatcher_jid=dispatcher_jid,
            token=token,
            min_message_id=min_message_id,
        )
        if session_ref:
            session_name, user_message_id = session_ref
            if on_spawned is not None:
                try:
                    on_spawned(session_name, user_message_id)
                except Exception:
                    pass
            break
        await asyncio.sleep(poll_interval)

    if not session_name or user_message_id is None:
        raise TimeoutError("timed out waiting for delegated session creation")

    while time.monotonic() < deadline:
        conn.commit()
        reply = find_assistant_reply(
            conn, session_name=session_name, after_id=user_message_id
        )
        if reply:
            content, reply_id = reply
            return DelegationResult(
                token=token,
                session_name=session_name,
                user_message_id=user_message_id,
                assistant_message_id=reply_id,
                content=content.strip(),
            )
        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"timed out waiting for delegated answer from {session_name}")
