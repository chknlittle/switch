"""Internal delegation helpers for Switch sessions and scripts.

DB-backed loopback waits used by ask-agent.py / spawn-session.py.
"""

from __future__ import annotations

import asyncio
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from src.utils import BaseXMPPBot, get_xmpp_config, run_ejabberdctl


@dataclass(frozen=True)
class DelegationResult:
    token: str
    session_name: str
    user_message_id: int
    assistant_message_id: int
    content: str


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


T = TypeVar("T")


async def _poll_until(
    *,
    deadline: float,
    poll_interval_s: float,
    step: Callable[[], T | None],
) -> T | None:
    poll_interval = max(0.1, float(poll_interval_s or 1.0))

    while time.monotonic() < deadline:
        result = step()
        if result is not None:
            return result
        await asyncio.sleep(poll_interval)

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
    deadline = time.monotonic() + max(5.0, float(timeout_s or 0.0))

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

    session_name: str | None = None
    user_message_id: int | None = None

    def _find_spawned_session() -> tuple[str, int] | None:
        conn.commit()
        return find_spawned_session_for_token(
            conn,
            dispatcher_jid=dispatcher_jid,
            token=token,
            min_message_id=min_message_id,
        )

    session_ref = await _poll_until(
        deadline=deadline,
        poll_interval_s=poll_interval_s,
        step=_find_spawned_session,
    )
    if session_ref:
        session_name, user_message_id = session_ref
        if on_spawned is not None:
            try:
                on_spawned(session_name, user_message_id)
            except Exception:
                pass

    if not session_name or user_message_id is None:
        raise TimeoutError("timed out waiting for delegated session creation")

    def _find_reply() -> tuple[str, int] | None:
        conn.commit()
        return find_assistant_reply(
            conn,
            session_name=session_name,
            after_id=user_message_id,
        )

    reply = await _poll_until(
        deadline=deadline,
        poll_interval_s=poll_interval_s,
        step=_find_reply,
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

    raise TimeoutError(f"timed out waiting for delegated answer from {session_name}")
