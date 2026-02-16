"""Session lifecycle operations.

Goal: keep session create/kill/close semantics in one place so the dispatcher,
session bot commands, and scripts don't drift.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Callable

from src.engines import opencode_model_for_agent
from src.helpers import (
    add_roster_subscription,
    create_tmux_session,
    delete_xmpp_account,
    kill_tmux_session,
    register_unique_account,
    slugify,
)

if TYPE_CHECKING:
    import sqlite3

    from src.bots.session import SessionBot
    from src.db import SessionRepository

_log = logging.getLogger("lifecycle.sessions")


class _SessionKillManager(Protocol):
    sessions: "SessionRepository"
    session_bots: dict[str, "SessionBot"]
    xmpp_server: str
    xmpp_domain: str
    xmpp_recipient: str
    ejabberd_ctl: str

    def notify_directory_sessions_changed(
        self, dispatcher_jid: str | None = None
    ) -> None: ...


class _SessionCreateManager(Protocol):
    db: "sqlite3.Connection"
    sessions: "SessionRepository"
    working_dir: str
    output_dir: Path
    xmpp_server: str
    xmpp_domain: str
    xmpp_recipient: str
    ejabberd_ctl: str
    session_bots: dict[str, "SessionBot"]

    async def start_session_bot(
        self, name: str, jid: str, password: str, xmpp_recipient: str
    ) -> "SessionBot": ...

    def notify_directory_sessions_changed(
        self, dispatcher_jid: str | None = None
    ) -> None: ...


async def create_session(
    manager: _SessionCreateManager,
    first_message: str,
    *,
    engine: str = "opencode",
    opencode_agent: str | None = "bridge",
    model_id: str | None = None,
    label: str | None = None,
    name_hint: str | None = None,
    announce: str | None = None,
    announce_vars: dict[str, str] | None = None,
    on_reserved: Callable[[str], None] | None = None,
    dispatcher_jid: str | None = None,
    owner_jid: str | None = None,
    collaborators: list[str] | None = None,
) -> str | None:
    """Create a session and start its bot.

    This centralizes the "register + roster + tmux + DB row + start bot" flow.
    Returns the new session name on success.
    """

    message = (first_message or "").strip()
    if not message:
        message = first_message

    base_name = (name_hint or slugify(message)).strip()

    account = register_unique_account(
        base_name,
        manager.db,
        manager.ejabberd_ctl,
        manager.xmpp_domain,
        _log,
    )
    if not account:
        return None

    name, password, jid = account

    if on_reserved:
        try:
            on_reserved(name)
        except Exception:
            # Purely informational.
            pass

    recipient = (owner_jid or manager.xmpp_recipient).split("/", 1)[0]
    recipient_user = recipient.split("@")[0]

    collab_members: list[str] = []
    seen_members: set[str] = set()
    for candidate in [recipient, *(collaborators or [])]:
        bare = (candidate or "").split("/", 1)[0].strip()
        if not bare or bare in seen_members:
            continue
        seen_members.add(bare)
        collab_members.append(bare)

    room_jid: str | None = None
    collab_mode = len(collab_members) > 1
    if collab_mode:
        muc_service = os.getenv("SWITCH_MUC_SERVICE", f"conference.{manager.xmpp_domain}")
        room_jid = f"{name}@{muc_service}".split("/", 1)[0]

    # Room-based collaborative sessions should not appear as direct 1:1 contacts.
    # Keep roster subscriptions only for non-collab sessions.
    if not collab_mode:
        add_roster_subscription(
            name,
            recipient,
            "Clients",
            manager.ejabberd_ctl,
            manager.xmpp_domain,
        )
        add_roster_subscription(
            recipient_user, jid, "Sessions", manager.ejabberd_ctl, manager.xmpp_domain
        )

    create_tmux_session(name, manager.working_dir)

    effective_model = model_id or opencode_model_for_agent(opencode_agent)
    manager.sessions.create(
        name=name,
        xmpp_jid=jid,
        xmpp_password=password,
        tmux_name=name,
        model_id=effective_model,
        opencode_agent=opencode_agent or "bridge",
        active_engine=engine,
        dispatcher_jid=dispatcher_jid,
        owner_jid=recipient,
        room_jid=room_jid,
    )
    manager.sessions.set_collaborators(name, collab_members)

    bot = await manager.start_session_bot(name, jid, password, recipient)
    connected = await bot.wait_connected(timeout=8)
    if not connected:
        _log.error(
            "Session startup failed for %s (%s)",
            name,
            getattr(bot, "startup_error", "unknown error"),
        )
        _rollback_failed_create(manager, name, jid, dispatcher_jid=dispatcher_jid)
        return None

    preview = (message or "").strip()[:50]
    if announce is None:
        label_str = f" ({label})" if label else ""
        announce = (
            f"Session '{name}'{label_str}. Processing: {preview}..."
            if preview
            else f"Session '{name}'{label_str}."
        )
    else:
        fmt_vars: dict[str, str] = {
            "name": name,
            "label": label or "",
            "preview": preview,
        }
        if announce_vars:
            fmt_vars.update({str(k): str(v) for k, v in announce_vars.items() if k})
        try:
            announce = announce.format(**fmt_vars)
        except Exception:
            pass

    bot.send_reply(announce)

    if message:
        await bot.process_message(message)

    manager.notify_directory_sessions_changed(dispatcher_jid=dispatcher_jid)
    return name


def _rollback_failed_create(
    manager: _SessionCreateManager,
    name: str,
    jid: str,
    *,
    dispatcher_jid: str | None,
) -> None:
    bot = manager.session_bots.pop(name, None)
    if bot:
        try:
            bot.shutting_down = True
            bot.disconnect()
        except Exception:
            pass

    try:
        delete_xmpp_account(
            jid.split("@", 1)[0],
            manager.ejabberd_ctl,
            manager.xmpp_domain,
            _log,
        )
    except Exception:
        pass

    try:
        kill_tmux_session(name)
    except Exception:
        pass

    try:
        manager.sessions.delete(name)
    except Exception:
        pass

    try:
        manager.notify_directory_sessions_changed(dispatcher_jid=dispatcher_jid)
    except Exception:
        pass


async def kill_session(
    manager: _SessionKillManager,
    name: str,
    *,
    goodbye: str = "Session closed. Goodbye!",
    send_goodbye: bool = True,
) -> bool:
    """Archive-with-goodbye session kill.

    Semantics:
    - Send a final goodbye message (best-effort)
    - Stop the in-memory bot and prevent reconnect
    - Unregister the XMPP account
    - Kill tmux
    - Mark session closed in DB
    """

    session = manager.sessions.get(name)
    if not session or session.status == "closed":
        return session is not None

    if send_goodbye:
        bot = manager.session_bots.get(name)
        if bot and bot.is_connected() and not bot.shutting_down:
            try:
                bot.send_reply(goodbye)
                await asyncio.sleep(0.25)
            except Exception:
                pass

    # If the bot is running, cancel in-flight work and prevent reconnects before we delete the account.
    bot = manager.session_bots.get(name)
    if bot:
        try:
            bot.shutting_down = True
            bot.cancel_operations(notify=False)
            bot.disconnect()
        except Exception:
            pass

    username = session.xmpp_jid.split("@")[0]
    delete_xmpp_account(
        username,
        manager.ejabberd_ctl,
        manager.xmpp_domain,
        getattr(bot, "log", None) or _log,
    )
    kill_tmux_session(name)
    manager.sessions.close(name)
    manager.session_bots.pop(name, None)

    manager.notify_directory_sessions_changed()
    return True
