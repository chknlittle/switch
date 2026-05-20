"""MUC collaboration room settings and join."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.db import SessionRepository


class RoomMixin:
    """Mixin for SessionBot: load room_jid and join collaboration MUC."""

    log: logging.Logger
    session_name: str
    sessions: SessionRepository
    room_jid: str | None
    room_nick: str
    startup_error: str | None

    def _load_room_settings(self) -> None:
        session = self.sessions.get(self.session_name)
        self.room_jid = (session.room_jid or "").split("/", 1)[0] if session else None
        self.room_nick = self.session_name

    async def _join_collaboration_room(self) -> bool:
        room = (self.room_jid or "").strip()
        if not room:
            return True
        try:
            muc = cast(Any, self["xep_0045"])
            await muc.join_muc(room, self.room_nick)  # type: ignore[attr-defined]
            participants = self.sessions.list_collaborators(self.session_name)
            for participant in participants:
                if participant == str(self.boundjid.bare):
                    continue
                try:
                    muc.invite(room, participant)  # type: ignore[attr-defined]
                except Exception:
                    continue
            return True
        except Exception:
            self.startup_error = f"failed to join collaboration room {room}"
            self.log.exception("Failed to join collaboration room: %s", room)
            return False
