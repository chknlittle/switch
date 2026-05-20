"""Session repository and model."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.db.locks import shared_write_lock
from src.engines import remote_session_attr


@dataclass
class Session:
    """Session record."""

    name: str
    xmpp_jid: str
    xmpp_password: str
    claude_session_id: str | None
    opencode_session_id: str | None
    pi_session_id: str | None
    cursor_session_id: str | None
    active_engine: str
    model_id: str | None
    vllm_base_url: str | None
    reasoning_mode: str
    opencode_agent: str | None
    dispatcher_jid: str | None
    owner_jid: str | None
    room_jid: str | None
    tmux_name: str | None
    created_at: str
    last_active: str
    status: str


class SessionRepository:
    """Repository for sessions table."""

    _UPDATABLE_COLUMNS = {
        "active_engine",
        "claude_session_id",
        "last_active",
        "model_id",
        "opencode_session_id",
        "pi_session_id",
        "cursor_session_id",
        "reasoning_mode",
        "status",
    }

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._write_lock = shared_write_lock(conn)

    async def _update_session_column(
        self, name: str, column: str, value: object | None
    ) -> None:
        if column not in self._UPDATABLE_COLUMNS:
            raise ValueError(f"Unsupported session column update: {column}")
        async with self._write_lock:
            self.conn.execute(
                f"UPDATE sessions SET {column} = ? WHERE name = ?",
                (value, name),
            )
            self.conn.commit()

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            name=row["name"],
            xmpp_jid=row["xmpp_jid"],
            xmpp_password=row["xmpp_password"],
            claude_session_id=row["claude_session_id"],
            opencode_session_id=row["opencode_session_id"]
            if "opencode_session_id" in row.keys()
            else None,
            pi_session_id=row["pi_session_id"]
            if "pi_session_id" in row.keys()
            else None,
            cursor_session_id=row["cursor_session_id"]
            if "cursor_session_id" in row.keys()
            else None,
            active_engine=row["active_engine"] or "pi",
            model_id=row["model_id"] or None,
            vllm_base_url=row["vllm_base_url"]
            if "vllm_base_url" in row.keys()
            else None,
            reasoning_mode=row["reasoning_mode"] or "normal",
            opencode_agent=row["opencode_agent"]
            if "opencode_agent" in row.keys()
            else None,
            dispatcher_jid=row["dispatcher_jid"]
            if "dispatcher_jid" in row.keys()
            else None,
            owner_jid=row["owner_jid"] if "owner_jid" in row.keys() else None,
            room_jid=row["room_jid"] if "room_jid" in row.keys() else None,
            tmux_name=row["tmux_name"],
            created_at=row["created_at"],
            last_active=row["last_active"],
            status=row["status"] or "active",
        )

    def get(self, name: str) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_session(row) if row else None

    def get_by_jid(self, jid: str) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE xmpp_jid = ?", (jid,)
        ).fetchone()
        return self._row_to_session(row) if row else None

    def exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sessions WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def list_recent(self, limit: int = 15) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY last_active DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def list_recent_for_owner(self, owner_jid: str, limit: int = 15) -> list[Session]:
        owner_bare = (owner_jid or "").split("/", 1)[0]
        rows = self.conn.execute(
            """SELECT * FROM sessions
               WHERE owner_jid = ?
                  OR EXISTS (
                      SELECT 1
                      FROM session_collaborators AS c
                      WHERE c.session_name = sessions.name
                        AND c.participant_jid = ?
                  )
               ORDER BY last_active DESC
               LIMIT ?""",
            (owner_bare, owner_bare, limit),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def list_active(self) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'active'"
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def list_active_recent(self, limit: int = 50) -> list[Session]:
        """List most recently active sessions that are still marked active."""
        rows = self.conn.execute(
            """SELECT * FROM sessions
               WHERE status = 'active'
               ORDER BY last_active DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def list_active_recent_for_owner(
        self, owner_jid: str, limit: int = 50
    ) -> list[Session]:
        owner_bare = (owner_jid or "").split("/", 1)[0]
        rows = self.conn.execute(
            """SELECT * FROM sessions
               WHERE status = 'active'
                 AND (
                     owner_jid = ?
                     OR EXISTS (
                         SELECT 1
                         FROM session_collaborators AS c
                         WHERE c.session_name = sessions.name
                           AND c.participant_jid = ?
                     )
                 )
               ORDER BY last_active DESC
               LIMIT ?""",
            (owner_bare, owner_bare, limit),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def list_recent_closed(self, limit: int = 10) -> list[Session]:
        """List most recently active closed sessions (for directory browsing)."""
        rows = self.conn.execute(
            """SELECT * FROM sessions
               WHERE status = 'closed'
               ORDER BY last_active DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def list_recent_closed_for_owner(
        self, owner_jid: str, limit: int = 10
    ) -> list[Session]:
        owner_bare = (owner_jid or "").split("/", 1)[0]
        rows = self.conn.execute(
            """SELECT * FROM sessions
               WHERE status = 'closed'
                 AND (
                     owner_jid = ?
                     OR EXISTS (
                         SELECT 1
                         FROM session_collaborators AS c
                         WHERE c.session_name = sessions.name
                           AND c.participant_jid = ?
                     )
                 )
               ORDER BY last_active DESC
               LIMIT ?""",
            (owner_bare, owner_bare, limit),
        ).fetchall()
        return [self._row_to_session(row) for row in rows]

    async def create(
        self,
        name: str,
        xmpp_jid: str,
        xmpp_password: str,
        tmux_name: str,
        model_id: str | None = None,
        vllm_base_url: str | None = None,
        active_engine: str = "pi",
        reasoning_mode: str = "normal",
        opencode_agent: str | None = None,
        dispatcher_jid: str | None = None,
        owner_jid: str | None = None,
        room_jid: str | None = None,
    ) -> Session:
        owner_bare = (owner_jid or "").split("/", 1)[0] or None
        room_bare = (room_jid or "").split("/", 1)[0] or None
        now = datetime.now().isoformat()
        async with self._write_lock:
            self.conn.execute(
                """INSERT INTO sessions
                   (name, xmpp_jid, xmpp_password, tmux_name, created_at, last_active,
                    model_id, vllm_base_url, active_engine, reasoning_mode, opencode_agent, dispatcher_jid, owner_jid, room_jid)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    xmpp_jid,
                    xmpp_password,
                    tmux_name,
                    now,
                    now,
                    model_id,
                    vllm_base_url,
                    active_engine,
                    reasoning_mode,
                    opencode_agent,
                    dispatcher_jid,
                    owner_bare,
                    room_bare,
                ),
            )
            self.conn.commit()
            created = self.get(name)
        if not created:
            raise RuntimeError(f"Failed to load newly created session: {name}")
        return created

    async def update_last_active(self, name: str) -> None:
        await self._update_session_column(
            name, "last_active", datetime.now().isoformat()
        )

    async def update_engine(self, name: str, engine: str) -> None:
        await self._update_session_column(name, "active_engine", engine)

    async def update_reasoning_mode(self, name: str, mode: str) -> None:
        await self._update_session_column(name, "reasoning_mode", mode)

    async def update_model(self, name: str, model_id: str) -> None:
        await self._update_session_column(name, "model_id", model_id)

    async def update_remote_session_id(
        self, name: str, engine: str, session_id: str
    ) -> None:
        column = remote_session_attr(engine)
        if not column:
            return
        await self._update_session_column(name, column, session_id)

    async def reset_remote_session(self, name: str, engine: str) -> None:
        column = remote_session_attr(engine)
        if not column:
            return
        await self._update_session_column(name, column, None)

    async def close(self, name: str) -> None:
        await self._update_session_column(name, "status", "closed")

    async def delete(self, name: str) -> None:
        async with self._write_lock:
            self.conn.execute(
                "DELETE FROM session_messages WHERE session_name = ?", (name,)
            )
            self.conn.execute("DELETE FROM ralph_loops WHERE session_name = ?", (name,))
            self.conn.execute("DELETE FROM sessions WHERE name = ?", (name,))
            self.conn.commit()

    async def set_collaborators(self, session_name: str, jids: list[str]) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        for jid in jids:
            bare = (jid or "").split("/", 1)[0].strip()
            if not bare or bare in seen:
                continue
            seen.add(bare)
            normalized.append(bare)

        async with self._write_lock:
            self.conn.execute(
                "DELETE FROM session_collaborators WHERE session_name = ?",
                (session_name,),
            )
            for bare in normalized:
                self.conn.execute(
                    """INSERT OR IGNORE INTO session_collaborators
                       (session_name, participant_jid)
                       VALUES (?, ?)""",
                    (session_name, bare),
                )
            self.conn.commit()

    def list_collaborators(self, session_name: str) -> list[str]:
        rows = self.conn.execute(
            """SELECT participant_jid
               FROM session_collaborators
               WHERE session_name = ?
               ORDER BY participant_jid ASC""",
            (session_name,),
        ).fetchall()
        return [str(row["participant_jid"]) for row in rows]
