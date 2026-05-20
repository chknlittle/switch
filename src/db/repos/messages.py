"""Session message repository and model."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.db.locks import shared_write_lock
from src.db.signals import notify_message_written


@dataclass
class SessionMessage:
    """Session message record."""

    id: int
    session_name: str
    role: str
    content: str
    engine: str
    created_at: str


class MessageRepository:
    """Repository for session_messages table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._write_lock = shared_write_lock(conn)

    def _row_to_message(self, row: sqlite3.Row) -> SessionMessage:
        return SessionMessage(
            id=row["id"],
            session_name=row["session_name"],
            role=row["role"],
            content=row["content"],
            engine=row["engine"],
            created_at=row["created_at"],
        )

    # Keep at most this many messages per session to prevent unbounded growth.
    MAX_MESSAGES_PER_SESSION = 500

    async def add(
        self,
        session_name: str,
        role: str,
        content: str,
        engine: str,
    ) -> None:
        async with self._write_lock:
            self.conn.execute(
                """INSERT INTO session_messages
                   (session_name, role, content, engine, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_name, role, content, engine, datetime.now().isoformat()),
            )
            # Trim old messages beyond the retention limit.
            self.conn.execute(
                """DELETE FROM session_messages
                   WHERE session_name = ? AND id NOT IN (
                       SELECT id FROM session_messages
                       WHERE session_name = ?
                       ORDER BY id DESC
                       LIMIT ?
                   )""",
                (session_name, session_name, self.MAX_MESSAGES_PER_SESSION),
            )
            self.conn.commit()
        await notify_message_written(self.conn)

    def list_recent(self, session_name: str, limit: int = 40) -> list[SessionMessage]:
        rows = self.conn.execute(
            """SELECT * FROM session_messages
               WHERE session_name = ?
               ORDER BY id DESC
               LIMIT ?""",
            (session_name, limit),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]
