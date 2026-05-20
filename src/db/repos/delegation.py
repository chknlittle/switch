"""Delegation task repository and model."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DelegationTask:
    """Delegation task record."""

    id: int
    token: str
    parent_session: str
    dispatcher_name: str
    dispatcher_jid: str
    prompt: str
    status: str
    delegated_session: str | None
    delegated_user_message_id: int | None
    delegated_reply_message_id: int | None
    error: str | None
    created_at: str
    updated_at: str


class DelegationTaskRepository:
    """Repository for delegation_tasks table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _row_to_task(self, row: sqlite3.Row) -> DelegationTask:
        return DelegationTask(
            id=row["id"],
            token=row["token"],
            parent_session=row["parent_session"],
            dispatcher_name=row["dispatcher_name"],
            dispatcher_jid=row["dispatcher_jid"],
            prompt=row["prompt"],
            status=row["status"],
            delegated_session=row["delegated_session"],
            delegated_user_message_id=row["delegated_user_message_id"],
            delegated_reply_message_id=row["delegated_reply_message_id"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(
        self,
        *,
        token: str,
        parent_session: str,
        dispatcher_name: str,
        dispatcher_jid: str,
        prompt: str,
    ) -> int:
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            """INSERT INTO delegation_tasks
               (token, parent_session, dispatcher_name, dispatcher_jid, prompt, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)""",
            (token, parent_session, dispatcher_name, dispatcher_jid, prompt, now, now),
        )
        self.conn.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("Failed to create delegation task")
        return int(cursor.lastrowid)

    def mark_running(self, token: str) -> None:
        self.conn.execute(
            """UPDATE delegation_tasks
               SET status = 'running', updated_at = ?
               WHERE token = ?""",
            (datetime.now().isoformat(), token),
        )
        self.conn.commit()

    def mark_spawned(
        self,
        token: str,
        *,
        delegated_session: str,
        delegated_user_message_id: int,
    ) -> None:
        self.conn.execute(
            """UPDATE delegation_tasks
               SET status = 'spawned',
                   delegated_session = ?,
                   delegated_user_message_id = ?,
                   updated_at = ?
               WHERE token = ?""",
            (
                delegated_session,
                delegated_user_message_id,
                datetime.now().isoformat(),
                token,
            ),
        )
        self.conn.commit()

    def mark_completed(self, token: str, *, delegated_reply_message_id: int) -> None:
        self.conn.execute(
            """UPDATE delegation_tasks
               SET status = 'completed',
                   delegated_reply_message_id = ?,
                   updated_at = ?
               WHERE token = ?""",
            (delegated_reply_message_id, datetime.now().isoformat(), token),
        )
        self.conn.commit()

    def mark_failed(self, token: str, *, error: str, status: str = "failed") -> None:
        self.conn.execute(
            """UPDATE delegation_tasks
               SET status = ?, error = ?, updated_at = ?
               WHERE token = ?""",
            (status, error, datetime.now().isoformat(), token),
        )
        self.conn.commit()
