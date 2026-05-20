"""Ralph loop repository and model."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from src.db.locks import shared_write_lock


@dataclass
class RalphLoop:
    """Ralph loop record."""

    id: int
    session_name: str
    prompt: str
    completion_promise: str | None
    max_iterations: int
    wait_seconds: float
    current_iteration: int
    total_cost: float
    status: str
    started_at: str
    finished_at: str | None


class RalphLoopRepository:
    """Repository for ralph_loops table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._write_lock = shared_write_lock(conn)

    def _row_to_ralph_loop(self, row: sqlite3.Row) -> RalphLoop:
        return RalphLoop(
            id=row["id"],
            session_name=row["session_name"],
            prompt=row["prompt"],
            completion_promise=row["completion_promise"],
            max_iterations=row["max_iterations"] or 0,
            wait_seconds=row["wait_seconds"]
            if row["wait_seconds"] is not None
            else 2.0,
            current_iteration=row["current_iteration"] or 0,
            total_cost=row["total_cost"] or 0.0,
            status=row["status"] or "running",
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def get_latest(self, session_name: str) -> RalphLoop | None:
        row = self.conn.execute(
            """SELECT * FROM ralph_loops
               WHERE session_name = ?
               ORDER BY started_at DESC LIMIT 1""",
            (session_name,),
        ).fetchone()
        return self._row_to_ralph_loop(row) if row else None

    async def create(
        self,
        session_name: str,
        prompt: str,
        max_iterations: int = 0,
        completion_promise: str | None = None,
        wait_seconds: float = 2.0,
    ) -> int:
        async with self._write_lock:
            cursor = self.conn.execute(
                """INSERT INTO ralph_loops
                   (session_name, prompt, completion_promise, max_iterations, wait_seconds, started_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_name,
                    prompt,
                    completion_promise,
                    max_iterations,
                    wait_seconds,
                    datetime.now().isoformat(),
                ),
            )
            self.conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to create ralph loop (no rowid)")
            return int(cursor.lastrowid)

    async def update_progress(
        self,
        loop_id: int,
        current_iteration: int,
        total_cost: float,
        status: str = "running",
    ) -> None:
        finished_at = datetime.now().isoformat() if status != "running" else None
        async with self._write_lock:
            self.conn.execute(
                """UPDATE ralph_loops
                   SET current_iteration = ?, total_cost = ?, status = ?,
                       finished_at = COALESCE(?, finished_at)
                   WHERE id = ?""",
                (current_iteration, total_cost, status, finished_at, loop_id),
            )
            self.conn.commit()
