#!/usr/bin/env python3
"""Database initialization and repositories for XMPP bridge.

Provides:
- Schema initialization with migrations
- SessionRepository: CRUD for sessions table
- RalphLoopRepository: CRUD for ralph_loops table
- MessageRepository: CRUD for session_messages table
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.engines import OPENCODE_MODEL_DEFAULT, OPENCODE_MODEL_GPT

DB_PATH = Path(__file__).parent.parent / "sessions.db"


@dataclass
class Session:
    """Session record."""

    name: str
    xmpp_jid: str
    xmpp_password: str
    claude_session_id: str | None
    opencode_session_id: str | None
    active_engine: str
    opencode_agent: str
    model_id: str
    reasoning_mode: str
    dispatcher_jid: str | None
    tmux_name: str | None
    created_at: str
    last_active: str
    status: str


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


@dataclass
class SessionMessage:
    """Session message record."""

    id: int
    session_name: str
    role: str
    content: str
    engine: str
    created_at: str


class SessionRepository:
    """Repository for sessions table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            name=row["name"],
            xmpp_jid=row["xmpp_jid"],
            xmpp_password=row["xmpp_password"],
            claude_session_id=row["claude_session_id"],
            opencode_session_id=row["opencode_session_id"],
            active_engine=row["active_engine"] or "opencode",
            opencode_agent=row["opencode_agent"] or "bridge",
            model_id=row["model_id"] or OPENCODE_MODEL_DEFAULT,
            reasoning_mode=row["reasoning_mode"] or "high",
            dispatcher_jid=row["dispatcher_jid"] if "dispatcher_jid" in row.keys() else None,
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

    def create(
        self,
        name: str,
        xmpp_jid: str,
        xmpp_password: str,
        tmux_name: str,
        model_id: str = OPENCODE_MODEL_DEFAULT,
        opencode_agent: str = "bridge",
        active_engine: str = "opencode",
        reasoning_mode: str = "high",
        dispatcher_jid: str | None = None,
    ) -> Session:
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO sessions
               (name, xmpp_jid, xmpp_password, tmux_name, created_at, last_active,
                model_id, opencode_agent, active_engine, reasoning_mode, dispatcher_jid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                xmpp_jid,
                xmpp_password,
                tmux_name,
                now,
                now,
                model_id,
                opencode_agent,
                active_engine,
                reasoning_mode,
                dispatcher_jid,
            ),
        )
        self.conn.commit()
        created = self.get(name)
        if not created:
            raise RuntimeError(f"Failed to load newly created session: {name}")
        return created

    def update_last_active(self, name: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET last_active = ? WHERE name = ?",
            (datetime.now().isoformat(), name),
        )
        self.conn.commit()

    def update_engine(self, name: str, engine: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET active_engine = ? WHERE name = ?",
            (engine, name),
        )
        self.conn.commit()

    def update_reasoning_mode(self, name: str, mode: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET reasoning_mode = ? WHERE name = ?",
            (mode, name),
        )
        self.conn.commit()

    def update_model(self, name: str, model_id: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET model_id = ? WHERE name = ?",
            (model_id, name),
        )
        self.conn.commit()

    def update_claude_session_id(self, name: str, session_id: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET claude_session_id = ? WHERE name = ?",
            (session_id, name),
        )
        self.conn.commit()

    def update_opencode_session_id(self, name: str, session_id: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET opencode_session_id = ? WHERE name = ?",
            (session_id, name),
        )
        self.conn.commit()

    def reset_claude_session(self, name: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET claude_session_id = NULL WHERE name = ?",
            (name,),
        )
        self.conn.commit()

    def reset_opencode_session(self, name: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET opencode_session_id = NULL WHERE name = ?",
            (name,),
        )
        self.conn.commit()

    def close(self, name: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET status = 'closed' WHERE name = ?",
            (name,),
        )
        self.conn.commit()


class RalphLoopRepository:
    """Repository for ralph_loops table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

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

    def create(
        self,
        session_name: str,
        prompt: str,
        max_iterations: int = 0,
        completion_promise: str | None = None,
        wait_seconds: float = 2.0,
    ) -> int:
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

    def update_progress(
        self,
        loop_id: int,
        current_iteration: int,
        total_cost: float,
        status: str = "running",
    ) -> None:
        finished_at = datetime.now().isoformat() if status != "running" else None
        self.conn.execute(
            """UPDATE ralph_loops
               SET current_iteration = ?, total_cost = ?, status = ?,
                   finished_at = COALESCE(?, finished_at)
               WHERE id = ?""",
            (current_iteration, total_cost, status, finished_at, loop_id),
        )
        self.conn.commit()


class MessageRepository:
    """Repository for session_messages table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _row_to_message(self, row: sqlite3.Row) -> SessionMessage:
        return SessionMessage(
            id=row["id"],
            session_name=row["session_name"],
            role=row["role"],
            content=row["content"],
            engine=row["engine"],
            created_at=row["created_at"],
        )

    def add(
        self,
        session_name: str,
        role: str,
        content: str,
        engine: str,
    ) -> None:
        self.conn.execute(
            """INSERT INTO session_messages
               (session_name, role, content, engine, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_name, role, content, engine, datetime.now().isoformat()),
        )
        self.conn.commit()

    def list_recent(self, session_name: str, limit: int = 40) -> list[SessionMessage]:
        rows = self.conn.execute(
            """SELECT * FROM session_messages
               WHERE session_name = ?
               ORDER BY id DESC
               LIMIT ?""",
            (session_name, limit),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]


def init_db() -> sqlite3.Connection:
    """Initialize SQLite database with schema and migrations."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Pragmas: reduce write amplification/lock pain.
    # Directory browsing does frequent reads while sessions append messages.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
    except sqlite3.OperationalError:
        # Best-effort; some environments may reject specific pragmas.
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            name TEXT PRIMARY KEY,
            xmpp_jid TEXT UNIQUE NOT NULL,
            xmpp_password TEXT NOT NULL,
            claude_session_id TEXT,
            opencode_session_id TEXT,
            active_engine TEXT DEFAULT 'opencode',
            opencode_agent TEXT DEFAULT 'bridge',
            model_id TEXT DEFAULT 'glm_vllm/glm-4.7-flash-heretic.Q8_0.gguf',
            reasoning_mode TEXT DEFAULT 'high',
            tmux_name TEXT,
            created_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        )
    """)

    # Indexes: session listing is on the hot path (directory/disco browsing).
    # Without these, SQLite scans/sorts the whole table on each request.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_status_last_active ON sessions(status, last_active DESC)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ralph_loops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            completion_promise TEXT,
            max_iterations INTEGER DEFAULT 0,
            wait_seconds REAL DEFAULT 2.0,
            current_iteration INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0,
            status TEXT DEFAULT 'running',
            started_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY (session_name) REFERENCES sessions(name)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            engine TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_name) REFERENCES sessions(name)
        )
    """)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_messages_session_name_id ON session_messages(session_name, id DESC)"
    )

    # Migrations for existing databases
    migrations = [
        ("opencode_session_id", "TEXT"),
        ("active_engine", "TEXT DEFAULT 'opencode'"),
        ("opencode_agent", "TEXT DEFAULT 'bridge'"),
        ("model_id", f"TEXT DEFAULT '{OPENCODE_MODEL_DEFAULT}'"),
        ("reasoning_mode", "TEXT DEFAULT 'high'"),
        ("dispatcher_jid", "TEXT"),
    ]
    for col_name, col_type in migrations:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass

    ralph_migrations = [
        ("wait_seconds", "REAL DEFAULT 2.0"),
    ]
    for col_name, col_type in ralph_migrations:
        try:
            conn.execute(f"ALTER TABLE ralph_loops ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass

    conn.commit()
    return conn
