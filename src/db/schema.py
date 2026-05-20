"""SQLite schema initialization and migrations."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "sessions.db"


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
        conn.execute("PRAGMA busy_timeout=30000")
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
            cursor_session_id TEXT,
            active_engine TEXT DEFAULT 'pi',
            opencode_agent TEXT DEFAULT 'bridge',
            model_id TEXT DEFAULT 'glm_vllm/glm-4.7-flash',
            vllm_base_url TEXT,
            reasoning_mode TEXT DEFAULT 'normal',
            dispatcher_jid TEXT,
            owner_jid TEXT,
            room_jid TEXT,
            tmux_name TEXT,
            created_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_collaborators (
            session_name TEXT NOT NULL,
            participant_jid TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (session_name, participant_jid),
            FOREIGN KEY (session_name) REFERENCES sessions(name) ON DELETE CASCADE
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
            FOREIGN KEY (session_name) REFERENCES sessions(name) ON DELETE CASCADE
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
            FOREIGN KEY (session_name) REFERENCES sessions(name) ON DELETE CASCADE
        )
    """)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_messages_session_name_id ON session_messages(session_name, id DESC)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS delegation_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            parent_session TEXT NOT NULL,
            dispatcher_name TEXT NOT NULL,
            dispatcher_jid TEXT NOT NULL,
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            delegated_session TEXT,
            delegated_user_message_id INTEGER,
            delegated_reply_message_id INTEGER,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (parent_session) REFERENCES sessions(name) ON DELETE CASCADE,
            FOREIGN KEY (delegated_session) REFERENCES sessions(name) ON DELETE SET NULL
        )
    """)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_delegation_tasks_parent_created ON delegation_tasks(parent_session, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_delegation_tasks_status_updated ON delegation_tasks(status, updated_at DESC)"
    )

    # Migrations for existing databases
    migrations = [
        ("opencode_session_id", "TEXT"),
        ("active_engine", "TEXT DEFAULT 'pi'"),
        ("opencode_agent", "TEXT DEFAULT 'bridge'"),
        ("model_id", "TEXT DEFAULT 'glm_vllm/glm-4.7-flash'"),
        ("vllm_base_url", "TEXT"),
        ("reasoning_mode", "TEXT DEFAULT 'normal'"),
        ("dispatcher_jid", "TEXT"),
        ("owner_jid", "TEXT"),
        ("room_jid", "TEXT"),
        ("pi_session_id", "TEXT"),
        ("cursor_session_id", "TEXT"),
    ]
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
    }
    for col_name, col_type in migrations:
        if col_name in existing_cols:
            continue
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_owner_last_active ON sessions(owner_jid, last_active DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_room_jid ON sessions(room_jid)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_collaborators_session ON session_collaborators(session_name)"
    )

    # Backfill ownership for existing rows from legacy single-user config.
    default_owner = (os.getenv("XMPP_RECIPIENT", "") or "").split("/", 1)[0].strip()
    if default_owner:
        try:
            conn.execute(
                "UPDATE sessions SET owner_jid = ? WHERE owner_jid IS NULL",
                (default_owner,),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass

    ralph_migrations = [
        ("wait_seconds", "REAL DEFAULT 2.0"),
    ]
    existing_ralph_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(ralph_loops)").fetchall()
    }
    for col_name, col_type in ralph_migrations:
        if col_name in existing_ralph_cols:
            continue
        try:
            conn.execute(f"ALTER TABLE ralph_loops ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

    conn.commit()
    return conn
