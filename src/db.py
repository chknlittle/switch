#!/usr/bin/env python3
"""Database initialization and schema for XMPP bridge."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sessions.db"


def init_db() -> sqlite3.Connection:
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            name TEXT PRIMARY KEY,
            xmpp_jid TEXT UNIQUE NOT NULL,
            xmpp_password TEXT NOT NULL,
            claude_session_id TEXT,
            opencode_session_id TEXT,
            active_engine TEXT DEFAULT 'opencode',
            opencode_agent TEXT DEFAULT 'bridge',
            model_id TEXT DEFAULT 'openai/gpt-5.2-codex',
            reasoning_mode TEXT DEFAULT 'normal',
            tmux_name TEXT,
            created_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ralph_loops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            completion_promise TEXT,
            max_iterations INTEGER DEFAULT 0,
            current_iteration INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0,
            status TEXT DEFAULT 'running',
            started_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY (session_name) REFERENCES sessions(name)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            engine TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_name) REFERENCES sessions(name)
        )
    """
    )
    # Migration: add columns for existing databases
    for column in [
        ("opencode_session_id", "TEXT"),
        ("active_engine", "TEXT DEFAULT 'opencode'"),
        ("opencode_agent", "TEXT DEFAULT 'bridge'"),
        ("model_id", "TEXT DEFAULT 'openai/gpt-5.2-codex'"),
        ("reasoning_mode", "TEXT DEFAULT 'normal'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {column[0]} {column[1]}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn
