"""Database initialization and repositories for XMPP bridge.

Provides:
- Schema initialization with migrations
- SessionRepository: CRUD for sessions table
- RalphLoopRepository: CRUD for ralph_loops table
- MessageRepository: CRUD for session_messages table
- DelegationTaskRepository: CRUD for delegation_tasks table
"""

from __future__ import annotations

from src.db.repos.delegation import DelegationTask, DelegationTaskRepository
from src.db.repos.messages import MessageRepository, SessionMessage
from src.db.repos.ralph import RalphLoop, RalphLoopRepository
from src.db.repos.sessions import Session, SessionRepository
from src.db.schema import DB_PATH, init_db
from src.db.signals import (
    get_message_signal_version,
    notify_message_written,
    wait_for_message_signal,
)

__all__ = [
    "DB_PATH",
    "DelegationTask",
    "DelegationTaskRepository",
    "MessageRepository",
    "RalphLoop",
    "RalphLoopRepository",
    "Session",
    "SessionMessage",
    "SessionRepository",
    "get_message_signal_version",
    "init_db",
    "notify_message_written",
    "wait_for_message_signal",
]
