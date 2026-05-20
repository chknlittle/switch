"""Shared asyncio write locks keyed by connection."""

from __future__ import annotations

import asyncio
import sqlite3

_write_locks: dict[int, asyncio.Lock] = {}


def shared_write_lock(conn: sqlite3.Connection) -> asyncio.Lock:
    """Return a single shared write lock for a given connection.

    All repositories sharing the same connection MUST use this lock so that
    concurrent writes are serialized.  Keyed by connection id().
    """
    key = id(conn)
    lock = _write_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _write_locks[key] = lock
    return lock
