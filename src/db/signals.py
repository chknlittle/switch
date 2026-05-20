"""In-process message-write signals for async wait/notify."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass

_message_signals: dict[int, "_MessageSignal"] = {}


@dataclass
class _MessageSignal:
    condition: asyncio.Condition
    version: int = 0


def _shared_message_signal(conn: sqlite3.Connection) -> _MessageSignal:
    key = id(conn)
    signal = _message_signals.get(key)
    if signal is None:
        signal = _MessageSignal(condition=asyncio.Condition())
        _message_signals[key] = signal
    return signal


def get_message_signal_version(conn: sqlite3.Connection) -> int:
    return _shared_message_signal(conn).version


async def notify_message_written(conn: sqlite3.Connection) -> None:
    signal = _shared_message_signal(conn)
    async with signal.condition:
        signal.version += 1
        signal.condition.notify_all()


async def wait_for_message_signal(
    conn: sqlite3.Connection, *, after_version: int, timeout_s: float
) -> int | None:
    signal = _shared_message_signal(conn)
    if signal.version > after_version:
        return signal.version

    async with signal.condition:
        if signal.version > after_version:
            return signal.version
        try:
            await asyncio.wait_for(
                signal.condition.wait_for(lambda: signal.version > after_version),
                timeout=max(0.0, timeout_s),
            )
        except asyncio.TimeoutError:
            return None
        return signal.version
