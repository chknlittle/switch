"""Shared runner pipeline helpers.

This module provides reusable orchestration primitives for runners that combine:
- a background producer task feeding an asyncio.Queue of raw events
- a background request task (e.g. message POST)
- a parser/processor that turns raw events into yielded Events

Engines supply:
- how to extract the session ID from raw events
- how to parse raw events into Events
- optional question handling
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, TypeVar

from src.runners.base import RunState


T = TypeVar("T")


@dataclass
class JSONLineStats:
    emitted_any: bool = False
    non_json_lines: list[str] = field(default_factory=list)


async def iter_queue_pipeline(
    *,
    event_queue: asyncio.Queue[dict],
    session_id: str,
    state: RunState,
    parse_event: Callable[[dict, RunState], T | list[T] | None],
    extract_session_id: Callable[[dict], str | None],
    sse_task: asyncio.Task,
    message_task: asyncio.Task,
    should_cancel: Callable[[], bool],
    idle_timeout_s: float,
    is_done: Callable[[RunState], bool],
) -> AsyncIterator[T]:
    """Drive a queue-based runner until completion."""

    message_done_at: float | None = None
    last_event_at = time.monotonic()

    while True:
        if should_cancel():
            break

        if sse_task.done() and not sse_task.cancelled():
            exc = sse_task.exception()
            if exc:
                raise exc

        if message_task.done() and is_done(state):
            break

        if message_task.done() and message_done_at is None:
            message_done_at = time.monotonic()

        if message_done_at is not None and not is_done(state):
            if (time.monotonic() - last_event_at) >= idle_timeout_s:
                break

        try:
            payload = await asyncio.wait_for(event_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue

        if not isinstance(payload, dict):
            continue

        payload_session = extract_session_id(payload)
        if payload_session and payload_session != session_id:
            continue

        parsed = parse_event(payload, state)
        if not parsed:
            continue

        items: list[T]
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [parsed]

        if not items:
            continue

        last_event_at = time.monotonic()

        for item in items:
            yield item


async def iter_json_line_pipeline(
    *,
    byte_stream: AsyncIterator[bytes],
    state: RunState,
    parse_event: Callable[[dict, RunState], list[T]],
    stats: JSONLineStats,
    non_json_limit: int = 50,
) -> AsyncIterator[T]:
    """Parse a JSON-lines byte stream and yield parsed events."""

    async for raw_line in byte_stream:
        line = raw_line.decode(errors="replace").strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if len(stats.non_json_lines) < non_json_limit:
                stats.non_json_lines.append(line)
            continue

        if not isinstance(event, dict):
            continue

        for result in parse_event(event, state):
            stats.emitted_any = True
            yield result
