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
import time
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from src.runners.base import RunState


T = TypeVar("T")


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
    is_question: Callable[[T], bool] | None = None,
    handle_question: Callable[[T], Awaitable[None]] | None = None,
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
            if is_question and handle_question and is_question(item):
                yield item
                await handle_question(item)
            else:
                yield item
