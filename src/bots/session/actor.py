"""SessionBot actor loop.

This isolates the "one thing at a time" concurrency model from XMPP handlers.
The SessionBot owns parsing, DB writes, and runner details; the actor owns
queueing, serialization, and cancellation of queued work.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass(frozen=True)
class QueuedMessage:
    generation: int
    body: str
    attachments: object
    trigger_response: bool
    scheduled: bool
    done: asyncio.Future[None] | None = None
    enqueued_at: float = field(default_factory=time.monotonic)


class SessionActor:
    def __init__(
        self,
        *,
        process_one: Callable[[str, bool, object], Awaitable[None]],
        on_error: Callable[[Exception], None],
        start_work: Callable[[], None],
        finish_work: Callable[[], None],
        is_shutting_down: Callable[[], bool],
        log_exception: Callable[[str], None],
    ):
        self._process_one = process_one
        self._on_error = on_error
        self._start_work = start_work
        self._finish_work = finish_work
        self._is_shutting_down = is_shutting_down
        self._log_exception = log_exception

        self._generation = 0
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue[QueuedMessage] = asyncio.Queue()

    def ensure_running(self) -> None:
        if self._is_shutting_down():
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    def pending_count(self) -> int:
        return self._queue.qsize()

    async def enqueue(
        self,
        body: str,
        attachments: object,
        *,
        trigger_response: bool,
        scheduled: bool,
        wait: bool,
    ) -> None:
        if self._is_shutting_down():
            return

        done: asyncio.Future[None] | None = None
        if wait:
            done = asyncio.get_running_loop().create_future()

        item = QueuedMessage(
            generation=self._generation,
            body=body,
            attachments=attachments,
            trigger_response=trigger_response,
            scheduled=scheduled,
            done=done,
        )

        await self._queue.put(item)
        self.ensure_running()

        if done is not None:
            await done

    def cancel_queued(self) -> bool:
        """Drop queued items; ignore anything already dequeued but not started."""
        self._generation += 1
        dropped_any = False
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                dropped_any = True
                if item.done and not item.done.done():
                    item.done.set_exception(asyncio.CancelledError())
        return dropped_any

    def shutdown(self) -> None:
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()

        self.cancel_queued()

    async def _loop(self) -> None:
        try:
            while not self._is_shutting_down():
                item = await self._queue.get()

                if item.generation != self._generation:
                    if item.done and not item.done.done():
                        item.done.set_exception(asyncio.CancelledError())
                    continue

                if item.trigger_response:
                    self._start_work()

                try:
                    await self._process_one(item.body, item.trigger_response, item.attachments)
                except asyncio.CancelledError:
                    if self._is_shutting_down():
                        raise
                except Exception as e:
                    self._log_exception("SessionActor loop error")
                    self._on_error(e)
                finally:
                    if item.trigger_response:
                        self._finish_work()
                    if item.done and not item.done.done():
                        item.done.set_result(None)
        except asyncio.CancelledError:
            return
