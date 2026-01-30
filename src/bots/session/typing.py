"""Typing indicator keepalive.

Slixmpp clients often clear "composing" after a short time. This helper
refreshes typing while work is in-flight.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable


class TypingIndicator:
    def __init__(
        self,
        *,
        send_typing: Callable[[], None],
        is_active: Callable[[], bool],
        is_shutting_down: Callable[[], bool],
    ):
        self._send_typing = send_typing
        self._is_active = is_active
        self._is_shutting_down = is_shutting_down

        self._task: asyncio.Task | None = None
        self._last_sent = 0.0

    def maybe_send(self, *, min_interval_s: float = 5.0) -> None:
        if self._is_shutting_down():
            return
        now = time.monotonic()
        if now - self._last_sent < min_interval_s:
            return
        self._last_sent = now
        self._send_typing()

    async def _loop(self, *, interval_s: float) -> None:
        try:
            while self._is_active() and not self._is_shutting_down():
                self.maybe_send(min_interval_s=0.0)
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

    def start(self, *, interval_s: float = 15.0) -> None:
        if self._task and not self._task.done():
            self.maybe_send(min_interval_s=0.0)
            return
        self.maybe_send(min_interval_s=0.0)
        self._task = asyncio.create_task(self._loop(interval_s=interval_s))

    def stop(self) -> None:
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
