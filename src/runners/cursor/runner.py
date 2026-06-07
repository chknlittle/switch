"""Cursor Agent ACP runner."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from src.runners.base import BaseRunner, RunState
from src.runners.cursor.config import CursorConfig
from src.runners.cursor.events import extract_session_id
from src.runners.cursor.processor import CursorEventProcessor
from src.runners.cursor.transport import CursorACPTransport
from src.runners.pipeline import iter_queue_pipeline
from src.runners.ports import RunnerEvent
from src.runners.timeouts import post_message_idle_timeout_s

log = logging.getLogger("cursor.runner")

Event = RunnerEvent


class CursorACPRunner(BaseRunner):
    """Runs Cursor Agent through ACP (`agent acp`) over stdio."""

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        config: CursorConfig | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)
        self._config = config or CursorConfig()
        self._transport = CursorACPTransport(self._config)
        self._processor = CursorEventProcessor(log_response=self._log_response)

    def _argv(self) -> list[str]:
        return [self._config.resolve_bin(), "--model", self._config.resolve_model(), "acp"]

    async def run(self, prompt: str, session_id: str | None = None) -> AsyncIterator[Event]:
        state = RunState()
        self._log_prompt(prompt)

        prompt_task: asyncio.Task | None = None
        reader_task: asyncio.Task | None = None
        cursor_session_id: str | None = None

        try:
            client = await self._transport.start(argv=self._argv(), cwd=self.working_dir)
            cursor_session_id = await self._transport.open_session(
                client,
                session_id=session_id,
                cwd=self.working_dir,
            )
            state.session_id = cursor_session_id
            yield ("session_id", cursor_session_id)

            prompt_task = self._transport.start_prompt(
                client,
                session_id=cursor_session_id,
                prompt=prompt,
            )
            reader_task = self._transport.reader_task()
            if reader_task is None:
                raise RuntimeError("Cursor ACP stdout reader did not start")

            idle_timeout_s = post_message_idle_timeout_s(
                override=self._config.post_message_idle_timeout_s,
            )

            def parse_event(msg: dict, run_state: RunState) -> Event | list[Event] | None:
                assert cursor_session_id is not None
                return self._processor.parse_event(
                    msg,
                    run_state,
                    cursor_session_id=cursor_session_id,
                )

            async for event in iter_queue_pipeline(
                event_queue=client.events,
                session_id=cursor_session_id,
                state=state,
                parse_event=parse_event,
                extract_session_id=extract_session_id,
                sse_task=reader_task,
                message_task=prompt_task,
                should_cancel=lambda: self._transport.cancelled,
                idle_timeout_s=idle_timeout_s,
                is_done=lambda s: s.saw_result or s.saw_error,
            ):
                event_type, data = event
                if event_type == "permission":
                    await self._transport.allow_permission(data)  # type: ignore[arg-type]
                    continue
                yield event

            if self._transport.cancelled:
                yield ("cancelled", None)
                return

            if prompt_task.cancelled():
                yield ("cancelled", None)
                return

            exc = prompt_task.exception()
            if exc is not None:
                raise exc

            if not state.saw_result and not state.saw_error:
                state.saw_result = True
                yield (
                    "result",
                    self._processor.make_result(state, prompt_task.result()),
                )
        except Exception as e:
            log.exception("Cursor ACP runner error")
            yield ("error", str(e))
        finally:
            await self._transport.cleanup(prompt_task=prompt_task)

    def cancel(self) -> None:
        self._transport.cancel()

    async def cleanup(self) -> None:
        await self._transport.cleanup(prompt_task=None)
