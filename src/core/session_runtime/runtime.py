"""SessionRuntime.

Owns message queue serialization, cancellation, and work dispatch.
Engine, Ralph, and question handling live in companion mixins.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.attachments import Attachment
from src.runners import Runner

from src.core.session_runtime.api import (
    EventSinkPort,
    OutboundMessage,
    ProcessingChanged,
    RalphConfig,
    RalphStatus,
    SessionEvent,
)
from src.core.session_runtime.engine_runner import EngineRunnerMixin
from src.core.session_runtime.ports import (
    AttachmentPromptPort,
    HistoryPort,
    MessageStorePort,
    RalphLoopStorePort,
    RunnerFactoryPort,
    SessionStorePort,
)
from src.core.session_runtime.questions import QuestionHandlerMixin
from src.core.session_runtime.ralph_runner import RalphRunnerMixin

log = logging.getLogger("session_runtime")


@dataclass(frozen=True)
class _WorkItem:
    generation: int
    kind: str  # "message" | "ralph"
    body: str
    attachments: list[Attachment] | None
    trigger_response: bool
    scheduled: bool
    ralph: RalphConfig | None = None
    done: asyncio.Future[None] | None = None
    enqueued_at: float = field(default_factory=time.monotonic)


class SessionRuntime(QuestionHandlerMixin, EngineRunnerMixin, RalphRunnerMixin):
    def __init__(
        self,
        *,
        session_name: str,
        working_dir: str,
        output_dir: Path,
        sessions: SessionStorePort,
        messages: MessageStorePort,
        events: EventSinkPort,
        runner_factory: RunnerFactoryPort,
        history: HistoryPort,
        prompt: AttachmentPromptPort,
        ralph_loops: RalphLoopStorePort | None = None,
        infer_meta_tool_from_summary: Callable[[str], str | None],
        startup_prompt_context: Callable[[], str] | None = None,
    ):
        self.session_name = session_name
        self.working_dir = working_dir
        self.output_dir = output_dir
        self._sessions = sessions
        self._messages = messages
        self._events = events
        self._runner_factory = runner_factory
        self._history = history
        self._prompt = prompt
        self._ralph_loops = ralph_loops
        self._infer_meta_tool_from_summary = infer_meta_tool_from_summary
        self._startup_prompt_context = startup_prompt_context

        self._generation = 0
        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._task: asyncio.Task | None = None

        self.processing = False
        self.shutting_down = False

        self.runner: Runner | None = None
        self._run_task: asyncio.Task | None = None
        self._startup_prompt_context_injected = False

        self._pending_question_answers: dict[str, asyncio.Future] = {}

        self._ralph_status: RalphStatus | None = None
        self._ralph_stop_requested = False
        self._ralph_wake = asyncio.Event()
        self._ralph_injected_prompt: str | None = None

        self._pending_handoff: tuple[str, str] | None = None
        self._context_prefix: str | None = None

        self._usage_tokens_total: dict[str, int] = {
            "claude": 0,
            "pi": 0,
            "opencode": 0,
            "cursor": 0,
        }
        self._usage_cost_total: dict[str, float] = {
            "claude": 0.0,
            "pi": 0.0,
            "opencode": 0.0,
            "cursor": 0.0,
        }
        self._last_remote_session_id: dict[str, str | None] = {
            "claude": None,
            "pi": None,
            "opencode": None,
            "cursor": None,
        }

        self._last_active_written_at: float = 0.0
        self._last_active_min_interval_s: float = 10.0

    def ensure_running(self) -> None:
        if self.shutting_down:
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())

    def pending_count(self) -> int:
        return self._queue.qsize()

    async def enqueue(
        self,
        body: str,
        attachments: list[Attachment] | None,
        *,
        trigger_response: bool,
        scheduled: bool,
        wait: bool,
    ) -> None:
        if self.shutting_down:
            return

        done: asyncio.Future[None] | None = None
        if wait:
            done = asyncio.get_running_loop().create_future()

        item = _WorkItem(
            generation=self._generation,
            kind="message",
            body=body,
            attachments=list(attachments) if attachments else None,
            trigger_response=trigger_response,
            scheduled=scheduled,
            done=done,
        )
        await self._queue.put(item)
        self.ensure_running()

        if done is not None:
            await done

    def set_context_prefix(self, text: str) -> None:
        """Store context to prepend to the next real user prompt."""
        self._context_prefix = text

    async def run_handoff(self, target_engine: str, prompt: str) -> None:
        """Run a prompt through a specific engine without changing the session's active engine."""
        if self.shutting_down:
            return

        self._pending_handoff = (target_engine, prompt)
        try:
            await self.enqueue(
                "",
                None,
                trigger_response=True,
                scheduled=False,
                wait=True,
            )
        finally:
            self._pending_handoff = None

    def cancel_queued(self) -> bool:
        """Drop queued items; ignore anything already dequeued."""
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

    def cancel_operations(self, *, notify: bool = False) -> bool:
        cancelled_any = False

        if self.cancel_queued():
            cancelled_any = True

        if self.runner and self.processing:
            cancelled_any = True
            self.runner.cancel()

        if self._ralph_status and self._ralph_status.status in {
            "queued",
            "running",
            "stopping",
        }:
            cancelled_any = True
            self._ralph_stop_requested = True
            self._ralph_wake.set()

        if self._run_task and not self._run_task.done():
            cancelled_any = True
            self._run_task.cancel()

        if self._pending_handoff is not None:
            self._pending_handoff = None
            cancelled_any = True

        self._context_prefix = None

        for fut in list(self._pending_question_answers.values()):
            if fut and not fut.done():
                fut.cancel()

        if notify and cancelled_any:
            self._emit_nowait(OutboundMessage("Cancelling current work..."))

        return cancelled_any

    def get_ralph_status(self) -> RalphStatus | None:
        return self._ralph_status

    def request_ralph_stop(self) -> bool:
        """Ask Ralph to stop after the current iteration."""
        if not self._ralph_status or self._ralph_status.status not in {
            "queued",
            "running",
        }:
            return False
        self._ralph_stop_requested = True
        self._ralph_status.status = "stopping"
        self._ralph_wake.set()
        return True

    def inject_ralph_prompt(self, prompt: str) -> bool:
        """Inject a user prompt into the running Ralph loop."""
        if not self._ralph_status or self._ralph_status.status not in {
            "running",
        }:
            return False
        self._ralph_injected_prompt = prompt
        self._ralph_wake.set()
        return True

    async def start_ralph(self, cfg: RalphConfig, *, wait: bool = False) -> None:
        """Enqueue a Ralph loop."""
        if self.shutting_down:
            return

        done: asyncio.Future[None] | None = None
        if wait:
            done = asyncio.get_running_loop().create_future()

        self._ralph_stop_requested = False
        self._ralph_wake = asyncio.Event()
        self._ralph_status = RalphStatus(
            status="queued",
            current_iteration=0,
            max_iterations=max(0, int(cfg.max_iterations or 0)),
            wait_seconds=float(cfg.wait_seconds or 0.0),
            completion_promise=(cfg.completion_promise or None),
            total_cost=0.0,
        )

        item = _WorkItem(
            generation=self._generation,
            kind="ralph",
            body=cfg.prompt,
            attachments=None,
            trigger_response=True,
            scheduled=False,
            ralph=cfg,
            done=done,
        )
        await self._queue.put(item)
        self.ensure_running()
        if done is not None:
            await done

    def shutdown(self) -> None:
        if self.shutting_down:
            return
        self.shutting_down = True
        self.cancel_operations(notify=False)
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
        self.cancel_queued()

    def answer_question(self, answer: object, *, request_id: str | None = None) -> bool:
        if not self._pending_question_answers:
            return False
        rid = request_id or list(self._pending_question_answers.keys())[-1]
        fut = self._pending_question_answers.get(rid)
        if fut and not fut.done():
            fut.set_result(answer)
            return True
        return False

    def _set_processing(self, active: bool) -> None:
        self.processing = active
        self._emit_nowait(ProcessingChanged(active=active))

    def _emit_nowait(self, event: SessionEvent) -> None:
        """Best-effort emit from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._events.emit(event))
        task.add_done_callback(self._emit_task_done)

    @staticmethod
    def _emit_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.warning("emit_nowait failed: %s", exc)

    async def _emit(self, event: SessionEvent) -> None:
        await self._events.emit(event)

    async def _loop(self) -> None:
        try:
            while not self.shutting_down:
                item = await self._queue.get()
                if item.generation != self._generation:
                    if item.done and not item.done.done():
                        item.done.set_exception(asyncio.CancelledError())
                    continue

                if item.trigger_response:
                    self._set_processing(True)

                try:
                    await self._process_one(item)
                except asyncio.CancelledError:
                    if self.shutting_down:
                        raise
                except Exception as e:
                    log.exception("SessionRuntime loop error for %s", self.session_name)
                    if item.trigger_response:
                        await self._emit(
                            OutboundMessage(f"Error: {type(e).__name__}: {e}")
                        )
                finally:
                    if item.trigger_response:
                        self._set_processing(False)
                    if item.done and not item.done.done():
                        item.done.set_result(None)
        except asyncio.CancelledError:
            return

    async def _process_one(self, item: _WorkItem) -> None:
        if item.kind == "ralph":
            cfg = item.ralph or RalphConfig(prompt=item.body)
            await self._run_ralph(cfg)
            return

        if self._pending_handoff is not None and not item.body:
            target_engine, handoff_prompt = self._pending_handoff
            self._pending_handoff = None
            self._context_prefix = None
            session = self._sessions.get(self.session_name)
            if not session:
                await self._emit(OutboundMessage("Session not found."))
                return
            await self._messages.add(
                self.session_name,
                "user",
                handoff_prompt[:500],
                target_engine,
            )
            try:
                self._create_runner_for_engine(target_engine, session)
                await self._run_engine_generic(
                    target_engine,
                    session,
                    handoff_prompt,
                    skip_runner_create=True,
                    ephemeral=True,
                )
            except Exception as e:
                await self._emit(
                    OutboundMessage(f"Handoff error: {type(e).__name__}: {e}")
                )
            return

        session = self._sessions.get(self.session_name)
        if not session:
            await self._emit(OutboundMessage("Session not found in database."))
            return

        now = time.monotonic()
        if (now - self._last_active_written_at) >= self._last_active_min_interval_s:
            await self._sessions.update_last_active(self.session_name)
            self._last_active_written_at = now

        body_for_history = self._prompt.augment_prompt(item.body, item.attachments)
        self._history.append_to_history(
            body_for_history, self.working_dir, session.claude_session_id
        )
        self._history.log_activity(item.body, session=self.session_name, source="xmpp")
        await self._messages.add(
            self.session_name, "user", body_for_history, session.active_engine
        )

        if not item.trigger_response:
            return

        engine = (session.active_engine or "pi").strip().lower()

        run_prompt = body_for_history
        if self._context_prefix:
            run_prompt = self._context_prefix + "\n\n" + body_for_history
            self._context_prefix = None

        if not self._startup_prompt_context_injected and self._startup_prompt_context:
            try:
                startup = (self._startup_prompt_context() or "").strip()
            except Exception:
                startup = ""
            if startup:
                run_prompt = f"{startup}\n\n{run_prompt}"
                self._startup_prompt_context_injected = True

        self._run_task = asyncio.create_task(
            self._run_engine(engine=engine, session=session, prompt=run_prompt)
        )
        try:
            await self._run_task
        finally:
            self._run_task = None
