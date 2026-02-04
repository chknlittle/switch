"""SessionRuntime.

This is the single place that owns:
- serialization (message queue)
- cancellation (drop queued + cancel in-flight)
- runner orchestration and question handling

It intentionally depends only on ports, not concrete XMPP/DB/runners.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from src.attachments import Attachment
from src.runners import Question, Runner
from src.runners.opencode.config import OpenCodeConfig

from src.core.session_runtime.api import (
    EventSinkPort,
    OutboundMessage,
    ProcessingChanged,
    RalphConfig,
    RalphStatus,
    SessionEvent,
)
from src.core.session_runtime.ports import (
    AttachmentPromptPort,
    HistoryPort,
    MessageStorePort,
    RalphLoopStorePort,
    RunnerFactoryPort,
    SessionState,
    SessionStorePort,
)


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


class SessionRuntime:
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

        self._generation = 0
        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue()
        self._task: asyncio.Task | None = None

        self.processing = False
        self.shutting_down = False

        self.runner: Runner | None = None
        self._run_task: asyncio.Task | None = None

        self._pending_question_answers: dict[str, asyncio.Future] = {}

        self._ralph_status: RalphStatus | None = None
        self._ralph_stop_requested = False
        self._ralph_wake = asyncio.Event()
        self._ralph_injected_prompt: str | None = None

        # Per-engine cumulative usage for the lifetime of this Switch session.
        # This reflects total tokens/cost used, regardless of remote session resets.
        self._usage_tokens_total: dict[str, int] = {"opencode": 0, "claude": 0}
        self._usage_cost_total: dict[str, float] = {"opencode": 0.0, "claude": 0.0}
        self._last_remote_session_id: dict[str, str | None] = {"opencode": None, "claude": None}

        # Throttle last_active writes. SQLite commits can become a bottleneck under
        # high message throughput (and directory browsing is read-heavy).
        self._last_active_written_at: float = 0.0
        self._last_active_min_interval_s: float = 10.0

    def _remember_remote_session_id(self, engine: str, session_id: str | None) -> None:
        if not engine or not session_id:
            return
        self._last_remote_session_id[engine] = session_id

    def _extract_run_tokens(self, engine: str, stats: dict) -> int:
        """Best-effort: normalize a per-run token count across engines."""
        if engine == "claude":
            t = stats.get("tokens_total")
            return int(t) if isinstance(t, (int, float)) else 0

        # OpenCode emits several categories; treat them as additive.
        total = 0
        for k in (
            "tokens_in",
            "tokens_out",
            "tokens_reasoning",
            "tokens_cache_read",
            "tokens_cache_write",
        ):
            v = stats.get(k)
            if isinstance(v, (int, float)):
                total += int(v)
        return total

    def _update_usage_totals(self, engine: str, stats: dict) -> None:
        tokens = self._extract_run_tokens(engine, stats)
        if tokens > 0:
            self._usage_tokens_total[engine] = int(self._usage_tokens_total.get(engine, 0)) + tokens

        cost = stats.get("cost_usd")
        if isinstance(cost, (int, float)) and cost:
            self._usage_cost_total[engine] = float(self._usage_cost_total.get(engine, 0.0)) + float(cost)

    def _format_session_tokens_suffix(self, engine: str) -> str:
        total = int(self._usage_tokens_total.get(engine, 0) or 0)
        if total >= 1000:
            return f"sess {total/1000.0:.1f}k tok"
        return f"sess {total} tok"

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

        # Best-effort: unblock any waiting question futures.
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
        """Inject a user prompt into the running Ralph loop.

        The injected prompt will be used for the current iteration continuation,
        waking the loop immediately if it's in the wait period.
        Returns True if injection was accepted (Ralph is running).
        """
        if not self._ralph_status or self._ralph_status.status not in {
            "running",
        }:
            return False
        self._ralph_injected_prompt = prompt
        self._ralph_wake.set()
        return True

    async def start_ralph(self, cfg: RalphConfig, *, wait: bool = False) -> None:
        """Enqueue a Ralph loop.

        Ralph runs as a single queued work item so it shares cancellation and
        serialization behavior with normal messages.
        """
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
        loop.create_task(self._events.emit(event))

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

        session = self._sessions.get(self.session_name)
        if not session:
            await self._emit(OutboundMessage("Session not found in database."))
            return

        now = time.monotonic()
        if (now - self._last_active_written_at) >= self._last_active_min_interval_s:
            self._sessions.update_last_active(self.session_name)
            self._last_active_written_at = now

        body_for_history = self._prompt.augment_prompt(item.body, item.attachments)
        self._history.append_to_history(
            body_for_history, self.working_dir, session.claude_session_id
        )
        self._history.log_activity(item.body, session=self.session_name, source="xmpp")
        self._messages.add(
            self.session_name, "user", body_for_history, session.active_engine
        )

        if not item.trigger_response:
            return

        engine = (session.active_engine or "opencode").strip().lower()
        self._run_task = asyncio.create_task(
            self._run_engine(engine=engine, session=session, prompt=body_for_history)
        )
        try:
            await self._run_task
        finally:
            self._run_task = None

    def _ralph_save(self, status: str) -> None:
        if (
            not self._ralph_loops
            or not self._ralph_status
            or not self._ralph_status.loop_id
        ):
            return
        self._ralph_loops.update_progress(
            self._ralph_status.loop_id,
            self._ralph_status.current_iteration,
            self._ralph_status.total_cost,
            status=status,
        )

    async def _run_ralph(self, cfg: RalphConfig) -> None:
        if not self._ralph_status:
            self._ralph_status = RalphStatus(status="running")
        self._ralph_status.status = "running"

        session = self._sessions.get(self.session_name)
        if not session:
            await self._emit(OutboundMessage("Session not found in database."))
            self._ralph_status.status = "error"
            self._ralph_status.error = "Session not found"
            return

        # Persist loop record.
        if self._ralph_loops:
            try:
                loop_id = self._ralph_loops.create(
                    self.session_name,
                    cfg.prompt,
                    int(cfg.max_iterations or 0),
                    cfg.completion_promise,
                    float(cfg.wait_seconds or 0.0),
                )
                self._ralph_status.loop_id = loop_id
            except Exception:
                pass

        promise_str = (
            f'"{cfg.completion_promise}"' if cfg.completion_promise else "none"
        )
        wait_minutes = float(cfg.wait_seconds or 0.0) / 60.0
        max_str = (
            str(cfg.max_iterations) if (cfg.max_iterations or 0) > 0 else "unlimited"
        )
        await self._emit(
            OutboundMessage(
                "Ralph loop started\n"
                f"Max: {max_str} | Wait: {wait_minutes:.2f} min | Done when: {promise_str}\n"
                "Use /ralph-cancel to stop after current iteration (or /cancel to abort immediately)"
            )
        )

        try:
            while True:
                if self.shutting_down:
                    self._ralph_status.status = "cancelled"
                    self._ralph_save("cancelled")
                    return

                if self._ralph_stop_requested:
                    self._ralph_status.status = "cancelled"
                    await self._emit(
                        OutboundMessage(
                            f"Ralph cancelled at iteration {self._ralph_status.current_iteration}\n"
                            f"Total cost: ${self._ralph_status.total_cost:.3f}"
                        )
                    )
                    self._ralph_save("cancelled")
                    return

                if cfg.max_iterations and cfg.max_iterations > 0:
                    if self._ralph_status.current_iteration >= cfg.max_iterations:
                        self._ralph_status.status = "max_iterations"
                        await self._emit(
                            OutboundMessage(
                                f"Ralph complete: hit max ({cfg.max_iterations})\n"
                                f"Total cost: ${self._ralph_status.total_cost:.3f}"
                            )
                        )
                        self._ralph_save("max_iterations")
                        return

                self._ralph_status.current_iteration += 1
                self._ralph_save("running")

                # Re-read session state each iteration so model/engine changes apply.
                session = self._sessions.get(self.session_name)
                if not session:
                    self._ralph_status.status = "error"
                    self._ralph_status.error = "Session not found"
                    await self._emit(OutboundMessage("Session not found in database."))
                    self._ralph_save("error")
                    return

                result = await self._run_ralph_iteration(cfg, session)
                if result.error:
                    self._ralph_status.status = "error"
                    self._ralph_status.error = result.error
                    await self._emit(
                        OutboundMessage(
                            f"Ralph error at iteration {self._ralph_status.current_iteration}: {result.error}\n"
                            f"Stopping. Total cost: ${self._ralph_status.total_cost:.3f}"
                        )
                    )
                    self._ralph_save("error")
                    return

                self._ralph_status.total_cost += float(result.cost)
                iter_str = (
                    f"{self._ralph_status.current_iteration}/{cfg.max_iterations}"
                    if cfg.max_iterations and cfg.max_iterations > 0
                    else str(self._ralph_status.current_iteration)
                )
                await self._emit(
                    OutboundMessage(
                        f"[Ralph {iter_str} | {result.tool_count}tools ${result.cost:.3f}]\n\n{result.text}"
                    )
                )

                if (
                    cfg.completion_promise
                    and f"<promise>{cfg.completion_promise}</promise>" in result.text
                ):
                    self._ralph_status.status = "completed"
                    await self._emit(
                        OutboundMessage(
                            f"Ralph COMPLETE at iteration {self._ralph_status.current_iteration}\n"
                            f"Detected: <promise>{cfg.completion_promise}</promise>\n"
                            f"Total cost: ${self._ralph_status.total_cost:.3f}"
                        )
                    )
                    self._ralph_save("completed")
                    return

                self._ralph_save("running")
                if cfg.wait_seconds and cfg.wait_seconds > 0:
                    try:
                        await asyncio.wait_for(
                            self._ralph_wake.wait(), timeout=cfg.wait_seconds
                        )
                    except asyncio.TimeoutError:
                        pass
                    finally:
                        self._ralph_wake.clear()

                # Check for injected prompt (user message during Ralph).
                # This counts as a continuation of the current iteration, not a new one.
                if self._ralph_injected_prompt:
                    injected = self._ralph_injected_prompt
                    self._ralph_injected_prompt = None

                    # Re-read session state for the continuation.
                    session = self._sessions.get(self.session_name)
                    if not session:
                        self._ralph_status.status = "error"
                        self._ralph_status.error = "Session not found"
                        await self._emit(OutboundMessage("Session not found in database."))
                        self._ralph_save("error")
                        return

                    await self._emit(
                        OutboundMessage(f"[Ralph inject] {injected[:100]}...")
                    )

                    result = await self._run_ralph_iteration(
                        cfg, session, prompt_override=injected
                    )
                    if result.error:
                        self._ralph_status.status = "error"
                        self._ralph_status.error = result.error
                        await self._emit(
                            OutboundMessage(
                                f"Ralph error during inject: {result.error}\n"
                                f"Stopping. Total cost: ${self._ralph_status.total_cost:.3f}"
                            )
                        )
                        self._ralph_save("error")
                        return

                    self._ralph_status.total_cost += float(result.cost)
                    iter_str = (
                        f"{self._ralph_status.current_iteration}/{cfg.max_iterations}"
                        if cfg.max_iterations and cfg.max_iterations > 0
                        else str(self._ralph_status.current_iteration)
                    )
                    await self._emit(
                        OutboundMessage(
                            f"[Ralph {iter_str}+ | {result.tool_count}tools ${result.cost:.3f}]\n\n{result.text}"
                        )
                    )

                    # Check for completion promise in injected response too.
                    if (
                        cfg.completion_promise
                        and f"<promise>{cfg.completion_promise}</promise>" in result.text
                    ):
                        self._ralph_status.status = "completed"
                        await self._emit(
                            OutboundMessage(
                                f"Ralph COMPLETE at iteration {self._ralph_status.current_iteration}\n"
                                f"Detected: <promise>{cfg.completion_promise}</promise>\n"
                                f"Total cost: ${self._ralph_status.total_cost:.3f}"
                            )
                        )
                        self._ralph_save("completed")
                        return

                    # Apply wait after injection too.
                    self._ralph_save("running")
                    if cfg.wait_seconds and cfg.wait_seconds > 0:
                        try:
                            await asyncio.wait_for(
                                self._ralph_wake.wait(), timeout=cfg.wait_seconds
                            )
                        except asyncio.TimeoutError:
                            pass
                        finally:
                            self._ralph_wake.clear()
        finally:
            if self._ralph_status and self._ralph_status.status == "running":
                self._ralph_status.status = "finished"
                self._ralph_save("finished")

    @dataclass
    class _RalphIterationResult:
        text: str = ""
        cost: float = 0.0
        tool_count: int = 0
        error: str | None = None

    def _build_ralph_prompt(self, cfg: RalphConfig, iteration: int) -> str:
        # Ralph loop orchestration is out-of-band: do not inject loop metadata
        # into the model prompt. Users want the exact prompt they wrote.
        return cfg.prompt

    async def _run_ralph_iteration(
        self, cfg: RalphConfig, session: SessionState, *, prompt_override: str | None = None
    ) -> "SessionRuntime._RalphIterationResult":
        result = SessionRuntime._RalphIterationResult()
        engine = (
            (cfg.force_engine or session.active_engine or "opencode").strip().lower()
        )
        if engine not in {"claude", "opencode"}:
            engine = "opencode"

        tool_summaries: list[str] = []
        last_progress_at = 0

        if engine == "opencode":
            question_callback = self._create_question_callback()
            self.runner = self._runner_factory.create(
                "opencode",
                working_dir=self.working_dir,
                output_dir=self.output_dir,
                session_name=self.session_name,
                opencode_config=OpenCodeConfig(
                    model=session.model_id,
                    reasoning_mode=session.reasoning_mode,
                    agent=session.opencode_agent,
                    question_callback=question_callback,
                ),
            )
        else:
            self.runner = self._runner_factory.create(
                "claude",
                working_dir=self.working_dir,
                output_dir=self.output_dir,
                session_name=self.session_name,
            )

        prompt = prompt_override or self._build_ralph_prompt(
            cfg, self._ralph_status.current_iteration if self._ralph_status else 1
        )
        try:
            # In prompt-only mode, each iteration should be isolated: do not
            # resume any remote session, and do not overwrite the session's
            # stored resume IDs.
            session_id = None
            if not cfg.prompt_only:
                session_id = (
                    session.opencode_session_id
                    if engine == "opencode"
                    else session.claude_session_id
                )
            accumulated = ""
            async for event_type, content in self.runner.run(prompt, session_id):
                if event_type == "session_id" and isinstance(content, str) and content:
                    if not cfg.prompt_only:
                        if engine == "opencode":
                            self._sessions.update_opencode_session_id(
                                self.session_name, content
                            )
                        else:
                            self._sessions.update_claude_session_id(
                                self.session_name, content
                            )
                elif event_type == "text" and isinstance(content, str):
                    if engine == "opencode":
                        accumulated += content
                        result.text = accumulated
                    else:
                        result.text = content
                elif event_type == "tool" and isinstance(content, str):
                    result.tool_count += 1
                    tool_summaries.append(content)

                    is_bash = content.startswith("[tool:bash")
                    if is_bash or len(tool_summaries) == 1:
                        last_progress_at = len(tool_summaries)
                        await self._emit(
                            OutboundMessage(
                                f"... {content}",
                                meta_type="tool",
                                meta_tool=self._infer_meta_tool_from_summary(content),
                            )
                        )
                    elif len(tool_summaries) - last_progress_at >= 8:
                        last_progress_at = len(tool_summaries)
                        await self._emit(
                            OutboundMessage(
                                f"... {' '.join(tool_summaries[-3:])}",
                                meta_type="tool",
                                meta_tool=self._infer_meta_tool_from_summary(
                                    tool_summaries[-1]
                                ),
                            )
                        )
                elif event_type == "result" and isinstance(content, dict):
                    cost = content.get("cost_usd")
                    if isinstance(cost, (int, float)):
                        result.cost = float(cost)
                elif event_type == "error":
                    result.error = str(content)
                elif event_type == "cancelled":
                    result.error = "cancelled"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        finally:
            self.runner = None
        return result

    async def _run_engine(
        self, *, engine: str, session: SessionState, prompt: str
    ) -> None:
        if engine == "claude":
            await self._run_claude(session, prompt)
            return
        if engine == "opencode":
            await self._run_opencode(session, prompt)
            return
        await self._emit(OutboundMessage(f"Unknown engine '{engine}'."))

    async def _run_claude(self, session: SessionState, prompt: str) -> None:
        self.runner = self._runner_factory.create(
            "claude",
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            session_name=self.session_name,
        )

        response_parts: list[str] = []
        tool_summaries: list[str] = []
        last_progress_at = 0

        async for event_type, content in self.runner.run(
            prompt, session.claude_session_id
        ):
            if self.shutting_down:
                return

            if event_type == "session_id" and isinstance(content, str) and content:
                self._sessions.update_claude_session_id(self.session_name, content)
            elif event_type == "text" and isinstance(content, str):
                response_parts = [content]
            elif event_type == "tool" and isinstance(content, str):
                tool_summaries.append(content)
                if len(tool_summaries) - last_progress_at >= 8:
                    last_progress_at = len(tool_summaries)
                    await self._emit(
                        OutboundMessage(
                            f"... {' '.join(tool_summaries[-3:])}",
                            meta_type="tool",
                            meta_tool=self._infer_meta_tool_from_summary(content),
                        )
                    )
            elif event_type == "result":
                await self._send_result(
                    tool_summaries, response_parts, content, engine="claude"
                )
            elif event_type == "error":
                await self._emit(OutboundMessage(f"Error: {content}"))
            elif event_type == "cancelled":
                await self._emit(OutboundMessage("Cancelled."))

    async def _run_opencode(self, session: SessionState, prompt: str) -> None:
        question_callback = self._create_question_callback()
        self.runner = self._runner_factory.create(
            "opencode",
            working_dir=self.working_dir,
            output_dir=self.output_dir,
            session_name=self.session_name,
            opencode_config=OpenCodeConfig(
                model=session.model_id,
                reasoning_mode=session.reasoning_mode,
                agent=session.opencode_agent,
                question_callback=question_callback,
            ),
        )

        response_parts: list[str] = []
        tool_summaries: list[str] = []
        accumulated = ""
        last_progress_at = 0

        async for event_type, content in self.runner.run(
            prompt, session.opencode_session_id
        ):
            if self.shutting_down:
                return

            if event_type == "session_id" and isinstance(content, str) and content:
                self._sessions.update_opencode_session_id(self.session_name, content)
            elif event_type == "text" and isinstance(content, str):
                accumulated += content
                response_parts = [accumulated]
            elif event_type == "tool" and isinstance(content, str):
                tool_summaries.append(content)
                is_bash = content.startswith("[tool:bash")
                if is_bash or len(tool_summaries) == 1:
                    last_progress_at = len(tool_summaries)
                    await self._emit(
                        OutboundMessage(
                            f"... {content}",
                            meta_type="tool",
                            meta_tool=self._infer_meta_tool_from_summary(content),
                        )
                    )
                elif len(tool_summaries) - last_progress_at >= 8:
                    last_progress_at = len(tool_summaries)
                    await self._emit(
                        OutboundMessage(
                            f"... {' '.join(tool_summaries[-3:])}",
                            meta_type="tool",
                            meta_tool=self._infer_meta_tool_from_summary(
                                tool_summaries[-1]
                            ),
                        )
                    )
            elif event_type == "question" and isinstance(content, Question):
                # The runner handles question flow via the callback.
                pass
            elif event_type == "result":
                await self._send_result(
                    tool_summaries, response_parts, content, engine="opencode"
                )
            elif event_type == "error":
                await self._emit(OutboundMessage(f"Error: {content}"))
            elif event_type == "cancelled":
                await self._emit(OutboundMessage("Cancelled."))

    async def _send_result(
        self,
        tool_summaries: list[str],
        response_parts: list[str],
        stats: object,
        *,
        engine: str,
    ) -> None:
        parts: list[str] = []
        if tool_summaries:
            tools = " ".join(tool_summaries[:5])
            if len(tool_summaries) > 5:
                tools += f" +{len(tool_summaries) - 5}"
            parts.append(tools)
        if response_parts:
            parts.append(response_parts[-1])

        meta_type = None
        meta_attrs: dict[str, str] | None = None
        if isinstance(stats, dict):
            # Keep stats meta small: the message body already contains the text.
            # Large attrs can cause transport/UI issues.
            allowed = {
                "engine",
                "model",
                "session_id",
                "turns",
                "tool_count",
                "tokens_in",
                "tokens_out",
                "tokens_reasoning",
                "tokens_cache_read",
                "tokens_cache_write",
                "tokens_total",
                "context_window",
                "cost_usd",
                "duration_s",
                "summary",
            }

            # Remember remote session ID (informational only; totals are lifetime).
            sid = stats.get("session_id") if isinstance(stats.get("session_id"), str) else None
            self._remember_remote_session_id(engine, sid)
            self._update_usage_totals(engine, stats)

            # Expose cumulative totals to the UI.
            stats = dict(stats)
            stats["session_tokens_total"] = int(self._usage_tokens_total.get(engine, 0) or 0)
            stats["session_cost_total"] = float(self._usage_cost_total.get(engine, 0.0) or 0.0)

            summary = stats.get("summary")
            if isinstance(summary, str) and summary:
                # Append a stable running token total.
                stats["summary"] = summary.rstrip() + " | " + self._format_session_tokens_suffix(engine)

            meta_type = "run-stats"
            meta_attrs = {
                str(k): str(v)
                for k, v in stats.items()
                if (k in allowed or k.startswith("session_")) and v is not None
            }

        await self._emit(
            OutboundMessage(
                "\n\n".join([p for p in parts if p]),
                meta_type=meta_type,
                meta_attrs=meta_attrs,
            )
        )
        self._messages.add(
            self.session_name,
            "assistant",
            response_parts[-1] if response_parts else "",
            engine,
        )

    def _create_question_callback(
        self,
    ) -> Callable[[Question], Awaitable[list[list[str]]]]:
        async def question_callback(question: Question) -> list[list[str]]:
            question_text = self._format_question(question)
            await self._emit(
                OutboundMessage(
                    question_text,
                    meta_type="question",
                    meta_tool="question",
                    meta_attrs={
                        "version": "1",
                        "engine": "opencode",
                        "request_id": question.request_id,
                        "question_count": str(len(question.questions or [])),
                    },
                    meta_payload={
                        "version": 1,
                        "engine": "opencode",
                        "request_id": question.request_id,
                        "questions": question.questions,
                    },
                )
            )

            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending_question_answers[question.request_id] = fut
            try:
                answer = await asyncio.wait_for(fut, timeout=300)
                return self._parse_question_answer(question, answer)
            except asyncio.TimeoutError:
                await self._emit(
                    OutboundMessage("[Question timed out - proceeding without answer]")
                )
                raise
            finally:
                self._pending_question_answers.pop(question.request_id, None)

        return question_callback

    def _parse_question_answer(
        self, question: Question, answer: object
    ) -> list[list[str]]:
        if isinstance(answer, list):
            return answer  # type: ignore[return-value]

        text = str(answer or "").strip()
        qs = question.questions or []
        if not qs:
            return []

        segments: list[str] = []
        if "\n" in text:
            segments = [s.strip() for s in text.splitlines() if s.strip()]
        if not segments and ";" in text and len(qs) > 1:
            segments = [s.strip() for s in text.split(";") if s.strip()]
        if not segments:
            segments = [text]

        answers: list[list[str]] = []
        for idx, q in enumerate(qs):
            seg = (
                segments[idx]
                if idx < len(segments)
                else (segments[0] if segments else "")
            )
            options = q.get("options") if isinstance(q, dict) else None
            if not isinstance(options, list) or not options:
                answers.append([seg] if seg else [])
                continue

            labels: list[str] = []
            for opt in options:
                if isinstance(opt, dict):
                    lab = str(opt.get("label", "") or "").strip()
                    if lab:
                        labels.append(lab)

            chosen: list[str] = []
            seg_norm = seg.strip().lower()
            direct = next((lab for lab in labels if lab.lower() == seg_norm), None)
            if direct:
                answers.append([direct])
                continue

            for tok in re.split(r"[\s,]+", seg.strip()):
                if not tok:
                    continue
                if tok.isdigit():
                    n = int(tok)
                    if 1 <= n <= len(labels):
                        chosen.append(labels[n - 1])
                    continue
                match = next(
                    (lab for lab in labels if lab.lower() == tok.lower()), None
                )
                if match:
                    chosen.append(match)

            seen: set[str] = set()
            chosen = [x for x in chosen if not (x in seen or seen.add(x))]
            answers.append(chosen)

        return answers

    def _format_question(self, question: Question) -> str:
        parts: list[str] = []
        parts.append("[Question]")
        for q_idx, q in enumerate(question.questions or [], 1):
            if not isinstance(q, dict):
                continue
            header = str(q.get("header", "") or "").strip()
            text = str(q.get("question", "") or "").strip()
            options = q.get("options", [])

            if header:
                parts.append(f"{q_idx}) {header}")
            elif len(question.questions or []) > 1:
                parts.append(f"{q_idx})")
            if text:
                parts.append(text)

            if isinstance(options, list) and options:
                parts.append("Options:")
                for i, opt in enumerate(options, 1):
                    if not isinstance(opt, dict):
                        continue
                    label = str(
                        opt.get("label", f"Option {i}") or f"Option {i}"
                    ).strip()
                    desc = str(opt.get("description", "") or "").strip()
                    parts.append(f"  {i}) {label}" + (f" - {desc}" if desc else ""))

        parts.append("Reply with option number(s) (e.g., '1' or '1,2') or label text.")
        return "\n".join([p for p in parts if p])
