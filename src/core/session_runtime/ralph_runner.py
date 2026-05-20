"""Ralph loop orchestration."""

from __future__ import annotations

import asyncio
import logging

from src.engines import ralph_engine_names
from src.core.session_runtime.api import OutboundMessage, RalphConfig, RalphStatus
from src.core.session_runtime.ports import SessionState
from src.core.session_runtime.runner_loop import (
    RalphIterationResult,
    build_ralph_handlers,
)

log = logging.getLogger("session_runtime.ralph")


class RalphRunnerMixin:
    async def _ralph_save(self, status: str) -> None:
        if (
            not self._ralph_loops
            or not self._ralph_status
            or not self._ralph_status.loop_id
        ):
            return
        await self._ralph_loops.update_progress(
            self._ralph_status.loop_id,
            self._ralph_status.current_iteration,
            self._ralph_status.total_cost,
            status=status,
        )

    async def _run_ralph(self, cfg: RalphConfig) -> None:
        if not await self._ralph_begin(cfg):
            return

        try:
            while True:
                if await self._ralph_should_stop(cfg):
                    return

                self._ralph_status.current_iteration += 1
                await self._ralph_save("running")

                if not await self._ralph_run_turn(cfg):
                    return

                await self._ralph_wait(cfg)

                injected = self._ralph_take_injected_prompt()
                if injected is not None:
                    if not await self._ralph_run_turn(
                        cfg, prompt_override=injected, label_suffix="+"
                    ):
                        return
                    await self._ralph_wait(cfg)
        finally:
            await self._ralph_finish_if_running()

    async def _ralph_begin(self, cfg: RalphConfig) -> bool:
        if not self._ralph_status:
            self._ralph_status = RalphStatus(status="running")
        self._ralph_status.status = "running"

        session = self._sessions.get(self.session_name)
        if not session:
            await self._emit(OutboundMessage("Session not found in database."))
            self._ralph_status.status = "error"
            self._ralph_status.error = "Session not found"
            return False

        if self._ralph_loops:
            try:
                loop_id = await self._ralph_loops.create(
                    self.session_name,
                    cfg.prompt,
                    int(cfg.max_iterations or 0),
                    cfg.completion_promise,
                    float(cfg.wait_seconds or 0.0),
                )
                self._ralph_status.loop_id = loop_id
            except Exception:
                log.warning(
                    "Failed to persist Ralph loop for %s",
                    self.session_name,
                    exc_info=True,
                )

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
        return True

    async def _ralph_should_stop(self, cfg: RalphConfig) -> bool:
        if self.shutting_down:
            self._ralph_status.status = "cancelled"
            await self._ralph_save("cancelled")
            return True

        if self._ralph_stop_requested:
            self._ralph_status.status = "cancelled"
            await self._emit(
                OutboundMessage(
                    f"Ralph cancelled at iteration {self._ralph_status.current_iteration}\n"
                    f"Total cost: ${self._ralph_status.total_cost:.3f}"
                )
            )
            await self._ralph_save("cancelled")
            return True

        if cfg.max_iterations and cfg.max_iterations > 0:
            if self._ralph_status.current_iteration >= cfg.max_iterations:
                self._ralph_status.status = "max_iterations"
                await self._emit(
                    OutboundMessage(
                        f"Ralph complete: hit max ({cfg.max_iterations})\n"
                        f"Total cost: ${self._ralph_status.total_cost:.3f}"
                    )
                )
                await self._ralph_save("max_iterations")
                return True

        return False

    async def _ralph_run_turn(
        self,
        cfg: RalphConfig,
        *,
        prompt_override: str | None = None,
        label_suffix: str = "",
    ) -> bool:
        """Run one Ralph turn. Returns True to continue the outer loop."""
        if prompt_override:
            await self._emit(
                OutboundMessage(f"[Ralph inject] {prompt_override[:100]}...")
            )

        session = self._sessions.get(self.session_name)
        if not session:
            await self._ralph_fail(
                "Session not found",
                "Session not found in database.",
            )
            return False

        result = await self._run_ralph_iteration(
            cfg, session, prompt_override=prompt_override
        )
        if result.error:
            when = (
                "during inject"
                if prompt_override
                else f"at iteration {self._ralph_status.current_iteration}"
            )
            await self._ralph_fail(
                result.error,
                f"Ralph error {when}: {result.error}\n"
                f"Stopping. Total cost: ${self._ralph_status.total_cost:.3f}",
            )
            return False

        self._ralph_status.total_cost += float(result.cost)
        iter_label = self._ralph_iter_label(cfg, suffix=label_suffix)
        await self._emit(
            OutboundMessage(
                f"[Ralph {iter_label} | {result.tool_count}tools ${result.cost:.3f}]\n\n{result.text}"
            )
        )

        if self._ralph_promise_met(cfg, result.text):
            await self._ralph_complete(cfg)
            return False

        await self._ralph_save("running")
        return True

    def _ralph_iter_label(self, cfg: RalphConfig, *, suffix: str = "") -> str:
        iteration = self._ralph_status.current_iteration
        if cfg.max_iterations and cfg.max_iterations > 0:
            return f"{iteration}/{cfg.max_iterations}{suffix}"
        return f"{iteration}{suffix}"

    def _ralph_promise_met(self, cfg: RalphConfig, text: str) -> bool:
        return bool(
            cfg.completion_promise
            and f"<promise>{cfg.completion_promise}</promise>" in text
        )

    async def _ralph_complete(self, cfg: RalphConfig) -> None:
        self._ralph_status.status = "completed"
        await self._emit(
            OutboundMessage(
                f"Ralph COMPLETE at iteration {self._ralph_status.current_iteration}\n"
                f"Detected: <promise>{cfg.completion_promise}</promise>\n"
                f"Total cost: ${self._ralph_status.total_cost:.3f}"
            )
        )
        await self._ralph_save("completed")

    async def _ralph_fail(self, error: str, message: str) -> None:
        self._ralph_status.status = "error"
        self._ralph_status.error = error
        await self._emit(OutboundMessage(message))
        await self._ralph_save("error")

    async def _ralph_wait(self, cfg: RalphConfig) -> None:
        if not cfg.wait_seconds or cfg.wait_seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._ralph_wake.wait(), timeout=cfg.wait_seconds)
        except asyncio.TimeoutError:
            pass
        finally:
            self._ralph_wake.clear()

    def _ralph_take_injected_prompt(self) -> str | None:
        injected = self._ralph_injected_prompt
        if not injected:
            return None
        self._ralph_injected_prompt = None
        return injected

    async def _ralph_finish_if_running(self) -> None:
        if self._ralph_status and self._ralph_status.status == "running":
            self._ralph_status.status = "finished"
            await self._ralph_save("finished")

    def _build_ralph_prompt(self, cfg: RalphConfig, iteration: int) -> str:
        return cfg.prompt

    async def _run_ralph_iteration(
        self,
        cfg: RalphConfig,
        session: SessionState,
        *,
        prompt_override: str | None = None,
    ) -> RalphIterationResult:
        result = RalphIterationResult()
        engine = (cfg.force_engine or session.active_engine or "pi").strip().lower()
        if engine not in ralph_engine_names():
            log.warning("Ralph: engine %r not supported, falling back to pi", engine)
            engine = "pi"

        self._create_runner_for_engine(engine, session)
        prompt = prompt_override or self._build_ralph_prompt(
            cfg, self._ralph_status.current_iteration if self._ralph_status else 1
        )
        session_id = (
            None
            if cfg.prompt_only
            else self._session_id_for_engine(engine, session)
        )
        handlers = build_ralph_handlers(
            result,
            save_session_id=(
                self._save_session_id_handler(engine) if not cfg.prompt_only else None
            ),
            handle_tool=self._handle_runner_tool,
            emit_tool_result=self._emit_runner_tool_result,
        )

        try:
            state = await self._run_runner_loop(
                engine=engine,
                prompt=prompt,
                session_id=session_id,
                handlers=handlers,
                catch_errors=True,
            )
            return RalphIterationResult.merge_loop_state(result, state)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            return result
