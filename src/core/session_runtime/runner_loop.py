"""Shared runner event loop for SessionRuntime."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from src.runners.ports import Runner

log = logging.getLogger("session_runtime.runner_loop")


def _never_abort() -> bool:
    return False


@dataclass
class RunnerEventLoopState:
    accumulated_text: str = ""
    response_parts: list[str] = field(default_factory=list)
    tool_summaries: list[str] = field(default_factory=list)
    last_progress_at: int = 0
    tool_count: int = 0
    error: str | None = None
    aborted: bool = False


@dataclass(frozen=True)
class RunnerEventHandlers:
    save_session_id: Callable[[str], Awaitable[None]] | None
    handle_tool: Callable[[str, list[str], int], Awaitable[int]]
    emit_tool_result: Callable[[str], Awaitable[None]]
    on_result: Callable[[object, RunnerEventLoopState], Awaitable[None]]
    on_error: Callable[[object], Awaitable[None]]
    on_cancelled: Callable[[], Awaitable[None]]
    should_abort: Callable[[], bool] = _never_abort
    count_tools: bool = False


@dataclass
class RalphIterationResult:
    text: str = ""
    cost: float = 0.0
    tool_count: int = 0
    error: str | None = None

    @classmethod
    def merge_loop_state(
        cls, result: RalphIterationResult, state: RunnerEventLoopState
    ) -> RalphIterationResult:
        if state.error and result.error is None:
            result.error = state.error
        result.text = state.response_parts[-1] if state.response_parts else ""
        result.tool_count = state.tool_count
        return result


def build_ralph_handlers(
    result: RalphIterationResult,
    *,
    save_session_id: Callable[[str], Awaitable[None]] | None,
    handle_tool: Callable[[str, list[str], int], Awaitable[int]],
    emit_tool_result: Callable[[str], Awaitable[None]],
) -> RunnerEventHandlers:
    async def on_result(content: object, _state: RunnerEventLoopState) -> None:
        if isinstance(content, dict):
            cost = content.get("cost_usd")
            if isinstance(cost, (int, float)):
                result.cost = float(cost)

    async def on_error(content: object) -> None:
        result.error = str(content)

    async def on_cancelled() -> None:
        result.error = "cancelled"

    return RunnerEventHandlers(
        save_session_id=save_session_id,
        handle_tool=handle_tool,
        emit_tool_result=emit_tool_result,
        on_result=on_result,
        on_error=on_error,
        on_cancelled=on_cancelled,
        count_tools=True,
    )


def build_engine_handlers(
    *,
    save_session_id: Callable[[str], Awaitable[None]] | None,
    handle_tool: Callable[[str, list[str], int], Awaitable[int]],
    emit_tool_result: Callable[[str], Awaitable[None]],
    send_result: Callable[[object, RunnerEventLoopState], Awaitable[None]],
    emit_text: Callable[[str], Awaitable[None]],
    should_abort: Callable[[], bool] = _never_abort,
) -> RunnerEventHandlers:
    async def on_error(content: object) -> None:
        await emit_text(f"Error: {content}")

    async def on_cancelled() -> None:
        await emit_text("Cancelled.")

    return RunnerEventHandlers(
        save_session_id=save_session_id,
        handle_tool=handle_tool,
        emit_tool_result=emit_tool_result,
        on_result=send_result,
        on_error=on_error,
        on_cancelled=on_cancelled,
        should_abort=should_abort,
    )


async def cleanup_runner(runner: Runner | None) -> None:
    if runner is None:
        return
    cleanup = getattr(runner, "cleanup", None)
    if callable(cleanup):
        try:
            await cleanup()
        except Exception:
            log.warning("Runner cleanup failed", exc_info=True)


async def run_runner_event_loop(
    runner: Runner,
    prompt: str,
    session_id: str | None,
    *,
    accumulate_text: bool,
    handlers: RunnerEventHandlers,
    catch_errors: bool = False,
) -> RunnerEventLoopState:
    state = RunnerEventLoopState()
    try:
        async for event_type, content in runner.run(prompt, session_id):
            if handlers.should_abort():
                state.aborted = True
                return state

            match (event_type, content):
                case ("session_id", str(sid)) if sid and handlers.save_session_id:
                    await handlers.save_session_id(sid)
                case ("text", str(text)):
                    if accumulate_text:
                        state.accumulated_text += text
                        state.response_parts = [state.accumulated_text]
                    else:
                        state.response_parts = [text]
                case ("tool", str(tool_content)):
                    if handlers.count_tools:
                        state.tool_count += 1
                    state.tool_summaries.append(tool_content)
                    state.last_progress_at = await handlers.handle_tool(
                        tool_content, state.tool_summaries, state.last_progress_at
                    )
                case ("tool_result", str(result_content)):
                    await handlers.emit_tool_result(result_content)
                case ("result", _):
                    await handlers.on_result(content, state)
                case ("error", _):
                    await handlers.on_error(content)
                case ("cancelled", _):
                    await handlers.on_cancelled()
                case _:
                    pass
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if catch_errors:
            state.error = f"{type(e).__name__}: {e}"
        else:
            raise
    return state
