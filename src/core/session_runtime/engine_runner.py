"""Engine dispatch, runner loop orchestration, and result formatting."""

from __future__ import annotations

import os
from typing import Awaitable, Callable

from src.engines import remote_session_id_for, runtime_engine_names
from src.runners.claude.config import ClaudeConfig
from src.runners.cursor.config import CursorConfig
from src.runners.opencode.config import OpenCodeConfig
from src.runners.pi.config import PiConfig
from src.runners.ports import ToolResult

from src.core.session_runtime.api import OutboundMessage
from src.core.session_runtime.ports import SessionState
from src.core.session_runtime.runner_loop import (
    RunnerEventHandlers,
    RunnerEventLoopState,
    build_engine_handlers,
    cleanup_runner,
    run_runner_event_loop,
)


class EngineRunnerMixin:
    def _remember_remote_session_id(self, engine: str, session_id: str | None) -> None:
        if not engine or not session_id:
            return
        self._last_remote_session_id[engine] = session_id

    @staticmethod
    def _as_non_negative_float(value: object) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        v = float(value)
        if v < 0:
            return None
        return v

    @staticmethod
    def _as_non_negative_int(value: object) -> int | None:
        if not isinstance(value, (int, float)):
            return None
        v = int(value)
        if v < 0:
            return None
        return v

    @staticmethod
    def _safe_tps(tokens: int | float | None, duration_s: float | None) -> float | None:
        if duration_s is None or duration_s <= 0:
            return None
        if not isinstance(tokens, (int, float)):
            return None
        t = float(tokens)
        if t <= 0:
            return None
        return t / duration_s

    def _augment_tps_stats(self, engine: str, stats: dict[str, object]) -> None:
        """Attach normalized throughput fields to run stats.

        Units are always tokens per second (tok/s) over wall-clock duration.
        """
        duration_s = self._as_non_negative_float(stats.get("duration_s"))
        if duration_s is None or duration_s <= 0:
            return
        generation_duration_s = self._as_non_negative_float(
            stats.get("generation_duration_s")
        )
        if generation_duration_s is None or generation_duration_s <= 0:
            generation_duration_s = duration_s

        output_tokens = self._as_non_negative_int(stats.get("tokens_out")) or 0
        reasoning_tokens = self._as_non_negative_int(stats.get("tokens_reasoning")) or 0
        input_tokens = self._as_non_negative_int(stats.get("tokens_in")) or 0
        cache_read_tokens = (
            self._as_non_negative_int(stats.get("tokens_cache_read")) or 0
        )
        cache_write_tokens = (
            self._as_non_negative_int(stats.get("tokens_cache_write")) or 0
        )

        total_tokens = stats.get("tokens_total")
        total_tokens_i = (
            int(total_tokens) if isinstance(total_tokens, (int, float)) else None
        )

        generated_tokens = output_tokens + reasoning_tokens

        processed_tokens = (
            input_tokens
            + output_tokens
            + reasoning_tokens
            + cache_read_tokens
            + cache_write_tokens
        )

        if engine == "claude" and total_tokens_i is not None:
            processed_tokens = total_tokens_i

        tps_output = self._safe_tps(output_tokens, generation_duration_s)
        tps_generated = self._safe_tps(generated_tokens, generation_duration_s)
        tps_processed = self._safe_tps(processed_tokens, duration_s)
        tps_total = self._safe_tps(total_tokens_i, duration_s)

        tps = None
        tps_basis = None
        for basis, value in (
            ("generated", tps_generated),
            ("output", tps_output),
            ("total", tps_total),
            ("processed", tps_processed),
        ):
            if value is not None:
                tps = value
                tps_basis = basis
                break

        if generated_tokens > 0:
            stats["tokens_generated"] = generated_tokens
        if processed_tokens > 0:
            stats["tokens_processed"] = processed_tokens

        if tps_output is not None:
            stats["tps_output"] = tps_output
        if tps_generated is not None:
            stats["tps_generated"] = tps_generated
        if tps_processed is not None:
            stats["tps_processed"] = tps_processed
        if tps_total is not None:
            stats["tps_total"] = tps_total

        if tps is not None and tps_basis is not None:
            stats["tps"] = tps
            stats["tps_basis"] = tps_basis
            stats["tps_unit"] = "tok/s"

    def _extract_run_tokens(self, engine: str, stats: dict) -> int:
        """Best-effort: normalize a per-run token count across engines."""
        if engine == "claude":
            t = stats.get("tokens_total")
            return int(t) if isinstance(t, (int, float)) else 0

        if engine == "opencode":
            t = stats.get("tokens_total")
            if isinstance(t, (int, float)):
                return int(t)

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
        # Some runners (notably Pi) can provide authoritative persisted-session
        # totals. Prefer those over summing local process-lifetime counters.
        session_tokens = stats.get("session_tokens_total")
        if isinstance(session_tokens, (int, float)) and session_tokens >= 0:
            self._usage_tokens_total[engine] = int(session_tokens)
        else:
            tokens = self._extract_run_tokens(engine, stats)
            if tokens > 0:
                self._usage_tokens_total[engine] = (
                    int(self._usage_tokens_total.get(engine, 0)) + tokens
                )

        session_cost = stats.get("session_cost_total")
        if isinstance(session_cost, (int, float)) and session_cost >= 0:
            self._usage_cost_total[engine] = float(session_cost)
        else:
            cost = stats.get("cost_usd")
            if isinstance(cost, (int, float)) and cost:
                self._usage_cost_total[engine] = float(
                    self._usage_cost_total.get(engine, 0.0)
                ) + float(cost)

    def _format_session_tokens_suffix(self, engine: str) -> str:
        total = int(self._usage_tokens_total.get(engine, 0) or 0)
        if total >= 1000:
            return f"sess {total / 1000.0:.1f}k tok"
        return f"sess {total} tok"

    def _create_runner_for_engine(
        self,
        engine: str,
        session: SessionState,
        *,
        pi_config: PiConfig | None = None,
    ) -> None:
        """Set self.runner for the given engine."""
        if engine == "claude":
            self.runner = self._runner_factory.create(
                "claude",
                working_dir=self.working_dir,
                output_dir=self.output_dir,
                session_name=self.session_name,
                claude_config=ClaudeConfig(model=session.model_id or None),
            )
        elif engine == "opencode":
            self.runner = self._runner_factory.create(
                "opencode",
                working_dir=self.working_dir,
                output_dir=self.output_dir,
                session_name=self.session_name,
                opencode_config=OpenCodeConfig(
                    model=session.model_id or None,
                    reasoning_mode=session.reasoning_mode,
                    agent=session.opencode_agent or "bridge",
                    question_callback=self._create_question_callback(engine="opencode"),
                ),
            )
        elif engine == "pi":
            from src.runners.pi.config import resolve_thinking_level

            pi_thinking = resolve_thinking_level(session.reasoning_mode)
            self.runner = self._runner_factory.create(
                "pi",
                working_dir=self.working_dir,
                output_dir=self.output_dir,
                session_name=self.session_name,
                pi_config=pi_config or PiConfig(
                    model=session.model_id or None,
                    thinking=pi_thinking,
                    system_prompt_extra=session.system_prompt_extra,
                ),
            )
        elif engine == "cursor":
            self.runner = self._runner_factory.create(
                "cursor",
                working_dir=self.working_dir,
                output_dir=self.output_dir,
                session_name=self.session_name,
                cursor_config=CursorConfig(model=session.model_id or "composer-2.5"),
            )
        elif engine == "vllm-direct":
            self.runner = self._runner_factory.create(
                "vllm-direct",
                working_dir=self.working_dir,
                output_dir=self.output_dir,
                session_name=self.session_name,
                pi_config=pi_config or PiConfig(model=session.model_id or None),
            )
        else:
            raise ValueError(f"Unknown engine: {engine}")

    @staticmethod
    def _session_id_for_engine(engine: str, session: SessionState) -> str | None:
        return remote_session_id_for(engine, session)

    async def _save_session_id(self, engine: str, session_id: str) -> None:
        await self._sessions.update_remote_session_id(
            self.session_name, engine, session_id
        )

    async def _run_engine(
        self, *, engine: str, session: SessionState, prompt: str
    ) -> None:
        if engine not in runtime_engine_names():
            await self._emit(OutboundMessage(f"Unknown engine '{engine}'."))
            return
        await self._run_engine_generic(engine, session, prompt)

    async def _run_engine_generic(
        self,
        engine: str,
        session: SessionState,
        prompt: str,
        *,
        skip_runner_create: bool = False,
        result_engine: str | None = None,
        ephemeral: bool = False,
    ) -> None:
        """Unified event loop for claude, opencode, and pi engines.

        If ephemeral=True, don't save session_id (used by /handoff to avoid
        overwriting the target engine's session state).
        """
        if not skip_runner_create:
            self._create_runner_for_engine(engine, session)
        label = result_engine or engine
        session_id = None if ephemeral else self._session_id_for_engine(engine, session)

        async def send_result(content: object, state: RunnerEventLoopState) -> None:
            await self._send_result(
                state.tool_summaries, state.response_parts, content, engine=label
            )

        handlers = build_engine_handlers(
            save_session_id=(
                self._save_session_id_handler(engine) if not ephemeral else None
            ),
            handle_tool=self._handle_runner_tool,
            emit_tool_result=self._emit_runner_tool_result,
            send_result=send_result,
            emit_text=self._emit_text,
            should_abort=lambda: self.shutting_down,
        )
        await self._run_runner_loop(
            engine=engine,
            prompt=prompt,
            session_id=session_id,
            handlers=handlers,
        )

    def _save_session_id_handler(
        self, engine: str
    ) -> Callable[[str], Awaitable[None]]:
        async def save(session_id: str) -> None:
            await self._save_session_id(engine, session_id)

        return save

    async def _handle_runner_tool(
        self, content: str, tool_summaries: list[str], last_progress_at: int
    ) -> int:
        await self._emit_tool_progress(content, tool_summaries, last_progress_at)
        return self._updated_progress_at(tool_summaries, last_progress_at, content)

    async def _run_runner_loop(
        self,
        *,
        engine: str,
        prompt: str,
        session_id: str | None,
        handlers: RunnerEventHandlers,
        accumulate_text: bool | None = None,
        catch_errors: bool = False,
    ) -> RunnerEventLoopState:
        runner = self.runner
        if runner is None:
            raise RuntimeError(f"Runner not initialized for engine '{engine}'")
        if accumulate_text is None:
            accumulate_text = engine != "claude"
        try:
            return await run_runner_event_loop(
                runner,
                prompt,
                session_id,
                accumulate_text=accumulate_text,
                handlers=handlers,
                catch_errors=catch_errors,
            )
        finally:
            runner_ref = self.runner
            self.runner = None
            await cleanup_runner(runner_ref)

    async def _emit_text(self, text: str) -> None:
        await self._emit(OutboundMessage(text))

    async def _emit_runner_tool_result(self, result: ToolResult) -> None:
        payload = None
        if result.full_text:
            try:
                detail_max = int(
                    os.getenv("SWITCH_TOOL_RESULT_DETAIL_MAX_LEN", "20000") or "20000"
                )
            except ValueError:
                detail_max = 20000
            detail_max = max(1000, min(detail_max, 100000))
            full_text = f"... {result.full_text}"
            if len(full_text) > detail_max:
                suffix = "\n... [full output capped]"
                full_text = full_text[: detail_max - len(suffix)] + suffix
            payload = {"full_text": full_text}

        await self._emit(
            OutboundMessage(
                f"... {result.summary}",
                meta_type="tool-result",
                meta_tool=self._infer_meta_tool_from_summary(result.summary),
                meta_payload=payload,
            )
        )

    async def _emit_tool_progress(
        self, content: str, tool_summaries: list[str], last_progress_at: int
    ) -> None:
        progress_every = max(
            1,
            int(os.getenv("SWITCH_TOOL_PROGRESS_EVERY", "8") or "8"),
        )
        verbose_bash = os.getenv(
            "SWITCH_TOOL_PROGRESS_BASH_VERBOSE", "0"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        is_bash = content.startswith("[tool:bash")
        if (is_bash and verbose_bash) or len(tool_summaries) == 1:
            await self._emit(
                OutboundMessage(
                    f"... {content}",
                    meta_type="tool",
                    meta_tool=self._infer_meta_tool_from_summary(content),
                )
            )
        elif len(tool_summaries) - last_progress_at >= progress_every:
            await self._emit(
                OutboundMessage(
                    f"... {' '.join(tool_summaries[-3:])}",
                    meta_type="tool",
                    meta_tool=self._infer_meta_tool_from_summary(tool_summaries[-1]),
                )
            )

    @staticmethod
    def _updated_progress_at(
        tool_summaries: list[str], last_progress_at: int, content: str
    ) -> int:
        progress_every = max(
            1,
            int(os.getenv("SWITCH_TOOL_PROGRESS_EVERY", "8") or "8"),
        )
        verbose_bash = os.getenv(
            "SWITCH_TOOL_PROGRESS_BASH_VERBOSE", "0"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        is_bash = content.startswith("[tool:bash")
        if (is_bash and verbose_bash) or len(tool_summaries) == 1:
            return len(tool_summaries)
        if len(tool_summaries) - last_progress_at >= progress_every:
            return len(tool_summaries)
        return last_progress_at

    async def _send_result(
        self,
        tool_summaries: list[str],
        response_parts: list[str],
        stats: object,
        *,
        engine: str,
    ) -> None:
        final_text = response_parts[-1] if response_parts else ""

        if not final_text and isinstance(stats, dict):
            maybe_text = stats.get("text")
            if isinstance(maybe_text, str):
                final_text = maybe_text

        # Tool calls are already streamed as separate progress messages;
        # don't repeat them at the top of the final reply.
        parts: list[str] = []
        if final_text:
            parts.append(final_text)

        meta_type = None
        meta_attrs: dict[str, str] | None = None
        if isinstance(stats, dict):
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
                "tokens_generated",
                "tokens_processed",
                "context_tokens",
                "context_window",
                "context_percent",
                "cost_usd",
                "duration_s",
                "generation_duration_s",
                "tps",
                "tps_output",
                "tps_generated",
                "tps_processed",
                "tps_total",
                "tps_basis",
                "tps_unit",
                "summary",
            }

            sid = (
                stats.get("session_id")
                if isinstance(stats.get("session_id"), str)
                else None
            )
            self._remember_remote_session_id(engine, sid)
            self._update_usage_totals(engine, stats)

            stats = dict(stats)
            self._augment_tps_stats(engine, stats)
            stats["session_tokens_total"] = int(
                self._usage_tokens_total.get(engine, 0) or 0
            )
            stats["session_cost_total"] = float(
                self._usage_cost_total.get(engine, 0.0) or 0.0
            )

            summary = stats.get("summary")
            if isinstance(summary, str) and summary:
                tps = stats.get("tps")
                basis = stats.get("tps_basis")
                if isinstance(tps, (int, float)) and isinstance(basis, str):
                    summary = summary.rstrip() + f" | {float(tps):.1f} tok/s ({basis})"

                stats["summary"] = (
                    summary.rstrip()
                    + " | "
                    + self._format_session_tokens_suffix(engine)
                )

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
        await self._messages.add(
            self.session_name,
            "assistant",
            final_text,
            engine,
        )
