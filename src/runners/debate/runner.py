"""Debate runner — two models plan, critique, then synthesize."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import aiohttp

from src.runners.base import BaseRunner
from src.runners.debate.config import DebateConfig
from src.runners.ports import RunnerEvent

log = logging.getLogger("debate")


@dataclass
class _StreamResult:
    """Tracks both thinking and content from a model response."""
    thinking: list[str] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def content_text(self) -> str:
        return "".join(self.content)

    @property
    def thinking_text(self) -> str:
        return "".join(self.thinking)

    @property
    def full_text(self) -> str:
        """Full display text (thinking + content)."""
        parts = []
        t = self.thinking_text
        if t:
            parts.append(t)
        c = self.content_text
        if c:
            parts.append(c)
        return "\n\n".join(parts)


class DebateRunner(BaseRunner):
    """Runs a two-model debate via OpenAI-compatible streaming APIs.

    Flow:
    1. Qwen generates a clarifying multiple-choice question for the user.
    2. User replies with choice(s).
    3. Round 1: Both models propose plans (Qwen streamed, GLM background).
    4. Round 2: Qwen synthesizes both plans into a pre-final plan.
    5. Round 3: GLM critiques the pre-final plan.
    6. Round 4: Qwen produces the final plan incorporating the critique.

    Thinking (reasoning_content) is shown to the user but only the content
    portion is passed to subsequent prompts, keeping context sizes manageable.
    """

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        config: DebateConfig | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)
        self._config = config or DebateConfig()
        self._cancelled = False

    def _make_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=None,
            sock_connect=30,
            sock_read=600,
        )

    async def _check_available(self, base_url: str) -> bool:
        """Quick health check: GET /v1/models, return True if 200."""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url}/v1/models") as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _resolve_models(
        self,
    ) -> tuple[str, str, str, str]:
        """Resolve primary/secondary model URLs and names based on availability.

        Returns (primary_url, primary_name, secondary_url, secondary_name).
        Primary = Qwen (model_a), Secondary = GLM (model_b).
        If one is down, both roles use the other.
        Raises RuntimeError if both are down.
        """
        a_url = self._config.resolve_model_a_url()
        a_name = self._config.resolve_model_a_name()
        b_url = self._config.resolve_model_b_url()
        b_name = self._config.resolve_model_b_name()

        a_ok, b_ok = await asyncio.gather(
            self._check_available(a_url),
            self._check_available(b_url),
        )

        if a_ok and b_ok:
            return a_url, a_name, b_url, b_name
        if a_ok and not b_ok:
            log.warning("GLM (%s) unavailable — Qwen handles all roles", b_url)
            return a_url, a_name, a_url, a_name
        if b_ok and not a_ok:
            log.warning("Qwen (%s) unavailable — GLM handles all roles", a_url)
            return b_url, b_name, b_url, b_name
        raise RuntimeError(
            f"Both models unavailable: {a_name} ({a_url}), {b_name} ({b_url})"
        )

    async def _chat_completion_stream_separated(
        self,
        base_url: str,
        messages: list[dict[str, str]],
        model_name: str,
    ) -> AsyncIterator[tuple[str, str]]:
        """Stream SSE, yielding (kind, text) where kind is 'thinking' or 'content'."""
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
        }

        async with aiohttp.ClientSession(timeout=self._make_timeout()) as session:
            async with session.post(
                url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.content:
                    if self._cancelled:
                        return

                    line = raw_line.decode(errors="replace").strip()
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        return

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices")
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield ("thinking", reasoning)
                    content = delta.get("content")
                    if content:
                        yield ("content", content)

    async def _safe_collect(
        self,
        base_url: str,
        messages: list[dict[str, str]],
        model_name: str,
    ) -> _StreamResult:
        """Collect full response with thinking/content separated."""
        result = _StreamResult()
        try:
            async for kind, text in self._chat_completion_stream_separated(
                base_url, messages, model_name
            ):
                if self._cancelled:
                    break
                if kind == "thinking":
                    result.thinking.append(text)
                else:
                    result.content.append(text)
        except Exception as e:
            log.warning("Model %s failed: %s", model_name, e)
            result.error = str(e)
        return result

    async def _safe_stream_to_user(
        self,
        base_url: str,
        messages: list[dict[str, str]],
        model_name: str,
    ) -> AsyncIterator[tuple[str, str] | Exception]:
        """Stream (kind, text) tuples to user, yielding Exception on failure."""
        try:
            async for kind, text in self._chat_completion_stream_separated(
                base_url, messages, model_name
            ):
                yield (kind, text)
        except Exception as e:
            log.warning("Model %s failed mid-stream: %s", model_name, e)
            yield e

    async def generate_question(
        self,
        prompt: str,
    ) -> AsyncIterator[RunnerEvent]:
        """Generate a clarifying multiple-choice question for the user.

        Streams the question text via ("text", ...) events.
        """
        self._cancelled = False

        try:
            primary_url, primary_name, _, _ = await self._resolve_models()
        except RuntimeError as e:
            yield ("error", str(e))
            return

        question_prompt = (
            "You are about to help with the following task:\n\n"
            f"{prompt}\n\n"
            "Before you begin planning, generate a single multiple-choice question "
            "that will help you understand the user's preferred approach. "
            "The question should have 3-5 numbered options. "
            "The user may select one or multiple options.\n\n"
            "Output ONLY the question and numbered options, nothing else."
        )
        messages = [{"role": "user", "content": question_prompt}]

        try:
            async for item in self._safe_stream_to_user(
                primary_url, messages, primary_name
            ):
                if self._cancelled:
                    yield ("cancelled", None)
                    return
                if isinstance(item, Exception):
                    yield ("text", f"\n\n[{primary_name} error: {item}]")
                else:
                    _kind, text = item
                    yield ("text", text)

            yield ("text", "\n\nReply with one or more numbers (e.g. 1,3) or your own answer.")
        except asyncio.CancelledError:
            yield ("cancelled", None)
        except Exception as e:
            log.exception("generate_question error")
            yield ("error", str(e))

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncIterator[RunnerEvent]:
        """Async generator yielding RunnerEvent tuples for the full debate.

        The prompt should already include the user's approach preference
        (appended by the runtime after the question phase).
        """
        self._cancelled = False
        start = time.monotonic()
        self._log_prompt(prompt)

        try:
            primary_url, primary_name, secondary_url, secondary_name = (
                await self._resolve_models()
            )
        except RuntimeError as e:
            yield ("error", str(e))
            return

        plan_prompt = (
            "You are participating in a collaborative planning process. "
            "Please propose a detailed plan to address the following:\n\n"
            f"{prompt}\n\n"
            "Be specific and thorough in your plan."
        )
        messages = [{"role": "user", "content": plan_prompt}]

        try:
            # ── Round 1: Both models propose plans ──
            yield ("text", f"## Plan A — {primary_name}\n\n")

            # Start secondary plan in background
            b_task = asyncio.ensure_future(
                self._safe_collect(secondary_url, messages, secondary_name)
            )

            # Stream primary plan
            plan_a = _StreamResult()
            async for item in self._safe_stream_to_user(
                primary_url, messages, primary_name
            ):
                if self._cancelled:
                    b_task.cancel()
                    yield ("cancelled", None)
                    return
                if isinstance(item, Exception):
                    plan_a.error = str(item)
                    yield ("text", f"\n\n[{primary_name} disconnected: {item}]")
                else:
                    kind, text = item
                    if kind == "thinking":
                        plan_a.thinking.append(text)
                    else:
                        plan_a.content.append(text)
                    yield ("text", text)

            self._log_response(f"[{primary_name} plan] {plan_a.content_text}")

            if self._cancelled:
                b_task.cancel()
                yield ("cancelled", None)
                return

            # Wait for secondary plan
            plan_b: _StreamResult = await b_task
            self._log_response(f"[{secondary_name} plan] {plan_b.content_text}")

            if plan_b.full_text:
                yield ("text", f"\n\n---\n\n## Plan B — {secondary_name}\n\n{plan_b.full_text}")
            if plan_b.error:
                yield ("text", f"\n\n[{secondary_name} error: {plan_b.error}]")

            if self._cancelled:
                yield ("cancelled", None)
                return

            # If both models failed completely, stop here
            if not plan_a.content_text and not plan_b.content_text:
                yield ("text", "\n\n[Both models failed — cannot continue debate]")
                duration = time.monotonic() - start
                yield ("result", {
                    "engine": "debate",
                    "model": f"{primary_name} vs {secondary_name}",
                    "duration_s": round(duration, 2),
                    "summary": f"debate failed | {duration:.1f}s",
                })
                return

            # ── Round 2: Synthesis by primary (Qwen) ──
            synth_parts = [
                "You are synthesizing a pre-final plan from two proposals.\n\n"
                f"Original request: \"{prompt}\"\n\n"
            ]
            if plan_a.content_text:
                synth_parts.append(f"**Plan A** (from {primary_name}):\n{plan_a.content_text}\n\n")
            if plan_b.content_text:
                synth_parts.append(f"**Plan B** (from {secondary_name}):\n{plan_b.content_text}\n\n")

            synth_parts.append(
                "Merge the best elements of both plans into a single pre-final plan. "
                "Output your plan as a numbered list of concrete, actionable steps. "
                "For each step, specify:\n"
                "- What to do (action)\n"
                "- Which files to create or modify (if applicable)\n"
                "- Any commands to run (if applicable)\n\n"
                "Be precise. No prose introductions or conclusions — just the steps."
            )

            synthesis_prompt = "".join(synth_parts)
            synthesis_messages = [{"role": "user", "content": synthesis_prompt}]

            yield ("text", f"\n\n---\n\n## Pre-final Synthesis — {primary_name}\n\n")

            pre_final = _StreamResult()
            async for item in self._safe_stream_to_user(
                primary_url, synthesis_messages, primary_name
            ):
                if self._cancelled:
                    yield ("cancelled", None)
                    return
                if isinstance(item, Exception):
                    pre_final.error = str(item)
                    yield ("text", f"\n\n[{primary_name} disconnected: {item}]")
                else:
                    kind, text = item
                    if kind == "thinking":
                        pre_final.thinking.append(text)
                    else:
                        pre_final.content.append(text)
                    yield ("text", text)

            self._log_response(f"[pre-final synthesis] {pre_final.content_text}")

            if self._cancelled:
                yield ("cancelled", None)
                return

            # If synthesis failed, fall back to best available plan
            if not pre_final.content_text:
                pre_final_text = plan_a.content_text or plan_b.content_text
                yield ("text", f"\n\n[Synthesis failed — using best available plan]")
            else:
                pre_final_text = pre_final.content_text

            # ── Round 3: Critique by secondary (GLM) ──
            critique_prompt = (
                f"You are reviewing a plan for: \"{prompt}\"\n\n"
                f"Pre-final plan:\n{pre_final_text}\n\n"
                "Critique this plan. What are its strengths? What are its weaknesses, "
                "gaps, or errors? What would you improve? Be specific and constructive."
            )
            critique_messages = [{"role": "user", "content": critique_prompt}]

            yield ("text", f"\n\n---\n\n## Critique — {secondary_name}\n\n")

            critique = _StreamResult()
            async for item in self._safe_stream_to_user(
                secondary_url, critique_messages, secondary_name
            ):
                if self._cancelled:
                    yield ("cancelled", None)
                    return
                if isinstance(item, Exception):
                    critique.error = str(item)
                    yield ("text", f"\n\n[{secondary_name} disconnected: {item}]")
                else:
                    kind, text = item
                    if kind == "thinking":
                        critique.thinking.append(text)
                    else:
                        critique.content.append(text)
                    yield ("text", text)

            self._log_response(f"[critique] {critique.content_text}")

            if self._cancelled:
                yield ("cancelled", None)
                return

            # ── Round 4: Final plan by primary (Qwen) incorporating critique ──
            final_parts = [
                "You are producing the final plan.\n\n"
                f"Original request: \"{prompt}\"\n\n"
                f"Pre-final plan:\n{pre_final_text}\n\n"
            ]
            if critique.content_text:
                final_parts.append(
                    f"Critique from {secondary_name}:\n{critique.content_text}\n\n"
                )
            final_parts.append(
                "Incorporate the valid points from the critique into the final plan. "
                "Output your plan as a numbered list of concrete, actionable steps. "
                "For each step, specify:\n"
                "- What to do (action)\n"
                "- Which files to create or modify (if applicable)\n"
                "- Any commands to run (if applicable)\n\n"
                "Be precise. No prose introductions or conclusions — just the steps."
            )

            final_prompt = "".join(final_parts)
            final_messages = [{"role": "user", "content": final_prompt}]

            yield ("text", f"\n\n---\n\n## Final Plan — {primary_name}\n\n")

            final_plan = _StreamResult()
            async for item in self._safe_stream_to_user(
                primary_url, final_messages, primary_name
            ):
                if self._cancelled:
                    yield ("cancelled", None)
                    return
                if isinstance(item, Exception):
                    final_plan.error = str(item)
                    yield ("text", f"\n\n[{primary_name} disconnected: {item}]")
                else:
                    kind, text = item
                    if kind == "thinking":
                        final_plan.thinking.append(text)
                    else:
                        final_plan.content.append(text)
                    yield ("text", text)

            self._log_response(f"[final plan] {final_plan.content_text}")

            # Use the best available final content
            handoff_text = (
                final_plan.content_text
                or pre_final_text
                or ""
            )

            # ── Stats ──
            duration = time.monotonic() - start
            yield (
                "result",
                {
                    "engine": "debate",
                    "model": f"{primary_name} vs {secondary_name}",
                    "duration_s": round(duration, 2),
                    "summary": (
                        f"debate {primary_name} vs {secondary_name} "
                        f"| {duration:.1f}s"
                    ),
                },
            )

            # Hand off the final plan (content only) for execution.
            if handoff_text.strip():
                yield ("handoff", handoff_text)

        except asyncio.CancelledError:
            yield ("cancelled", None)
        except Exception as e:
            log.exception("Debate runner error")
            yield ("error", str(e))

    def cancel(self) -> None:
        """Request cancellation of the running debate."""
        self._cancelled = True
