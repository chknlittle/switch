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

    Round 1: Both models propose a plan for the user prompt.
    Round 2: Each model critiques the other's plan.
    Round 3: Model B (GLM) synthesizes a final plan from both plans + critiques.

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

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncIterator[RunnerEvent]:
        """Async generator yielding RunnerEvent tuples for the full debate."""
        self._cancelled = False
        start = time.monotonic()
        self._log_prompt(prompt)

        model_a_url = self._config.resolve_model_a_url()
        model_a_name = self._config.resolve_model_a_name()
        model_b_url = self._config.resolve_model_b_url()
        model_b_name = self._config.resolve_model_b_name()

        plan_prompt = (
            "You are participating in a collaborative planning process. "
            "Please propose a detailed plan to address the following:\n\n"
            f"{prompt}\n\n"
            "Be specific and thorough in your plan."
        )
        messages = [{"role": "user", "content": plan_prompt}]

        try:
            # ── Round 1: Both models propose plans ──
            yield ("text", f"## Plan A — {model_a_name}\n\n")

            # Start Model B plan in background
            b_task = asyncio.ensure_future(
                self._safe_collect(model_b_url, messages, model_b_name)
            )

            # Stream Model A plan — show thinking + content to user
            plan_a = _StreamResult()
            async for item in self._safe_stream_to_user(
                model_a_url, messages, model_a_name
            ):
                if self._cancelled:
                    b_task.cancel()
                    yield ("cancelled", None)
                    return
                if isinstance(item, Exception):
                    plan_a.error = str(item)
                    yield ("text", f"\n\n[{model_a_name} disconnected: {item}]")
                else:
                    kind, text = item
                    if kind == "thinking":
                        plan_a.thinking.append(text)
                    else:
                        plan_a.content.append(text)
                    yield ("text", text)

            self._log_response(f"[{model_a_name} plan] {plan_a.content_text}")

            if self._cancelled:
                b_task.cancel()
                yield ("cancelled", None)
                return

            # Wait for Model B plan
            plan_b: _StreamResult = await b_task
            self._log_response(f"[{model_b_name} plan] {plan_b.content_text}")

            if plan_b.full_text:
                yield ("text", f"\n\n---\n\n## Plan B — {model_b_name}\n\n{plan_b.full_text}")
            if plan_b.error:
                yield ("text", f"\n\n[{model_b_name} error: {plan_b.error}]")

            if self._cancelled:
                yield ("cancelled", None)
                return

            # If both models failed completely, stop here
            if not plan_a.content_text and not plan_b.content_text:
                yield ("text", "\n\n[Both models failed — cannot continue debate]")
                duration = time.monotonic() - start
                yield ("result", {
                    "engine": "debate",
                    "model": f"{model_a_name} vs {model_b_name}",
                    "duration_s": round(duration, 2),
                    "summary": f"debate failed | {duration:.1f}s",
                })
                return

            # ── Round 2: Critiques (only content passed, not thinking) ──
            critique_a = _StreamResult()
            critique_b = _StreamResult()

            if plan_a.content_text and plan_b.content_text:
                critique_prompt_for_a = (
                    f"Another model ({model_b_name}) proposed the following plan "
                    f"for: \"{prompt}\"\n\n"
                    f"Their plan:\n{plan_b.content_text}\n\n"
                    f"Critique this plan. What are its strengths? What are its weaknesses, "
                    f"gaps, or errors? What would you improve? Be specific and constructive."
                )
                critique_prompt_for_b = (
                    f"Another model ({model_a_name}) proposed the following plan "
                    f"for: \"{prompt}\"\n\n"
                    f"Their plan:\n{plan_a.content_text}\n\n"
                    f"Critique this plan. What are its strengths? What are its weaknesses, "
                    f"gaps, or errors? What would you improve? Be specific and constructive."
                )

                critique_a_messages = [{"role": "user", "content": critique_prompt_for_a}]
                critique_b_messages = [{"role": "user", "content": critique_prompt_for_b}]

                yield ("text", f"\n\n---\n\n## {model_a_name} critiques Plan B\n\n")

                # Start Model B critique in background
                b_critique_task = asyncio.ensure_future(
                    self._safe_collect(model_b_url, critique_b_messages, model_b_name)
                )

                # Stream Model A critique
                async for item in self._safe_stream_to_user(
                    model_a_url, critique_a_messages, model_a_name
                ):
                    if self._cancelled:
                        b_critique_task.cancel()
                        yield ("cancelled", None)
                        return
                    if isinstance(item, Exception):
                        yield ("text", f"\n\n[{model_a_name} disconnected: {item}]")
                    else:
                        kind, text = item
                        if kind == "thinking":
                            critique_a.thinking.append(text)
                        else:
                            critique_a.content.append(text)
                        yield ("text", text)

                self._log_response(f"[{model_a_name} critique] {critique_a.content_text}")

                if self._cancelled:
                    b_critique_task.cancel()
                    yield ("cancelled", None)
                    return

                critique_b = await b_critique_task
                self._log_response(f"[{model_b_name} critique] {critique_b.content_text}")

                if critique_b.full_text:
                    yield (
                        "text",
                        f"\n\n---\n\n## {model_b_name} critiques Plan A\n\n{critique_b.full_text}",
                    )
                if critique_b.error:
                    yield ("text", f"\n\n[{model_b_name} error: {critique_b.error}]")

            elif plan_a.content_text and not plan_b.content_text:
                yield ("text", f"\n\n---\n\n[Skipping critiques — only {model_a_name} produced a plan]")
            elif plan_b.content_text and not plan_a.content_text:
                yield ("text", f"\n\n---\n\n[Skipping critiques — only {model_b_name} produced a plan]")

            if self._cancelled:
                yield ("cancelled", None)
                return

            # ── Round 3: Structured synthesis by Model A ──
            # Only content (not thinking) from prior rounds goes into the synthesis prompt.
            synth_parts = [
                "You are synthesizing a final plan from a collaborative debate.\n\n"
                f"Original request: \"{prompt}\"\n\n"
            ]
            if plan_a.content_text:
                synth_parts.append(f"**Plan A** (from {model_a_name}):\n{plan_a.content_text}\n\n")
            if plan_b.content_text:
                synth_parts.append(f"**Plan B** (from {model_b_name}):\n{plan_b.content_text}\n\n")
            if critique_a.content_text:
                synth_parts.append(f"**{model_a_name}'s critique of Plan B**:\n{critique_a.content_text}\n\n")
            if critique_b.content_text:
                synth_parts.append(f"**{model_b_name}'s critique of Plan A**:\n{critique_b.content_text}\n\n")

            synth_parts.append(
                "Synthesize the best elements of both plans into a single final plan. "
                "Address the critiques raised by both models.\n\n"
                "Output your plan as a numbered list of concrete, actionable steps. "
                "For each step, specify:\n"
                "- What to do (action)\n"
                "- Which files to create or modify (if applicable)\n"
                "- Any commands to run (if applicable)\n\n"
                "Be precise. No prose introductions or conclusions — just the steps."
            )

            synthesis_prompt = "".join(synth_parts)
            synthesis_messages = [{"role": "user", "content": synthesis_prompt}]

            # GLM (Model B) synthesizes the final plan; fall back to Model A
            synth_url = model_b_url
            synth_name = model_b_name
            yield ("text", f"\n\n---\n\n## Final Synthesis — {synth_name}\n\n")

            synthesis = _StreamResult()
            synth_failed = False
            async for item in self._safe_stream_to_user(
                synth_url, synthesis_messages, synth_name
            ):
                if self._cancelled:
                    yield ("cancelled", None)
                    return
                if isinstance(item, Exception):
                    synth_failed = True
                    yield ("text", f"\n\n[{synth_name} disconnected: {item}]")
                else:
                    kind, text = item
                    if kind == "thinking":
                        synthesis.thinking.append(text)
                    else:
                        synthesis.content.append(text)
                    yield ("text", text)

            # If GLM failed synthesis completely, fall back to Model A (Qwen)
            if synth_failed and not synthesis.content_text:
                synth_url = model_a_url
                synth_name = model_a_name
                yield ("text", f"\n\n---\n\n## Final Synthesis (fallback) — {synth_name}\n\n")

                synthesis = _StreamResult()
                async for item in self._safe_stream_to_user(
                    synth_url, synthesis_messages, synth_name
                ):
                    if self._cancelled:
                        yield ("cancelled", None)
                        return
                    if isinstance(item, Exception):
                        yield ("text", f"\n\n[{synth_name} also failed: {item}]")
                    else:
                        kind, text = item
                        if kind == "thinking":
                            synthesis.thinking.append(text)
                        else:
                            synthesis.content.append(text)
                        yield ("text", text)

            self._log_response(f"[synthesis] {synthesis.content_text}")

            # ── Stats ──
            duration = time.monotonic() - start
            yield (
                "result",
                {
                    "engine": "debate",
                    "model": f"{model_a_name} vs {model_b_name}",
                    "duration_s": round(duration, 2),
                    "summary": (
                        f"debate {model_a_name} vs {model_b_name} "
                        f"| {duration:.1f}s"
                    ),
                },
            )

            # Hand off the synthesized plan (content only) for execution.
            if synthesis.content_text.strip():
                yield ("handoff", synthesis.content_text)

        except asyncio.CancelledError:
            yield ("cancelled", None)
        except Exception as e:
            log.exception("Debate runner error")
            yield ("error", str(e))

    def cancel(self) -> None:
        """Request cancellation of the running debate."""
        self._cancelled = True
