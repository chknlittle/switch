"""Ralph autonomous iteration loop."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.runners import ClaudeRunner

if TYPE_CHECKING:
    from src.db import RalphLoopRepository, SessionRepository


def parse_ralph_command(body: str) -> dict | None:
    """Parse /ralph command into components.

    Formats:
        /ralph <prompt> --max <N> --done "<promise>" --wait <M>
        /ralph <N> <prompt>  (shorthand)
        /ralph <prompt>  (infinite - dangerous!)

    Returns dict with: prompt, max_iterations, completion_promise, wait_minutes
    """
    if not body.lower().startswith("/ralph"):
        return None

    rest = body[6:].strip()
    if not rest:
        return None

    try:
        parts = shlex.split(rest)
    except ValueError:
        parts = rest.split()

    max_iterations = 0
    completion_promise = None
    wait_minutes = 2.0 / 60.0
    prompt_parts = []

    i = 0
    while i < len(parts):
        part = parts[i]
        if part in ("--max", "--max-iterations", "-m") and i + 1 < len(parts):
            try:
                max_iterations = int(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif part in ("--done", "--completion-promise", "-d") and i + 1 < len(parts):
            completion_promise = parts[i + 1]
            i += 2
            continue
        elif part in ("--wait", "--wait-min", "--wait-minutes", "--interval", "--sleep", "-w") and i + 1 < len(parts):
            try:
                wait_minutes = float(parts[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        prompt_parts.append(part)
        i += 1

    # Shorthand: first number is max iterations
    if prompt_parts and prompt_parts[0].isdigit():
        max_iterations = int(prompt_parts[0])
        prompt_parts = prompt_parts[1:]

    prompt = " ".join(prompt_parts)
    if not prompt:
        return None
    return {
        "prompt": prompt,
        "max_iterations": max_iterations,
        "completion_promise": completion_promise,
        "wait_minutes": max(0.0, wait_minutes),
    }


@dataclass
class IterationResult:
    """Result of a single Ralph iteration."""

    text: str = ""
    cost: float = 0.0
    tool_count: int = 0
    error: str | None = None
    session_id: str | None = None


class RalphLoop:
    """Manages an autonomous iteration loop."""

    def __init__(
        self,
        session_bot,
        prompt: str,
        working_dir: str,
        output_dir: Path,
        max_iterations: int = 0,
        completion_promise: str | None = None,
        wait_minutes: float = 2.0 / 60.0,
        sessions: "SessionRepository | None" = None,
        ralph_loops: "RalphLoopRepository | None" = None,
    ):
        self.bot = session_bot
        self.prompt = prompt
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.completion_promise = completion_promise
        self.wait_seconds = max(0.0, wait_minutes) * 60.0
        self.sessions = sessions
        self.ralph_loops = ralph_loops
        self.current_iteration = 0
        self.total_cost = 0.0
        self.cancelled = False
        self._cancel_event = asyncio.Event()
        self.loop_id: int | None = None
        self.log = logging.getLogger(f"ralph.{session_bot.session_name}")

    def cancel(self):
        """Signal loop to stop after current iteration."""
        self.cancelled = True
        self._cancel_event.set()
        self._save_state("cancelled")

    def _save_state(self, status: str = "running"):
        """Persist loop state to database."""
        if self.ralph_loops and self.loop_id:
            self.ralph_loops.update_progress(self.loop_id, self.current_iteration, self.total_cost, status)

    def _format_max(self) -> str:
        return str(self.max_iterations) if self.max_iterations > 0 else "unlimited"

    def _build_prompt(self) -> str:
        """Build the full prompt with iteration info and promise instructions."""
        iter_info = f"[Ralph iteration {self.current_iteration}"
        if self.max_iterations > 0:
            iter_info += f"/{self.max_iterations}"
        iter_info += "]"

        prompt = f"{iter_info}\n\n{self.prompt}"
        if self.completion_promise:
            prompt += (
                f"\n\nTo signal completion, output EXACTLY: "
                f"<promise>{self.completion_promise}</promise>\n"
                f"ONLY output this when the task is genuinely complete."
            )
        return prompt

    async def _run_iteration(self) -> IterationResult:
        """Execute a single iteration."""
        result = IterationResult()
        runner = ClaudeRunner(self.working_dir, self.output_dir, self.bot.session_name)
        self.bot.runner = runner

        # Get current session ID
        claude_session_id = None
        if self.sessions:
            session = self.sessions.get(self.bot.session_name)
            if session:
                claude_session_id = session.claude_session_id

        try:
            async for event_type, content in runner.run(self._build_prompt(), claude_session_id):
                if event_type == "session_id" and self.sessions:
                    self.sessions.update_claude_session_id(self.bot.session_name, content)
                    result.session_id = content
                elif event_type == "text":
                    result.text = content
                elif event_type == "tool":
                    result.tool_count += 1
                elif event_type == "result":
                    cost_match = re.search(r"\$(\d+\.?\d*)", content)
                    if cost_match:
                        result.cost = float(cost_match.group(1))
                elif event_type == "error":
                    result.error = content
        finally:
            self.bot.runner = None

        return result

    def _check_promise(self, text: str) -> bool:
        """Check if completion promise is in response."""
        if not self.completion_promise:
            return False
        return f"<promise>{self.completion_promise}</promise>" in text

    async def run(self):
        """Run the autonomous loop."""
        # Initialize DB record
        if self.ralph_loops:
            self.loop_id = self.ralph_loops.create(
                self.bot.session_name,
                self.prompt,
                self.max_iterations,
                self.completion_promise,
                self.wait_seconds,
            )

        promise_str = f'"{self.completion_promise}"' if self.completion_promise else "none"
        wait_minutes = self.wait_seconds / 60.0
        self.bot.send_reply(
            f"Ralph loop started\n"
            f"Max: {self._format_max()} | Wait: {wait_minutes:.2f} min | Done when: {promise_str}\n"
            f"Use /ralph-cancel to stop"
        )

        while True:
            # Check cancellation
            if self.cancelled:
                self.bot.send_reply(f"Ralph cancelled at iteration {self.current_iteration}\nTotal cost: ${self.total_cost:.3f}")
                break

            # Check max iterations
            if self.max_iterations > 0 and self.current_iteration >= self.max_iterations:
                self.bot.send_reply(f"Ralph complete: hit max ({self.max_iterations})\nTotal cost: ${self.total_cost:.3f}")
                self._save_state("max_iterations")
                break

            # Run iteration
            self.current_iteration += 1
            self._save_state()
            self.log.info(f"Ralph iteration {self.current_iteration}")

            result = await self._run_iteration()

            # Handle error
            if result.error:
                self.bot.send_reply(
                    f"Ralph error at iteration {self.current_iteration}: {result.error}\n"
                    f"Stopping. Total cost: ${self.total_cost:.3f}"
                )
                self._save_state("error")
                return

            # Update cost and send progress
            self.total_cost += result.cost
            preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
            iter_str = f"{self.current_iteration}/{self.max_iterations}" if self.max_iterations > 0 else str(self.current_iteration)
            self.bot.send_reply(f"[Ralph {iter_str} | {result.tool_count}tools ${result.cost:.3f}]\n\n{preview}")

            # Check promise completion
            if self._check_promise(result.text):
                self.bot.send_reply(
                    f"Ralph COMPLETE at iteration {self.current_iteration}\n"
                    f"Detected: <promise>{self.completion_promise}</promise>\n"
                    f"Total cost: ${self.total_cost:.3f}"
                )
                self._save_state("completed")
                return

            self._save_state()
            if self.wait_seconds > 0:
                try:
                    await asyncio.wait_for(self._cancel_event.wait(), timeout=self.wait_seconds)
                except asyncio.TimeoutError:
                    pass

        self._save_state("finished")
