#!/usr/bin/env python3
"""Ralph autonomous iteration loop for XMPP bridge."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from src.claude_runner import ClaudeRunner

if TYPE_CHECKING:
    from src.db import RalphLoopRepository, SessionRepository


def parse_ralph_command(body: str) -> dict | None:
    """Parse /ralph command into components.

    Formats supported:
      /ralph <prompt> --max <N> --done "<promise>"
      /ralph <N> <prompt>  (shorthand: first number is max iterations)
      /ralph <prompt>  (infinite loop - dangerous!)

    Returns:
        Dict with: prompt, max_iterations, completion_promise
        Or None if not a ralph command.
    """
    if not body.lower().startswith("/ralph"):
        return None

    rest = body[6:].strip()
    if not rest:
        return None

    max_iterations = 0
    completion_promise = None

    try:
        parts = shlex.split(rest)
    except ValueError:
        parts = rest.split()

    prompt_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in ("--max", "--max-iterations", "-m"):
            if i + 1 < len(parts):
                try:
                    max_iterations = int(parts[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
        elif part in ("--done", "--completion-promise", "-d"):
            if i + 1 < len(parts):
                completion_promise = parts[i + 1]
                i += 2
                continue
        prompt_parts.append(part)
        i += 1

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
    }


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
        sessions: "SessionRepository | None" = None,
        ralph_loops: "RalphLoopRepository | None" = None,
    ):
        self.session_bot = session_bot
        self.prompt = prompt
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.completion_promise = completion_promise
        self.sessions = sessions
        self.ralph_loops = ralph_loops
        self.current_iteration = 0
        self.total_cost = 0.0
        self.cancelled = False
        self.loop_id: int | None = None
        self.log = logging.getLogger(f"ralph.{session_bot.session_name}")

    def _save_state(self, status: str = "running"):
        """Save loop state to database."""
        if not self.ralph_loops or not self.loop_id:
            return
        self.ralph_loops.update_progress(
            self.loop_id,
            self.current_iteration,
            self.total_cost,
            status,
        )

    def cancel(self):
        """Signal the loop to stop after current iteration."""
        self.cancelled = True
        self._save_state("cancelled")

    async def run(self):
        """Run the autonomous loop."""
        if self.ralph_loops:
            self.loop_id = self.ralph_loops.create(
                self.session_bot.session_name,
                self.prompt,
                self.max_iterations,
                self.completion_promise,
            )

        max_str = str(self.max_iterations) if self.max_iterations > 0 else "unlimited"
        promise_str = (
            f'"{self.completion_promise}"' if self.completion_promise else "none"
        )
        self.session_bot.send_reply(
            f"Ralph loop started\n"
            f"Max: {max_str} | Done when: {promise_str}\n"
            f"Use /ralph-cancel to stop"
        )

        while True:
            if self.cancelled:
                self.session_bot.send_reply(
                    f"Ralph cancelled at iteration {self.current_iteration}\n"
                    f"Total cost: ${self.total_cost:.3f}"
                )
                break

            if (
                self.max_iterations > 0
                and self.current_iteration >= self.max_iterations
            ):
                self.session_bot.send_reply(
                    f"Ralph complete: hit max iterations ({self.max_iterations})\n"
                    f"Total cost: ${self.total_cost:.3f}"
                )
                self._save_state("max_iterations")
                break

            self.current_iteration += 1
            self._save_state()

            iter_info = f"[Ralph iteration {self.current_iteration}"
            if self.max_iterations > 0:
                iter_info += f"/{self.max_iterations}"
            iter_info += "]"

            full_prompt = f"{iter_info}\n\n{self.prompt}"
            if self.completion_promise:
                full_prompt += (
                    f"\n\nTo signal completion, output EXACTLY: "
                    f"<promise>{self.completion_promise}</promise>\n"
                    f"ONLY output this when the task is genuinely complete."
                )

            self.log.info(f"Ralph iteration {self.current_iteration}")

            runner = ClaudeRunner(
                self.working_dir,
                self.output_dir,
                session_name=self.session_bot.session_name,
            )
            self.session_bot.runner = runner

            # Get current session's claude_session_id
            claude_session_id = None
            if self.sessions:
                session = self.sessions.get(self.session_bot.session_name)
                if session:
                    claude_session_id = session.claude_session_id

            response_text = ""
            iteration_cost = 0.0
            tool_count = 0

            try:
                async for event_type, content in runner.run(
                    full_prompt, claude_session_id
                ):
                    if event_type == "session_id" and self.sessions:
                        self.sessions.update_claude_session_id(
                            self.session_bot.session_name, content
                        )
                    elif event_type == "text":
                        response_text = content
                    elif event_type == "tool":
                        tool_count += 1
                    elif event_type == "result":
                        cost_match = re.search(r"\$(\d+\.?\d*)", content)
                        if cost_match:
                            iteration_cost = float(cost_match.group(1))
                        self.total_cost += iteration_cost
                    elif event_type == "error":
                        self.session_bot.send_reply(
                            f"Ralph error at iteration {self.current_iteration}: {content}\n"
                            f"Stopping loop. Total cost: ${self.total_cost:.3f}"
                        )
                        self._save_state("error")
                        return
            finally:
                self.session_bot.runner = None

            preview = (
                response_text[:200] + "..."
                if len(response_text) > 200
                else response_text
            )
            self.session_bot.send_reply(
                f"[Ralph {self.current_iteration}"
                f"{'/' + str(self.max_iterations) if self.max_iterations > 0 else ''}"
                f" | {tool_count}tools ${iteration_cost:.3f}]\n\n{preview}"
            )

            if self.completion_promise:
                promise_tag = f"<promise>{self.completion_promise}</promise>"
                if promise_tag in response_text:
                    self.session_bot.send_reply(
                        f"Ralph COMPLETE at iteration {self.current_iteration}\n"
                        f"Detected: {promise_tag}\n"
                        f"Total cost: ${self.total_cost:.3f}"
                    )
                    self._save_state("completed")
                    return

            self._save_state()
            await asyncio.sleep(2)

        self._save_state("finished")
