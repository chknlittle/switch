"""Claude Code CLI runner."""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import AsyncIterator

from src.runners.base import BaseRunner, RunState
from src.runners.claude.processor import ClaudeEventProcessor
from src.runners.pipeline import JSONLineStats, iter_json_line_pipeline
from src.runners.subprocess_transport import SubprocessTransport
from src.runners.ports import RunnerEvent

log = logging.getLogger("claude")

Event = RunnerEvent


class ClaudeRunner(BaseRunner):
    """Runs Claude Code and streams parsed events."""

    # Claude Code CLI flags vary by version. We default to attempting to enable
    # thinking, but fall back gracefully if the local CLI doesn't recognize the
    # flag(s).
    _THINKING_ARG_TRIES: tuple[tuple[str, ...], ...] = (
        ("--thinking",),
        ("--thinking", "high"),
    )

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)
        self._transport = SubprocessTransport()
        self._processor = ClaudeEventProcessor(
            log_to_file=self._log_to_file,
            log_response=self._log_response,
        )

    def _build_command(
        self,
        prompt: str,
        session_id: str | None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build the claude command line."""
        cmd = [
            "claude", "-p", prompt,
            "--model", "opus",
            "--output-format", "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]

        if extra_args:
            cmd.extend(extra_args)

        if session_id:
            cmd.extend(["--resume", session_id])
        return cmd

    def _thinking_args(self) -> list[list[str]]:
        """Return thinking flag candidates.

        Can be overridden via SWITCH_CLAUDE_THINKING_ARGS. Example:
            SWITCH_CLAUDE_THINKING_ARGS='--thinking high'
        """
        raw = os.getenv("SWITCH_CLAUDE_THINKING_ARGS")
        if raw and raw.strip():
            return [shlex.split(raw.strip())]
        return [list(args) for args in self._THINKING_ARG_TRIES]

    @staticmethod
    def _looks_like_unknown_flag_error(lines: list[str]) -> bool:
        haystack = "\n".join(lines).lower()
        needles = (
            "unknown option",
            "unrecognized option",
            "unknown argument",
            "unexpected argument",
            "invalid option",
            "unknown flag",
            "no such option",
            "flag provided but not defined",
        )
        return any(n in haystack for n in needles)

    async def run(
        self, prompt: str, session_id: str | None = None
    ) -> AsyncIterator[Event]:
        """Run Claude, yielding (event_type, content) tuples.

        Events:
            ("session_id", str) - Session ID for continuity
            ("text", str) - Response text
            ("tool", str) - Tool invocation description
            ("result", dict) - Final result stats payload
            ("error", str) - Error message
        """
        log.info(f"Claude: {prompt[:50]}...")
        self._log_prompt(prompt)

        thinking_tries = self._thinking_args()
        attempt_args: list[list[str] | None] = thinking_tries + [None]

        for idx, extra_args in enumerate(attempt_args, 1):
            state = RunState()
            cmd = self._build_command(prompt, session_id, extra_args=extra_args)
            emitted_any = False
            non_json_lines: list[str] = []

            try:
                stdout = await self._transport.start(
                    cmd,
                    cwd=self.working_dir,
                    stdout_limit=10 * 1024 * 1024,
                )

                stats = JSONLineStats()
                async for result in iter_json_line_pipeline(
                    byte_stream=stdout,
                    state=state,
                    parse_event=self._processor.parse_event,
                    stats=stats,
                ):
                    yield result

                emitted_any = stats.emitted_any
                non_json_lines = stats.non_json_lines

                returncode = await self._transport.wait()

                # If we got no structured events and the process failed, this is
                # often a CLI flag mismatch. Retry with the next arg variant.
                if (
                    not emitted_any
                    and returncode != 0
                    and extra_args is not None
                    and self._looks_like_unknown_flag_error(non_json_lines)
                    and idx < len(attempt_args)
                ):
                    log.warning(
                        "Claude CLI rejected thinking flags; retrying without them"
                    )
                    continue

                # If we got nothing at all, surface the raw output.
                if not emitted_any and non_json_lines:
                    yield ("error", "Claude runner produced no JSON events:\n" + "\n".join(non_json_lines))

                break

            except Exception as e:
                log.exception("Claude runner error")
                yield ("error", str(e))
                break

    def cancel(self) -> None:
        """Terminate the running process."""
        self._transport.cancel()

    async def cleanup(self) -> None:
        """Terminate and force-kill if the process doesn't exit."""
        await self._transport.cancel_and_kill()
