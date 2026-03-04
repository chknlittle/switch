"""Base runner functionality shared by runner implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from typing import Dict


@dataclass
class RunState:
    """Accumulates state during a runner execution."""

    start_time: datetime = field(default_factory=datetime.now)
    session_id: str | None = None
    text: str = ""
    tool_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    tokens_cache_read: int = 0
    tokens_cache_write: int = 0
    cost: float = 0.0
    saw_result: bool = False
    saw_error: bool = False
    raw_output: list[str] = field(default_factory=list)

    # Server-mode runners can emit message events for both user and assistant.
    # Track roles by message ID so we can ignore user echoes.
    message_roles: Dict[str, str] = field(default_factory=dict)

    # Track seen tool IDs to deduplicate SSE updates for the same tool call.
    seen_tool_ids: set = field(default_factory=set)

    # For streaming tool events, input/args can arrive in a later update for the
    # same tool ID. Track which tool IDs we've already logged input for.
    tool_input_logged_ids: set = field(default_factory=set)

    # Track tool IDs whose plain headers were later upgraded with details
    # (e.g. command preview arriving in a follow-up SSE update).
    tool_header_upgraded_ids: set = field(default_factory=set)

    # Track seen tool-result IDs to avoid duplicate result summaries when the
    # server emits multiple updates for the same finished tool call.
    tool_result_seen_ids: set = field(default_factory=set)

    @property
    def duration_s(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()


class BaseRunner:
    """Base class for CLI runners."""

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
    ):
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.session_name = session_name
        self.output_file: Path | None = None

        if session_name:
            output_dir.mkdir(exist_ok=True)
            self.output_file = output_dir / f"{session_name}.log"

    def _log_to_file(self, content: str) -> None:
        """Append content to the output log file."""
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(content)

    def _log_prompt(self, prompt: str) -> None:
        """Log the prompt to the output file."""
        self._log_to_file(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Prompt: {prompt}\n"
        )

    def _log_response(self, text: str) -> None:
        """Log a response to the output file."""
        self._log_to_file(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] Response:\n{text}\n"
        )
