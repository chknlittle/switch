"""Base runner functionality shared by Claude and OpenCode runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


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
        self._log_to_file(f"\n[{datetime.now().strftime('%H:%M:%S')}] Prompt: {prompt}\n")

    def _log_response(self, text: str) -> None:
        """Log a response to the output file."""
        self._log_to_file(f"\n[{datetime.now().strftime('%H:%M:%S')}] Response:\n{text}\n")
