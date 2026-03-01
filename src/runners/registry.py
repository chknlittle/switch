"""Runner registry.

This provides a single place to map an engine name to its concrete runner
implementation. Callers should depend on the `Runner` port.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.runners.debate.config import DebateConfig
from src.runners.pi.config import PiConfig

if TYPE_CHECKING:
    from src.runners.ports import Runner


def create_runner(
    engine: str,
    *,
    working_dir: str,
    output_dir: Path,
    session_name: str | None = None,
    pi_config: PiConfig | None = None,
    debate_config: DebateConfig | None = None,
) -> Runner:
    engine = (engine or "").strip().lower()

    if engine == "claude":
        from src.runners.claude.runner import ClaudeRunner
        return ClaudeRunner(working_dir, output_dir, session_name)

    if engine == "pi":
        from src.runners.pi.runner import PiRunner

        return PiRunner(
            working_dir,
            output_dir,
            session_name,
            config=pi_config,
        )

    if engine == "debate":
        from src.runners.debate.runner import DebateRunner

        return DebateRunner(
            working_dir,
            output_dir,
            session_name,
            config=debate_config,
        )

    raise ValueError(f"Unknown engine: {engine}")
