"""Runner registry.

This provides a single place to map an engine name to its concrete runner
implementation. Callers should depend on the `Runner` port.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.runners.opencode.config import OpenCodeConfig
from src.runners.pi.config import PiConfig
from src.runners.weaver.config import WeaverConfig

if TYPE_CHECKING:
    from src.runners.ports import Runner


def create_runner(
    engine: str,
    *,
    working_dir: str,
    output_dir: Path,
    session_name: str | None = None,
    pi_config: PiConfig | None = None,
    opencode_config: OpenCodeConfig | None = None,
    weaver_config: WeaverConfig | None = None,
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

    if engine == "opencode":
        from src.runners.opencode.runner import OpenCodeRunner

        return OpenCodeRunner(
            working_dir,
            output_dir,
            session_name,
            config=opencode_config,
        )

    if engine == "weaver":
        from src.runners.weaver.runner import WeaverRunner

        return WeaverRunner(
            working_dir,
            output_dir,
            session_name,
            config=weaver_config,
        )

    raise ValueError(f"Unknown engine: {engine}")
