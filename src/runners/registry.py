"""Runner registry.

This provides a single place to map an engine name to its concrete runner
implementation. Callers should depend on the `Runner` port.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.runners.opencode.config import OpenCodeConfig
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
    opencode_config: OpenCodeConfig | None = None,
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

    if engine == "vllm-direct":
        from src.runners.vllm_direct.runner import VLLMDirectRunner

        model = None
        if opencode_config and opencode_config.model:
            model = opencode_config.model
        elif pi_config and pi_config.model:
            model = pi_config.model
        return VLLMDirectRunner(
            working_dir,
            output_dir,
            session_name,
            model=model,
        )

    raise ValueError(f"Unknown engine: {engine}")
