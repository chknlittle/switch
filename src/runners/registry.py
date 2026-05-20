"""Runner registry.

This provides a single place to map an engine name to its concrete runner
implementation. Callers should depend on the `Runner` port.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from src.engines import ENGINE_SPECS
from src.runners.claude.config import ClaudeConfig
from src.runners.cursor.config import CursorConfig
from src.runners.opencode.config import OpenCodeConfig
from src.runners.pi.config import PiConfig

if TYPE_CHECKING:
    from src.runners.ports import Runner


def _create_claude_runner(
    working_dir: str,
    output_dir: Path,
    session_name: str | None,
    *,
    claude_config: ClaudeConfig | None = None,
    **_kwargs: object,
) -> Runner:
    from src.runners.claude.runner import ClaudeRunner

    return ClaudeRunner(working_dir, output_dir, session_name, config=claude_config)


def _create_pi_runner(
    working_dir: str,
    output_dir: Path,
    session_name: str | None,
    *,
    pi_config: PiConfig | None = None,
    **_kwargs: object,
) -> Runner:
    from src.runners.pi.runner import PiRunner

    return PiRunner(working_dir, output_dir, session_name, config=pi_config)


def _create_opencode_runner(
    working_dir: str,
    output_dir: Path,
    session_name: str | None,
    *,
    opencode_config: OpenCodeConfig | None = None,
    **_kwargs: object,
) -> Runner:
    from src.runners.opencode.runner import OpenCodeRunner

    return OpenCodeRunner(
        working_dir,
        output_dir,
        session_name,
        config=opencode_config,
    )


def _create_cursor_runner(
    working_dir: str,
    output_dir: Path,
    session_name: str | None,
    *,
    cursor_config: CursorConfig | None = None,
    **_kwargs: object,
) -> Runner:
    from src.runners.cursor.runner import CursorACPRunner

    return CursorACPRunner(
        working_dir,
        output_dir,
        session_name,
        config=cursor_config,
    )


def _create_vllm_direct_runner(
    working_dir: str,
    output_dir: Path,
    session_name: str | None,
    *,
    pi_config: PiConfig | None = None,
    opencode_config: OpenCodeConfig | None = None,
    **_kwargs: object,
) -> Runner:
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


RUNNER_FACTORIES: dict[str, Callable[..., Runner]] = {
    "claude": _create_claude_runner,
    "pi": _create_pi_runner,
    "opencode": _create_opencode_runner,
    "cursor": _create_cursor_runner,
    "vllm-direct": _create_vllm_direct_runner,
}

assert set(RUNNER_FACTORIES) == set(ENGINE_SPECS)


def create_runner(
    engine: str,
    *,
    working_dir: str,
    output_dir: Path,
    session_name: str | None = None,
    pi_config: PiConfig | None = None,
    opencode_config: OpenCodeConfig | None = None,
    claude_config: ClaudeConfig | None = None,
    cursor_config: CursorConfig | None = None,
) -> Runner:
    engine = (engine or "").strip().lower()

    factory = RUNNER_FACTORIES.get(engine)
    if factory is None:
        raise ValueError(f"Unknown engine: {engine}")

    return factory(
        working_dir,
        output_dir,
        session_name,
        pi_config=pi_config,
        opencode_config=opencode_config,
        claude_config=claude_config,
        cursor_config=cursor_config,
    )
