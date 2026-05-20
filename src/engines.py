"""Engine registry and helpers for session behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass

PI_MODEL_DEFAULT = os.getenv("PI_MODEL_DEFAULT", "")


@dataclass(frozen=True)
class EngineSpec:
    name: str
    supports_reasoning: bool
    supports_runtime: bool = True
    supports_ralph: bool = True
    remote_session_attr: str | None = None


ENGINE_SPECS = {
    "claude": EngineSpec(
        name="claude",
        supports_reasoning=False,
        remote_session_attr="claude_session_id",
    ),
    "opencode": EngineSpec(
        name="opencode",
        supports_reasoning=True,
        remote_session_attr="opencode_session_id",
    ),
    "pi": EngineSpec(
        name="pi",
        supports_reasoning=True,
        remote_session_attr="pi_session_id",
    ),
    "cursor": EngineSpec(
        name="cursor",
        supports_reasoning=False,
        remote_session_attr="cursor_session_id",
    ),
    "vllm-direct": EngineSpec(
        name="vllm-direct",
        supports_reasoning=False,
        remote_session_attr=None,
    ),
}

ENGINE_ALIASES = {
    "cc": "claude",
    "claude": "claude",
    "oc": "opencode",
    "opencode": "opencode",
    "pi": "pi",
    "cursor": "cursor",
    "cu": "cursor",
    "vllm": "vllm-direct",
    "vllm-direct": "vllm-direct",
}


def get_engine_spec(engine: str) -> EngineSpec | None:
    return ENGINE_SPECS.get(engine)


def normalize_engine(engine: str) -> str | None:
    return ENGINE_ALIASES.get(engine.lower())


def runtime_engine_names() -> set[str]:
    return {name for name, spec in ENGINE_SPECS.items() if spec.supports_runtime}


def ralph_engine_names() -> set[str]:
    return {name for name, spec in ENGINE_SPECS.items() if spec.supports_ralph}


def reasoning_engine_names() -> set[str]:
    return {name for name, spec in ENGINE_SPECS.items() if spec.supports_reasoning}


def remote_session_attr(engine: str) -> str | None:
    spec = get_engine_spec(engine)
    return spec.remote_session_attr if spec else None


def remote_session_id_for(engine: str, session: object) -> str | None:
    attr = remote_session_attr(engine)
    if not attr:
        return None
    value = getattr(session, attr, None)
    return value if isinstance(value, str) and value else None
