"""Engine registry and helpers for session behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass

PI_MODEL_DEFAULT = os.getenv("PI_MODEL_DEFAULT", "")


@dataclass(frozen=True)
class EngineSpec:
    name: str
    supports_reasoning: bool


ENGINE_SPECS = {
    "claude": EngineSpec(name="claude", supports_reasoning=False),
    "opencode": EngineSpec(name="opencode", supports_reasoning=True),
    "pi": EngineSpec(name="pi", supports_reasoning=True),
    "vllm-direct": EngineSpec(name="vllm-direct", supports_reasoning=False),
}

ENGINE_ALIASES = {
    "cc": "claude",
    "claude": "claude",
    "oc": "opencode",
    "opencode": "opencode",
    "pi": "pi",
    "vllm-direct": "vllm-direct",
}


def get_engine_spec(engine: str) -> EngineSpec | None:
    return ENGINE_SPECS.get(engine)


def normalize_engine(engine: str) -> str | None:
    return ENGINE_ALIASES.get(engine.lower())
