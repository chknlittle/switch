"""Engine registry and helpers for session behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass

OPENCODE_MODEL_DEFAULT = os.getenv(
    "OPENCODE_MODEL_DEFAULT", "glm_vllm/glm-4.7-flash-heretic.Q8_0.gguf"
)
OPENCODE_MODEL_GPT = "openai/gpt-5.2"
OPENCODE_MODEL_CODEX = "openai/codex-5.3"
OPENCODE_MODEL_ZEN = "opencode/glm-4.7"
OPENCODE_MODEL_GPT_OR = "openrouter/openai/gpt-5.2"
OPENCODE_MODEL_KIMI_CODING = "kimi-for-coding/kimi-k2.5"


@dataclass(frozen=True)
class EngineSpec:
    name: str
    supports_reasoning: bool


ENGINE_SPECS = {
    "claude": EngineSpec(name="claude", supports_reasoning=False),
    "opencode": EngineSpec(name="opencode", supports_reasoning=True),
}

ENGINE_ALIASES = {
    "cc": "claude",
    "claude": "claude",
    "oc": "opencode",
    "opencode": "opencode",
}


def get_engine_spec(engine: str) -> EngineSpec | None:
    return ENGINE_SPECS.get(engine)


def normalize_engine(engine: str) -> str | None:
    return ENGINE_ALIASES.get(engine.lower())


def opencode_model_for_agent(agent: str | None) -> str:
    if agent == "bridge-gpt":
        return OPENCODE_MODEL_GPT
    if agent == "bridge-codex":
        return OPENCODE_MODEL_CODEX
    if agent == "bridge-zen":
        return OPENCODE_MODEL_ZEN
    if agent == "bridge-gpt-or":
        return OPENCODE_MODEL_GPT_OR
    if agent == "bridge-kimi-coding":
        return OPENCODE_MODEL_KIMI_CODING
    return OPENCODE_MODEL_DEFAULT
