"""Debate runner configuration — resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DebateConfig:
    model_a_url: str | None = None
    model_a_name: str | None = None
    model_b_url: str | None = None
    model_b_name: str | None = None

    def resolve_model_a_url(self) -> str:
        return self.model_a_url or os.getenv(
            "DEBATE_MODEL_A_URL", "http://127.0.0.1:8080"
        )

    def resolve_model_a_name(self) -> str:
        return self.model_a_name or os.getenv(
            "DEBATE_MODEL_A_NAME", "Qwen3.5-35B-A3B"
        )

    def resolve_model_b_url(self) -> str:
        return self.model_b_url or os.getenv(
            "DEBATE_MODEL_B_URL", "http://127.0.0.1:8081"
        )

    def resolve_model_b_name(self) -> str:
        return self.model_b_name or os.getenv(
            "DEBATE_MODEL_B_NAME", "GLM-4.7-Flash"
        )
