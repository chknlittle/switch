"""Minimal OpenAI-compatible direct runner for vLLM-style endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp

from src.runners.base import BaseRunner

log = logging.getLogger("vllm_direct")


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class VLLMDirectRunner(BaseRunner):
    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        *,
        model: str | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)
        self._base_url = (
            os.getenv("SWITCH_VLLM_DIRECT_BASE_URL", "http://127.0.0.1:8030/v1")
            .strip()
            .rstrip("/")
        )
        self._model = (model or os.getenv("SWITCH_VLLM_DIRECT_MODEL", "")).strip()
        self._system_prompt = os.getenv("SWITCH_VLLM_DIRECT_SYSTEM_PROMPT", "").strip()
        self._temperature = _env_float("SWITCH_VLLM_DIRECT_TEMPERATURE", 0.5)
        self._min_p = _env_float("SWITCH_VLLM_DIRECT_MIN_P", 0.05)
        self._repetition_penalty = _env_float(
            "SWITCH_VLLM_DIRECT_REPETITION_PENALTY", 1.2
        )
        self._max_tokens = int(_env_float("SWITCH_VLLM_DIRECT_MAX_TOKENS", 1024))
        self._timeout_s = max(5.0, _env_float("SWITCH_VLLM_DIRECT_TIMEOUT_S", 600.0))
        self._cancel_requested = False
        self._request_task: asyncio.Task | None = None

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""

        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                return "".join(parts)

        text = first.get("text")
        return text if isinstance(text, str) else ""

    async def _post_chat(self, prompt: str) -> dict[str, Any]:
        if not self._model:
            raise RuntimeError("SWITCH_VLLM_DIRECT model is not configured")

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": self._temperature,
            "repetition_penalty": self._repetition_penalty,
            "max_tokens": self._max_tokens,
        }
        if self._min_p > 0:
            payload["min_p"] = self._min_p

        timeout = aiohttp.ClientTimeout(total=self._timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    detail = text.strip() or resp.reason
                    raise RuntimeError(f"HTTP {resp.status}: {detail}")
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    raise RuntimeError("Invalid JSON from direct vLLM endpoint") from e
                if not isinstance(data, dict):
                    raise RuntimeError("Unexpected response from direct vLLM endpoint")
                return data

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncIterator[tuple[str, object]]:
        del session_id
        self._cancel_requested = False
        self._log_prompt(prompt)
        log.info("vLLM direct: %s", prompt[:80])

        try:
            self._request_task = asyncio.create_task(self._post_chat(prompt))
            payload = await self._request_task
            if self._cancel_requested:
                yield ("cancelled", None)
                return

            text = self._extract_text(payload)
            self._log_response(text)
            if text:
                yield ("text", text)

            usage = payload.get("usage") if isinstance(payload, dict) else {}
            stats = {
                "text": text,
                "cost_usd": 0.0,
                "duration_ms": int(time.time() * 1000),
            }
            if isinstance(usage, dict):
                for src, dst in (
                    ("prompt_tokens", "tokens_in"),
                    ("completion_tokens", "tokens_out"),
                    ("total_tokens", "tokens_total"),
                ):
                    value = usage.get(src)
                    if isinstance(value, int):
                        stats[dst] = value
            yield ("result", stats)
        except asyncio.CancelledError:
            self._log_response("Cancelled.")
            yield ("cancelled", None)
            return
        except Exception as e:
            log.warning("vLLM direct request failed", exc_info=True)
            yield ("error", str(e))
        finally:
            self._request_task = None

    def cancel(self) -> None:
        self._cancel_requested = True
        task = self._request_task
        if task and not task.done():
            task.cancel()
