"""OpenAI-compatible LLM client for llamacpp / local inference."""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from src.runners.weaver.config import WeaverConfig

log = logging.getLogger(__name__)


def create_client(config: WeaverConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key="local",  # llamacpp doesn't need a real key
    )


async def chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call the LLM and return the first choice message as a dict."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
    }
