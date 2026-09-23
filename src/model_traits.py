"""Which local models get Helga's vLLM pause/resume cancel."""

from __future__ import annotations


def is_vllm_served(model_id: str | None) -> bool:
    # Only the live Helga Qwen 3.8. Older Qwen, llama.cpp, and other local
    # names are left out so /cancel does not pause this server for them.
    return (model_id or "").strip().lower().startswith("qwen38_helga/")
