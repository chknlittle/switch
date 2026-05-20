"""Best-effort vLLM inference abort when OpenCode cancel is insufficient."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db import SessionRepository


class VllmAbortMixin:
    """Mixin for SessionBot: Helga vLLM hard-cancel nudge on /cancel."""

    log: logging.Logger
    session_name: str
    sessions: SessionRepository
    _last_vllm_abort_ts: float
    _vllm_abort_task: asyncio.Task | None

    def _maybe_abort_vllm_inference(self) -> None:
        """Best-effort: ask Helga vLLM to stop active inference.

        This is intentionally conservative:
        - Only triggers for sessions using vLLM-backed models (Pi, OpenCode, vllm-direct)
        - Only triggers for known vLLM-backed models (glm_vllm/, heretic_local/, heretic-v2, etc.)
        - Cooldown + single in-flight task to avoid request storms
        """

        enabled = os.getenv("SWITCH_VLLM_HARD_CANCEL", "1").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return

        session = self.sessions.get(self.session_name)
        if not session:
            return
        engine = (session.active_engine or "").strip().lower()
        model_id = (session.model_id or "").strip()

        # Check if this is a vLLM-backed engine
        vllm_engines = {"pi", "opencode", "vllm-direct"}
        if engine not in vllm_engines:
            return

        # Check if this is a vLLM-backed model
        is_vllm_model = (
            model_id.startswith("glm_vllm/")
            or model_id.startswith("heretic_local/")
            or "heretic" in model_id.lower()
            or model_id.startswith("qwen35-")  # Heretic models often use qwen35 prefix
        )
        if not is_vllm_model:
            return

        # Avoid spamming this if multiple cancellation paths fire.
        try:
            cooldown_s = float(os.getenv("SWITCH_VLLM_HARD_CANCEL_COOLDOWN_S", "10"))
        except ValueError:
            cooldown_s = 10.0
        now = time.monotonic()
        if now - self._last_vllm_abort_ts < max(0.0, cooldown_s):
            return
        self._last_vllm_abort_ts = now

        if self._vllm_abort_task and not self._vllm_abort_task.done():
            return

        self._vllm_abort_task = self.spawn_guarded(
            self._abort_vllm_inference(), context="session.vllm.abort_inference"
        )

    async def _abort_vllm_inference(self) -> None:
        """Call Helga vLLM control endpoints to stop active inference."""

        host = os.getenv("SWITCH_VLLM_SSH_HOST", "chkn_gpus").strip() or "chkn_gpus"
        health_url = os.getenv(
            "SWITCH_VLLM_HEALTH_URL", "http://127.0.0.1:8027/v1/models"
        )
        pause_url = os.getenv("SWITCH_VLLM_PAUSE_URL", "http://127.0.0.1:8027/pause")
        resume_url = os.getenv("SWITCH_VLLM_RESUME_URL", "http://127.0.0.1:8027/resume")

        try:
            timeout_s = float(os.getenv("SWITCH_VLLM_HARD_CANCEL_TIMEOUT_S", "90"))
        except ValueError:
            timeout_s = 90.0

        remote_cmd = (
            "set -euo pipefail; "
            f"curl -fsS -X POST {pause_url} >/dev/null; "
            "sleep 0.2; "
            f"curl -fsS -X POST {resume_url} >/dev/null; "
            f"curl -fsS {health_url} >/dev/null"
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            self.log.warning("vLLM hard abort: failed to spawn ssh: %s", e)
            return
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            with suppress(Exception):
                proc.kill()
            self.log.warning(
                "vLLM cancel nudge timed out host=%s pause=%s resume=%s timeout_s=%s",
                host,
                pause_url,
                resume_url,
                timeout_s,
            )
            return

        if proc.returncode != 0:
            out = (stdout or b"").decode("utf-8", errors="replace").strip()
            err = (stderr or b"").decode("utf-8", errors="replace").strip()
            self.log.warning(
                "vLLM cancel nudge failed (rc=%s) host=%s health=%s stdout=%s stderr=%s",
                proc.returncode,
                host,
                health_url,
                out[-1000:],
                err[-1000:],
            )
