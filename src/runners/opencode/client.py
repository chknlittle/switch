"""HTTP client for the OpenCode server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Callable

import aiohttp

from src.attachments import Attachment
from src.runners.opencode.models import Question

log = logging.getLogger("opencode")


class OpenCodeClient:
    """HTTP + SSE transport for OpenCode."""

    def __init__(self, server_url: str | None = None):
        self.server_url = server_url or self._resolve_server_url()
        self._auth = self._build_auth()

    @property
    def auth(self) -> aiohttp.BasicAuth | None:
        return self._auth

    def _resolve_server_url(self) -> str:
        base_url = os.getenv("OPENCODE_SERVER_URL")
        if base_url:
            return base_url.rstrip("/")

        host = os.getenv("OPENCODE_SERVER_HOST", "127.0.0.1")
        port = os.getenv("OPENCODE_SERVER_PORT", "4096")
        return f"http://{host}:{port}"

    def _build_auth(self) -> aiohttp.BasicAuth | None:
        password = os.getenv("OPENCODE_SERVER_PASSWORD")
        if not password:
            return None
        username = os.getenv("OPENCODE_SERVER_USERNAME", "opencode")
        return aiohttp.BasicAuth(username, password)

    def _make_url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    async def request_json(
        self, session: aiohttp.ClientSession, method: str, url: str, **kwargs
    ) -> object | None:
        async with session.request(method, url, **kwargs) as resp:
            if resp.status == 204:
                return None
            text = await resp.text()
            if resp.status >= 400:
                detail = text.strip() or resp.reason
                raise RuntimeError(f"OpenCode HTTP {resp.status}: {detail}")
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

    async def check_health(self, session: aiohttp.ClientSession) -> None:
        url = self._make_url("/global/health")
        response = await self.request_json(session, "GET", url)
        if isinstance(response, dict) and response.get("healthy") is True:
            return
        raise RuntimeError("OpenCode server unhealthy or unreachable")

    async def create_session(
        self, session: aiohttp.ClientSession, session_name: str | None
    ) -> str:
        payload: dict[str, object] = {}
        if session_name:
            payload["title"] = session_name
        payload["permission"] = [{"permission": "*", "action": "allow", "pattern": "*"}]
        url = self._make_url("/session")
        response = await self.request_json(session, "POST", url, json=payload)
        if isinstance(response, dict):
            session_id = response.get("id") or response.get("sessionID")
            if isinstance(session_id, str) and session_id:
                return session_id
        raise RuntimeError("OpenCode session creation failed")

    async def send_message(
        self,
        session: aiohttp.ClientSession,
        session_id: str,
        prompt: str,
        attachments: list[Attachment] | None,
        model_payload: dict | None,
        agent: str,
        reasoning_mode: str,
    ) -> object | None:
        # Attachments are represented as local paths in the prompt text.
        parts: list[dict[str, object]] = [{"type": "text", "text": prompt}]

        body: dict[str, object] = {"parts": parts}
        if model_payload:
            body["model"] = model_payload
        if agent:
            body["agent"] = agent
        if reasoning_mode == "high" and model_payload:
            body["model"] = {**model_payload, "variant": "high"}
        url = self._make_url(f"/session/{session_id}/message")
        return await self.request_json(session, "POST", url, json=body)

    async def answer_question(
        self,
        session: aiohttp.ClientSession,
        question: Question,
        answers: list[list[str]],
    ) -> bool:
        url = self._make_url(f"/question/{question.request_id}/reply")
        try:
            await self.request_json(session, "POST", url, json={"answers": answers})
            log.info(f"Answered question {question.request_id}")
            return True
        except Exception as e:
            log.error(f"Failed to answer question: {e}")
            return False

    async def reject_question(
        self, session: aiohttp.ClientSession, question: Question
    ) -> bool:
        url = self._make_url(f"/question/{question.request_id}/reject")
        try:
            await self.request_json(session, "POST", url)
            return True
        except Exception as e:
            log.error(f"Failed to reject question: {e}")
            return False

    async def abort_session(
        self, session: aiohttp.ClientSession, session_id: str
    ) -> None:
        url = self._make_url(f"/session/{session_id}/abort")
        try:
            await self.request_json(session, "POST", url)
        except Exception as e:
            log.debug(f"Failed to abort session {session_id}: {e}")

    async def get_session_messages(
        self, session: aiohttp.ClientSession, session_id: str
    ) -> list[dict]:
        """Return the server's stored message list for a session."""
        url = self._make_url(f"/session/{session_id}/message")
        data = await self.request_json(session, "GET", url)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def _extract_assistant_text(self, messages: list[dict]) -> str | None:
        """Extract the latest assistant text (if any) from messages."""
        for msg in reversed(messages):
            info = msg.get("info")
            if not isinstance(info, dict):
                continue
            if info.get("role") != "assistant":
                continue
            parts = msg.get("parts")
            if not isinstance(parts, list):
                continue
            out: list[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        out.append(text)
            if out:
                return "".join(out)
        return None

    async def poll_assistant_text(
        self,
        session: aiohttp.ClientSession,
        session_id: str,
        *,
        timeout_s: float = 30.0,
        interval_s: float = 0.5,
    ) -> str | None:
        """Fallback for when SSE doesn't deliver text events.

        Polls the session's stored messages until an assistant message appears.
        """
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            messages = await self.get_session_messages(session, session_id)
            text = self._extract_assistant_text(messages)
            if text:
                return text
            await asyncio.sleep(interval_s)
        return None

    async def stream_events(
        self,
        session: aiohttp.ClientSession,
        queue: asyncio.Queue[dict],
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        # Prefer /global/event first. Some OpenCode server builds primarily emit
        # session/message events on the global bus, while /event can include only
        # heartbeats/file-watcher events.
        urls = [self._make_url("/global/event"), self._make_url("/event")]
        headers = {"Accept": "text/event-stream"}

        last_exc: Exception | None = None

        connect_timeout_s = float(os.getenv("OPENCODE_SSE_CONNECT_TIMEOUT_S", "10"))
        request_timeout = aiohttp.ClientTimeout(total=None)

        for url in urls:
            try:
                resp = await asyncio.wait_for(
                    session.get(url, headers=headers, timeout=request_timeout),
                    timeout=connect_timeout_s,
                )
                async with resp:
                    if resp.status == 404:
                        continue
                    if resp.status >= 400:
                        detail = (await resp.text()).strip() or resp.reason
                        raise RuntimeError(f"OpenCode SSE HTTP {resp.status}: {detail}")
                    await self.read_sse_stream(resp, queue, should_stop=should_stop)
                    return
            except asyncio.TimeoutError as e:
                last_exc = e
                log.warning(
                    f"OpenCode SSE connect timed out for {url} after {connect_timeout_s}s"
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                # This is a critical reliability issue; surface it at normal log level.
                log.warning(
                    f"OpenCode SSE connect failed for {url}: {type(e).__name__}: {e}"
                )

        if last_exc:
            raise RuntimeError(
                f"Failed to connect to OpenCode SSE stream (last error: {type(last_exc).__name__}: {last_exc})"
            ) from last_exc
        raise RuntimeError("Failed to connect to OpenCode SSE stream")

    async def read_sse_stream(
        self,
        resp: aiohttp.ClientResponse,
        queue: asyncio.Queue[dict],
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        # Avoid aiohttp's line-based iteration (`readline()`), which can raise
        # ValueError("Chunk too big") when a single SSE line exceeds the stream
        # reader limit. Instead, read raw chunks and split events on blank lines.
        max_buf = int(os.getenv("OPENCODE_SSE_MAX_BUFFER_BYTES", str(16 * 1024 * 1024)))

        buf = bytearray()

        def _split_event(buf_bytes: bytearray) -> tuple[bytes | None, int]:
            """Return (event_bytes, consumed_bytes) for the next event, if any."""
            idx_nl = buf_bytes.find(b"\n\n")
            idx_crlf = buf_bytes.find(b"\r\n\r\n")
            if idx_nl == -1 and idx_crlf == -1:
                return None, 0
            if idx_crlf != -1 and (idx_nl == -1 or idx_crlf < idx_nl):
                return bytes(buf_bytes[:idx_crlf]), idx_crlf + 4
            return bytes(buf_bytes[:idx_nl]), idx_nl + 2

        async for chunk in resp.content.iter_any():
            if should_stop and should_stop():
                break
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > max_buf:
                raise ValueError(f"SSE buffer too big ({len(buf)} bytes)")

            while True:
                event_bytes, consumed = _split_event(buf)
                if event_bytes is None:
                    break
                del buf[:consumed]

                if not event_bytes.strip():
                    continue

                data_lines: list[str] = []
                for raw_line in event_bytes.splitlines():
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[len("data:") :].lstrip())

                if not data_lines:
                    continue

                payload = "\n".join(data_lines)
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    await queue.put(event)

        # If the stream ends without a trailing blank line, ignore trailing bytes.
