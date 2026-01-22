#!/usr/bin/env python3
"""OpenCode CLI runner for XMPP bridge."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger("opencode")


@dataclass
class OpenCodeResult:
    text: str
    session_id: str | None
    cost: float
    tokens_in: int
    tokens_out: int
    tokens_reasoning: int
    tokens_cache_read: int
    tokens_cache_write: int
    duration_s: float
    tool_count: int


class OpenCodeRunner:
    """Runs OpenCode CLI and parses json output."""

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        model: str | None = None,
        reasoning_mode: str = "normal",
        agent: str = "bridge",
    ):
        self.working_dir = working_dir
        self.process: asyncio.subprocess.Process | None = None
        self.session_name = session_name
        self.output_file: Path | None = None
        self.model = model
        self.reasoning_mode = reasoning_mode
        self.agent = agent
        if session_name:
            output_dir.mkdir(exist_ok=True)
            self.output_file = output_dir / f"{session_name}.log"

    async def run(self, prompt: str, session_id: str | None = None):
        """Run OpenCode, yielding (event_type, content) tuples."""
        cmd = ["opencode", "run", "--format", "json", "--agent", self.agent]
        if session_id:
            cmd.extend(["--session", session_id])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.reasoning_mode == "high":
            cmd.extend(["--variant", "high"])
        cmd.extend(["--", prompt])

        log.info(f"OpenCode: {prompt[:50]}...")

        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Prompt: {prompt}\n")

        start_time = datetime.now()
        tool_count = 0
        last_text = ""
        tokens_in = 0
        tokens_out = 0
        tokens_reasoning = 0
        tokens_cache_read = 0
        tokens_cache_write = 0
        total_cost = 0.0

        current_session_id: str | None = None
        saw_result = False
        saw_error = False
        raw_output: list[str] = []

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.working_dir,
            )

            stdout = self.process.stdout
            if stdout is None:
                raise RuntimeError("OpenCode process stdout missing")

            async for raw_line in stdout:
                line = raw_line.decode().strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if line and len(raw_output) < 5:
                        raw_output.append(line)
                    continue

                if not isinstance(event, dict):
                    continue

                event_type = event.get("type")
                if event_type == "step_start":
                    session_value = event.get("sessionID")
                    if isinstance(session_value, str) and session_value:
                        current_session_id = session_value
                        yield ("session_id", session_value)

                elif event_type == "text":
                    part = event.get("part", {})
                    text = part.get("text", "") if isinstance(part, dict) else ""
                    if isinstance(text, str) and text:
                        last_text += text
                        yield ("text", text)
                        if self.output_file:
                            with open(self.output_file, "a") as f:
                                f.write(f"\n[TEXT]\n{text}\n")

                elif event_type == "tool_use":
                    part = event.get("part", {})
                    tool = part.get("tool") if isinstance(part, dict) else None
                    tool_state = part.get("state", {}) if isinstance(part, dict) else {}
                    title = (
                        tool_state.get("title")
                        if isinstance(tool_state, dict)
                        else None
                    )
                    tool_count += 1
                    if tool:
                        desc = f"[tool:{tool}]"
                        if title:
                            desc = f"[tool:{tool} {title}]"
                        yield ("tool", desc)
                        if self.output_file:
                            with open(self.output_file, "a") as f:
                                f.write(f"{desc}\n")

                elif event_type == "step_finish":
                    part = event.get("part", {})
                    tokens = part.get("tokens", {}) if isinstance(part, dict) else {}
                    cache = tokens.get("cache", {}) if isinstance(tokens, dict) else {}
                    tokens_in += int(tokens.get("input", 0) or 0)
                    tokens_out += int(tokens.get("output", 0) or 0)
                    tokens_reasoning += int(tokens.get("reasoning", 0) or 0)
                    tokens_cache_read += int(cache.get("read", 0) or 0)
                    tokens_cache_write += int(cache.get("write", 0) or 0)
                    total_cost += float(part.get("cost", 0) or 0)
                    reason = part.get("reason")
                    if reason == "stop":
                        duration_s = (datetime.now() - start_time).total_seconds()
                        saw_result = True
                        yield (
                            "result",
                            OpenCodeResult(
                                text=last_text,
                                session_id=current_session_id,
                                cost=total_cost,
                                tokens_in=tokens_in,
                                tokens_out=tokens_out,
                                tokens_reasoning=tokens_reasoning,
                                tokens_cache_read=tokens_cache_read,
                                tokens_cache_write=tokens_cache_write,
                                duration_s=duration_s,
                                tool_count=tool_count,
                            ),
                        )

                elif event_type == "error":
                    message = event.get("message")
                    error = event.get("error")
                    if isinstance(message, dict):
                        message = message.get("data", {}).get("message") or message.get(
                            "message"
                        )
                    if not message:
                        message = error
                    saw_error = True
                    yield ("error", str(message) if message else "OpenCode error")

            await self.process.wait()

            if self.process.returncode is not None and self.process.returncode != 0:
                saw_error = True

            if not saw_result and not saw_error:
                if raw_output:
                    preview = " | ".join(raw_output)
                    yield ("error", f"OpenCode output (non-JSON): {preview}")
                elif self.process.returncode:
                    yield (
                        "error",
                        f"OpenCode exited with code {self.process.returncode}",
                    )
                else:
                    yield ("error", "OpenCode exited without output")

        except Exception as e:
            log.exception("OpenCode runner error")
            yield ("error", str(e))

    def cancel(self):
        if self.process:
            self.process.terminate()
