"""Pi coding agent runner using RPC mode (stdin/stdout JSON lines)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from src.runners.base import BaseRunner, RunState
from src.runners.pi.config import PiConfig
from src.runners.pi.processor import PiEventProcessor
from src.runners.ports import RunnerEvent

log = logging.getLogger("pi")

Event = RunnerEvent


class PiRunner(BaseRunner):
    """Runs Pi via RPC mode subprocess with stdin/stdout JSON lines.

    Each run spawns `pi --mode rpc`, sends a prompt command, reads events
    from stdout, and collects session stats on completion.
    """

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
        config: PiConfig | None = None,
    ):
        super().__init__(working_dir, output_dir, session_name)
        self._config = config or PiConfig()
        self._processor = PiEventProcessor(
            log_to_file=self._log_to_file,
            log_response=self._log_response,
        )
        self._process: asyncio.subprocess.Process | None = None
        self._cancelled = False
        self._stderr_task: asyncio.Task | None = None
        self._stderr_output: list[bytes] = []

    def _build_command(self, session_id: str | None) -> list[str]:
        pi_bin = self._config.resolve_bin()
        cmd = [pi_bin, "--mode", "rpc"]

        provider = self._config.resolve_provider()
        if provider:
            cmd.extend(["--provider", provider])
        model = self._config.resolve_model()
        if model:
            cmd.extend(["--model", model])
        if self._config.thinking:
            cmd.extend(["--thinking", self._config.thinking])

        session_dir = self._config.resolve_session_dir()
        if session_dir:
            cmd.extend(["--session-dir", session_dir])

        if session_id:
            cmd.extend(["--session", session_id])

        _DEFAULT_SYSTEM_PROMPT = (
            "NEVER use sudo or run commands as root. "
            "ALWAYS use virtual environments for ALL projects — "
            "python venvs for Python, local node_modules (never npm install -g) for Node. "
            "NEVER install packages globally. "
            "Bash commands timeout after 30s — for long-running processes "
            "(servers, builds, etc.) use nohup or launch in a tmux session. "
            "NEVER run a server in the foreground — it will block and hang. "
            "Stay focused on the task — do not explore the filesystem or read unrelated files."
        )
        if self._config.system_prompt is not None:
            sys_prompt = self._config.system_prompt
        else:
            # Try to load AGENTS.md for Switch-aware context.
            agents_md = Path(self.working_dir).expanduser() / "AGENTS.md"
            if not agents_md.is_file():
                agents_md = Path.home() / "AGENTS.md"
            if agents_md.is_file():
                try:
                    sys_prompt = agents_md.read_text().strip()
                except OSError:
                    sys_prompt = _DEFAULT_SYSTEM_PROMPT
            else:
                sys_prompt = _DEFAULT_SYSTEM_PROMPT
        if sys_prompt:
            cmd.extend(["--append-system-prompt", sys_prompt])

        return cmd

    async def _drain_stderr(self) -> None:
        """Read and discard stderr to prevent pipe buffer deadlock.

        Without this, Pi can block on stderr writes when the OS pipe buffer
        fills (~64KB), which prevents it from responding to stdin commands
        like get_session_stats — silently breaking session resumption.
        """
        if not self._process or not self._process.stderr:
            return
        try:
            while True:
                chunk = await self._process.stderr.read(8192)
                if not chunk:
                    break
                self._stderr_output.append(chunk)
        except (asyncio.CancelledError, ConnectionResetError):
            pass

    async def _send(self, msg: dict) -> None:
        if not self._process or not self._process.stdin:
            return
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

    async def _read_events(
        self, state: RunState
    ) -> AsyncIterator[Event]:
        if not self._process or not self._process.stdout:
            return

        while True:
            if self._cancelled:
                break

            try:
                raw = await self._process.stdout.readline()
            except asyncio.CancelledError:
                break

            if not raw:
                break

            line = raw.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            event_type = event.get("type")
            # Responses to our commands (prompt, abort, get_session_stats).
            if event_type == "response":
                cmd = event.get("command")
                if cmd == "get_session_stats" and event.get("success"):
                    state._stats_response = event  # type: ignore[attr-defined]
                continue

            # Agent lifecycle.
            if event_type == "agent_end":
                state.saw_result = True
                break

            for parsed in self._processor.parse_event(event, state):
                yield parsed

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> AsyncIterator[Event]:
        """Run Pi, yielding (event_type, content) tuples."""
        state = RunState()
        self._cancelled = False
        log.info(f"Pi: {prompt[:50]}...")
        self._log_prompt(prompt)

        cmd = self._build_command(session_id)
        log.debug(f"Pi command: {cmd}")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
                limit=10 * 1024 * 1024,
            )

            # Drain stderr in background to prevent pipe buffer deadlock.
            self._stderr_output = []
            self._stderr_task = asyncio.create_task(self._drain_stderr())

            # Send the prompt.
            await self._send({"type": "prompt", "message": prompt})

            # Stream events.
            async for event in self._read_events(state):
                yield event

            if self._cancelled:
                yield ("cancelled", None)
                return

            # Fetch session stats before closing.
            stats: dict | None = None
            if self._process.stdin and not self._process.stdin.is_closing():
                try:
                    await self._send(
                        {"type": "get_session_stats", "id": "stats"}
                    )
                    # Read until we get the stats response or EOF.
                    deadline = asyncio.get_event_loop().time() + 5.0
                    while asyncio.get_event_loop().time() < deadline:
                        if not self._process.stdout:
                            break
                        try:
                            raw = await asyncio.wait_for(
                                self._process.stdout.readline(), timeout=2.0
                            )
                        except asyncio.TimeoutError:
                            break
                        if not raw:
                            break
                        line = raw.decode(errors="replace").strip()
                        if not line:
                            continue
                        try:
                            resp = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (
                            isinstance(resp, dict)
                            and resp.get("type") == "response"
                            and resp.get("command") == "get_session_stats"
                        ):
                            stats = resp
                            break
                except Exception:
                    log.warning(
                        "Failed to fetch session stats for %s",
                        self.session_name,
                        exc_info=True,
                    )

            # Stats data is nested under "data" key.
            stats_data = stats.get("data", {}) if isinstance(stats, dict) else {}
            result = self._processor.make_result(state, stats_data)

            # Extract session path from stats for resume.
            session_path = None
            if isinstance(stats_data, dict):
                session_path = stats_data.get("sessionFile") or stats_data.get("session")
            if isinstance(session_path, str) and session_path:
                log.info("Pi session file for %s: %s", self.session_name, session_path)
                yield ("session_id", session_path)
            else:
                log.warning(
                    "No session file in Pi stats for %s (stats=%s)",
                    self.session_name,
                    "present" if stats else "None",
                )

            yield ("result", result)

        except Exception as e:
            log.exception("Pi runner error")
            yield ("error", str(e))
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        # Cancel stderr drainer first.
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
        self._stderr_task = None

        proc = self._process
        self._process = None
        if not proc:
            return

        if proc.stdin and not proc.stdin.is_closing():
            try:
                proc.stdin.close()
            except Exception:
                pass

        try:
            proc.terminate()
        except ProcessLookupError:
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def cancel(self) -> None:
        """Request cancellation of the running session."""
        self._cancelled = True
        if self._process and self._process.stdin and not self._process.stdin.is_closing():
            try:
                msg = json.dumps({"type": "abort"}, separators=(",", ":")) + "\n"
                self._process.stdin.write(msg.encode())
                # Best-effort, don't await drain.
            except Exception:
                pass

    async def compact(self) -> bool:
        """Send a compact command to the running Pi process.

        Returns True if the command was sent successfully.
        """
        if not self._process or not self._process.stdin or self._process.stdin.is_closing():
            return False
        try:
            await self._send({"type": "compact"})
            return True
        except Exception:
            return False
