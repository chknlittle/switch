#!/usr/bin/env python3
"""Claude Code CLI runner for XMPP bridge."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("claude")


class ClaudeRunner:
    """Runs Claude Code and parses stream-json output."""

    def __init__(
        self,
        working_dir: str,
        output_dir: Path,
        session_name: str | None = None,
    ):
        self.working_dir = working_dir
        self.process: asyncio.subprocess.Process | None = None
        self.session_name = session_name
        self.output_file: Path | None = None
        if session_name:
            output_dir.mkdir(exist_ok=True)
            self.output_file = output_dir / f"{session_name}.log"

    async def run(
        self,
        prompt: str,
        session_id: str | None = None,
        system_prompt_file: str | None = None,
    ):
        """Run Claude, yielding (event_type, content) tuples."""
        cmd = [
            "claude",
            "-p",
            prompt,
            "--model",
            "opus",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        if system_prompt_file and Path(system_prompt_file).exists():
            cmd.extend(["--system-prompt", system_prompt_file])

        log.info(f"Claude: {prompt[:50]}...")

        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Prompt: {prompt}\n")

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.working_dir,
                limit=10 * 1024 * 1024,
            )

            tool_count = 0

            stdout = self.process.stdout
            if stdout is None:
                raise RuntimeError("Claude process stdout missing")

            async for line in stdout:
                line = line.decode().strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "system" and event.get("subtype") == "init":
                    sid = event.get("session_id")
                    if sid:
                        yield ("session_id", sid)

                elif event_type == "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text":
                            text = block.get("text", "").strip()
                            if text:
                                yield ("text", text)
                                if self.output_file:
                                    with open(self.output_file, "a") as f:
                                        f.write(f"\n[TEXT]\n{text}\n")
                        elif block.get("type") == "tool_use":
                            tool_count += 1
                            name = block.get("name", "?")
                            if name == "Bash":
                                preview = block.get("input", {}).get("command", "")[:40]
                                tool_desc = f"[{name}: {preview}]"
                            elif name in ("Read", "Write", "Edit"):
                                path = block.get("input", {}).get("file_path", "")
                                tool_desc = f"[{name}: {Path(path).name}]"
                            else:
                                tool_desc = f"[{name}]"
                            yield ("tool", tool_desc)
                            if self.output_file:
                                with open(self.output_file, "a") as f:
                                    f.write(f"{tool_desc}\n")

                elif event_type == "result":
                    is_error = event.get("is_error", False)
                    if is_error:
                        yield ("error", event.get("result", "Unknown error"))
                    else:
                        cost = event.get("total_cost_usd", 0)
                        turns = event.get("num_turns", 0)
                        duration = event.get("duration_ms", 0) / 1000

                        usage = event.get("usage", {})
                        total_tokens = (
                            usage.get("input_tokens", 0)
                            + usage.get("cache_creation_input_tokens", 0)
                            + usage.get("cache_read_input_tokens", 0)
                            + usage.get("output_tokens", 0)
                        )

                        context_window = 200000
                        model_usage = event.get("modelUsage", {})
                        for model_name, model_data in model_usage.items():
                            if "opus" in model_name:
                                context_window = model_data.get("contextWindow", 200000)
                                break
                            context_window = model_data.get(
                                "contextWindow", context_window
                            )

                        tokens_k = total_tokens / 1000
                        context_k = context_window / 1000
                        token_str = f"{tokens_k:.1f}k/{context_k:.0f}k"
                        yield (
                            "result",
                            f"[{turns}t {tool_count}tools ${cost:.3f} {duration:.1f}s | {token_str}]",
                        )

            await self.process.wait()

        except Exception as e:
            log.exception("Claude runner error")
            yield ("error", str(e))

    def cancel(self):
        if self.process:
            self.process.terminate()
