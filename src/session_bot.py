#!/usr/bin/env python3
"""Session bot - one XMPP bot per OpenCode/Claude session."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, cast

from src.claude_runner import ClaudeRunner
from src.helpers import (
    add_roster_subscription,
    append_to_history,
    create_tmux_session,
    log_activity,
    register_unique_account,
)
from src.opencode_runner import OpenCodeResult, OpenCodeRunner
from src.ralph import RalphLoop, parse_ralph_command
from src.utils import BaseXMPPBot

if TYPE_CHECKING:
    from src.manager import SessionManager


class SessionBot(BaseXMPPBot):
    """XMPP bot for a single session."""

    def __init__(
        self,
        session_name: str,
        jid: str,
        password: str,
        db: sqlite3.Connection,
        working_dir: str,
        output_dir: Path,
        xmpp_recipient: str,
        xmpp_domain: str,
        xmpp_server: str,
        ejabberd_ctl: str,
        manager: "SessionManager | None" = None,
    ):
        super().__init__(jid, password, recipient=xmpp_recipient)
        self.session_name = session_name
        self.db = db
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.xmpp_recipient = xmpp_recipient
        self.xmpp_domain = xmpp_domain
        self.xmpp_server = xmpp_server
        self.ejabberd_ctl = ejabberd_ctl
        self.manager = manager
        self.runner: OpenCodeRunner | ClaudeRunner | None = None
        self.processing = False
        self.ralph_loop: RalphLoop | None = None
        self.log = logging.getLogger(f"session.{session_name}")

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

    async def on_start(self, event):
        try:
            self.send_presence()
            await self.get_roster()
            await self["xep_0280"].enable()  # type: ignore[attr-defined]
            self.log.info("Connected")
            self._connected = True
        except Exception:
            self.log.exception("Session start error")
            self._connected = False

    def is_connected(self) -> bool:
        return getattr(self, "_connected", False)

    def on_disconnected(self, event):
        self.log.warning("Disconnected, reconnecting...")
        asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        await asyncio.sleep(5)
        self.connect()

    def send_reply(
        self,
        text: str,
        recipient: str | None = None,
        max_len: int = 3500,
    ):
        """Send message to user, splitting into multiple messages if needed."""
        target = recipient or self.xmpp_recipient
        if len(text) <= max_len:
            msg = self.make_message(mto=target, mbody=text, mtype="chat")
            msg["chat_state"] = "active"
            msg.send()
            return

        parts = []
        current = ""
        for para in text.split("\n\n"):
            if len(current) + len(para) + 2 <= max_len:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    parts.append(current)
                if len(para) > max_len:
                    lines = para.split("\n")
                    for line in lines:
                        if len(current) + len(line) + 1 <= max_len:
                            current = current + "\n" + line if current else line
                        else:
                            if current:
                                parts.append(current)
                            while len(line) > max_len:
                                parts.append(line[:max_len])
                                line = line[max_len:]
                            current = line
                else:
                    current = para
        if current:
            parts.append(current)

        total = len(parts)
        for i, part in enumerate(parts, 1):
            if total > 1:
                header = f"[{i}/{total}]\n" if i > 1 else ""
                footer = f"\n[{i}/{total}]" if i < total else ""
                body = header + part + footer
            else:
                body = part
            msg = self.make_message(mto=target, mbody=body, mtype="chat")
            msg["chat_state"] = "active" if i == total else "composing"
            msg.send()

    async def run_shell_command(self, cmd: str):
        """Run a shell command, send output to user, and inform Claude."""
        if not cmd:
            self.send_reply("Usage: !<command> (e.g., !pwd, !ls, !git status)")
            return

        self.log.info(f"Shell command: {cmd}")
        self.send_typing()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.working_dir,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace").strip()

            if output:
                display_output = output
                if len(display_output) > 4000:
                    display_output = display_output[:4000] + "\n... (truncated)"
                self.send_reply(f"$ {cmd}\n{display_output}")
            else:
                output = "(no output)"
                self.send_reply(f"$ {cmd}\n{output}")

            context_msg = (
                f"[I ran a shell command: `{cmd}`]\n\nOutput:\n```\n"
                f"{output[:8000]}\n```\n"
                "\n(Just acknowledge briefly - I may ask about this next.)"
            )
            await self.process_message(context_msg)

        except asyncio.TimeoutError:
            self.send_reply(f"$ {cmd}\n(timed out after 30s)")
        except Exception as e:
            self.send_reply(f"$ {cmd}\nError: {e}")

    async def peek_output(self, num_lines: int = 30):
        """Show recent output without adding to context."""
        output_file = self.output_dir / f"{self.session_name}.log"
        if not output_file.exists():
            self.send_reply("No output captured yet.")
            return

        try:
            with open(output_file, "rb") as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                if file_size == 0:
                    self.send_reply("Output file empty.")
                    return

                buffer = b""
                chunk_size = 4096
                line_target = max(num_lines, 100)
                while len(buffer.splitlines()) <= line_target and f.tell() > 0:
                    read_size = min(chunk_size, f.tell())
                    f.seek(-read_size, os.SEEK_CUR)
                    buffer = f.read(read_size) + buffer
                    f.seek(-read_size, os.SEEK_CUR)

                lines = buffer.splitlines()
                if lines and f.tell() > 0:
                    lines = lines[1:]
                if not lines:
                    self.send_reply("Output file empty.")
                    return

            effective_lines = max(num_lines, 100)
            recent = [
                line.decode("utf-8", errors="replace")
                for line in lines[-effective_lines:]
            ]
            status = "RUNNING" if self.processing else "IDLE"
            header = f"[{status}] Last {len(recent)} lines:\n"
            output = header + "\n".join(recent)

            if len(output) > 3500:
                output = output[:3500] + "\n... (truncated)"

            self.send_reply(output)
        except Exception as e:
            self.send_reply(f"Error reading output: {e}")

    async def on_message(self, msg):
        try:
            await self._handle_session_message(msg)
        except Exception:
            self.log.exception("Session message error")
            self.send_reply("Error handling message")

    async def _handle_session_message(self, msg):
        if msg["type"] not in ("chat", "normal"):
            return
        if not msg["body"]:
            return

        sender = str(msg["from"].bare)
        dispatcher_jid = f"oc@{self.xmpp_domain}"
        if sender != self.xmpp_recipient and sender != dispatcher_jid:
            return

        body = msg["body"].strip()
        is_scheduled = sender == dispatcher_jid

        if body.startswith("@"):
            body = "/" + body[1:]

        self.log.info(f"Message{'[scheduled]' if is_scheduled else ''}: {body[:50]}...")

        if not is_scheduled and body.strip().lower() == "/kill":
            self.send_reply("Ending session. Goodbye!")
            asyncio.ensure_future(self._self_destruct())
            return

        if not is_scheduled and body.strip().lower() == "/cancel":
            if self.ralph_loop:
                self.ralph_loop.cancel()
                if self.runner:
                    self.runner.cancel()
                self.send_reply("Cancelling Ralph loop...")
            elif self.runner and self.processing:
                self.runner.cancel()
                self.send_reply("Cancelling current run...")
            else:
                self.send_reply("Nothing running to cancel.")
            return

        if not is_scheduled and body.strip().lower().startswith("/peek"):
            parts = body.strip().split()
            num_lines = 30
            if len(parts) > 1:
                try:
                    num_lines = int(parts[1])
                except ValueError:
                    pass
            await self.peek_output(num_lines)
            return

        if not is_scheduled and body.strip().lower().startswith("/agent"):
            parts = body.strip().split()
            if len(parts) < 2:
                self.send_reply("Usage: /agent oc|cc")
                return
            choice = parts[1].lower()
            engine = None
            if choice in ("oc", "opencode"):
                engine = "opencode"
            elif choice in ("cc", "claude"):
                engine = "claude"
            if not engine:
                self.send_reply("Usage: /agent oc|cc")
                return
            self.db.execute(
                "UPDATE sessions SET active_engine = ? WHERE name = ?",
                (engine, self.session_name),
            )
            self.db.commit()
            self.send_reply(f"Active engine set to {engine}.")
            return

        if not is_scheduled and body.strip().lower().startswith("/thinking"):
            parts = body.strip().split()
            if len(parts) < 2:
                self.send_reply("Usage: /thinking normal|high")
                return
            mode = parts[1].lower()
            if mode not in ("normal", "high"):
                self.send_reply("Usage: /thinking normal|high")
                return
            row = self.db.execute(
                "SELECT active_engine FROM sessions WHERE name = ?",
                (self.session_name,),
            ).fetchone()
            active_engine = row["active_engine"] if row else "opencode"
            if active_engine != "opencode":
                self.send_reply("/thinking only applies to OpenCode sessions.")
                return
            self.db.execute(
                "UPDATE sessions SET reasoning_mode = ? WHERE name = ?",
                (mode, self.session_name),
            )
            self.db.commit()
            self.send_reply(f"Reasoning mode set to {mode}.")
            return

        if not is_scheduled and body.strip().lower().startswith("/model"):
            parts = body.strip().split(maxsplit=1)
            if len(parts) < 2:
                self.send_reply("Usage: /model <model-id>")
                return
            model_id = parts[1].strip()
            if not model_id:
                self.send_reply("Usage: /model <model-id>")
                return
            self.db.execute(
                "UPDATE sessions SET model_id = ? WHERE name = ?",
                (model_id, self.session_name),
            )
            self.db.commit()
            self.send_reply(f"Model set to {model_id}.")
            return

        if not is_scheduled and body.strip().lower() == "/reset":
            row = self.db.execute(
                "SELECT active_engine FROM sessions WHERE name = ?",
                (self.session_name,),
            ).fetchone()
            active_engine = row["active_engine"] if row else "opencode"
            if active_engine == "claude":
                self.db.execute(
                    "UPDATE sessions SET claude_session_id = NULL WHERE name = ?",
                    (self.session_name,),
                )
            else:
                self.db.execute(
                    "UPDATE sessions SET opencode_session_id = NULL WHERE name = ?",
                    (self.session_name,),
                )
            self.db.commit()
            self.send_reply("Session reset.")
            return

        if not is_scheduled and body.strip().lower() in (
            "/ralph-cancel",
            "/ralph-stop",
        ):
            if self.ralph_loop:
                self.ralph_loop.cancel()
                self.send_reply("Ralph loop will stop after current iteration...")
            else:
                self.send_reply("No Ralph loop running.")
            return

        if not is_scheduled and body.strip().lower() == "/ralph-status":
            if self.ralph_loop:
                rl = self.ralph_loop
                max_str = (
                    str(rl.max_iterations) if rl.max_iterations > 0 else "unlimited"
                )
                self.send_reply(
                    f"Ralph RUNNING\n"
                    f"Iteration: {rl.current_iteration}/{max_str}\n"
                    f"Cost so far: ${rl.total_cost:.3f}\n"
                    f"Promise: {rl.completion_promise or 'none'}"
                )
            else:
                row = self.db.execute(
                    """
                    SELECT * FROM ralph_loops
                    WHERE session_name = ?
                    ORDER BY started_at DESC LIMIT 1
                """,
                    (self.session_name,),
                ).fetchone()
                if row:
                    self.send_reply(
                        f"Last Ralph: {row['status']}\n"
                        f"Iterations: {row['current_iteration']}/{row['max_iterations'] or 'unlimited'}\n"
                        f"Cost: ${row['total_cost']:.3f}"
                    )
                else:
                    self.send_reply("No Ralph loops in this session.")
            return

        if not is_scheduled and body.strip().lower().startswith("/ralph"):
            ralph_args = parse_ralph_command(body)
            if ralph_args is None:
                self.send_reply(
                    "Usage: /ralph <prompt> [--max N] [--done 'promise']\n"
                    "  or:  /ralph <N> <prompt>  (shorthand)\n\n"
                    "Examples:\n"
                    "  /ralph 20 Fix all type errors\n"
                    "  /ralph Refactor auth --max 10 --done 'All tests pass'\n\n"
                    "Commands:\n"
                    "  /ralph-status - check progress\n"
                    "  /ralph-cancel - stop loop"
                )
                return

            if self.processing:
                self.send_reply("Already running. Use /ralph-cancel first.")
                return

            self.ralph_loop = RalphLoop(
                self,
                ralph_args["prompt"],
                self.working_dir,
                self.output_dir,
                max_iterations=ralph_args["max_iterations"],
                completion_promise=ralph_args["completion_promise"],
                db=self.db,
            )
            self.processing = True
            asyncio.ensure_future(cast(Awaitable[Any], self._run_ralph()))
            return

        if body.startswith("!"):
            await self.run_shell_command(body[1:].strip())
            return

        if self.processing:
            if is_scheduled:
                return

            if body.startswith("+") and self.manager:
                sibling_msg = body[1:].strip()
                if sibling_msg:
                    await self.spawn_sibling_session(sibling_msg)
                    return

            self.send_reply(
                "Still processing... (use +message to spawn sibling session)"
            )
            return

        await self.process_message(body)

    async def _self_destruct(self):
        await asyncio.sleep(1)
        self.disconnect()

    async def _run_ralph(self):
        if not self.ralph_loop:
            return
        loop = self.ralph_loop
        try:
            await loop.run()
        except Exception as e:
            self.log.exception("Ralph loop error")
            self.send_reply(f"Ralph crashed: {e}")
        finally:
            self.ralph_loop = None
            self.processing = False

    async def spawn_sibling_session(self, first_message: str):
        """Spawn an independent sibling session while this one is busy."""
        if not self.manager:
            self.send_reply("Session manager unavailable.")
            return
        self.send_reply("Spawning sibling session...")

        base_name = f"{self.session_name}-sib"
        account = register_unique_account(
            base_name, self.db, self.ejabberd_ctl, self.xmpp_domain, self.log
        )
        if not account:
            self.send_reply("Failed to create sibling session")
            return

        name, password, jid = account

        recipient_user = self.xmpp_recipient.split("@")[0]
        add_roster_subscription(
            name, self.xmpp_recipient, "Clients", self.ejabberd_ctl, self.xmpp_domain
        )
        add_roster_subscription(
            recipient_user, jid, "Sessions", self.ejabberd_ctl, self.xmpp_domain
        )

        create_tmux_session(name, self.working_dir)

        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO sessions
               (name, xmpp_jid, xmpp_password, tmux_name, created_at, last_active, model_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, jid, password, name, now, now, "openai/gpt-5.2-codex"),
        )
        self.db.commit()

        bot = await self.manager.start_session_bot(name, jid, password)
        if bot:
            for _ in range(50):
                if bot.is_connected():
                    break
                await asyncio.sleep(0.1)

            bot.send_reply(
                f"Sibling session '{name}' (spawned from {self.session_name})"
            )
            await bot.process_message(first_message)
        else:
            self.send_reply("Failed to start sibling session")

    async def process_message(self, body: str):
        """Send message to OpenCode or Claude and relay response."""
        self.processing = True
        self.send_typing()

        try:
            row = self.db.execute(
                "SELECT * FROM sessions WHERE name = ?",
                (self.session_name,),
            ).fetchone()
            claude_session_id = row["claude_session_id"] if row else None
            opencode_session_id = row["opencode_session_id"] if row else None
            active_engine = row["active_engine"] if row else "opencode"
            opencode_agent = row["opencode_agent"] if row else "bridge"
            model_id = row["model_id"] if row else "glm_gguf/glm-4.7-flash-q8"

            self.db.execute(
                "UPDATE sessions SET last_active = ? WHERE name = ?",
                (datetime.now().isoformat(), self.session_name),
            )
            self.db.commit()

            append_to_history(body, self.working_dir, claude_session_id)
            log_activity(body, session=self.session_name, source="xmpp")

            self.db.execute(
                """INSERT INTO session_messages (session_name, role, content, engine, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    self.session_name,
                    "user",
                    body,
                    active_engine,
                    datetime.now().isoformat(),
                ),
            )
            self.db.commit()

            response_parts: list[str] = []
            tool_summaries: list[str] = []
            last_progress_at = 0

            if active_engine == "claude":
                self.runner = ClaudeRunner(
                    self.working_dir,
                    self.output_dir,
                    session_name=self.session_name,
                )
                async for event_type, content in self.runner.run(
                    body, claude_session_id
                ):
                    if event_type == "session_id" and self.db:
                        self.db.execute(
                            "UPDATE sessions SET claude_session_id = ? WHERE name = ?",
                            (content, self.session_name),
                        )
                        self.db.commit()
                    elif event_type == "text":
                        response_parts = [content]
                    elif event_type == "tool":
                        tool_summaries.append(content)
                        tool_count = len(tool_summaries)
                        if tool_count - last_progress_at >= 8:
                            last_progress_at = tool_count
                            recent = tool_summaries[-3:]
                            self.send_reply(f"... {' '.join(recent)}")
                    elif event_type == "result":
                        parts = []
                        if tool_summaries:
                            tools = " ".join(tool_summaries[:5])
                            if len(tool_summaries) > 5:
                                tools += f" +{len(tool_summaries) - 5}"
                            parts.append(tools)
                        if response_parts:
                            parts.append(response_parts[-1])
                        parts.append(content)
                        response = "\n\n".join(parts)
                        self.send_reply(response)
                        self.db.execute(
                            """INSERT INTO session_messages (session_name, role, content, engine, created_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                self.session_name,
                                "assistant",
                                response_parts[-1] if response_parts else "",
                                active_engine,
                                datetime.now().isoformat(),
                            ),
                        )
                        self.db.commit()
                    elif event_type == "error":
                        self.send_reply(f"Error: {content}")
                    elif event_type == "cancelled":
                        self.send_reply("Cancelled.")

            else:
                self.runner = OpenCodeRunner(
                    self.working_dir,
                    self.output_dir,
                    session_name=self.session_name,
                    model=model_id,
                    reasoning_mode=row["reasoning_mode"] if row else "normal",
                    agent=opencode_agent,
                )
                accumulated = ""
                oc_runner = self.runner
                async for event_type, content in oc_runner.run(
                    body, opencode_session_id
                ):
                    if event_type == "session_id" and self.db:
                        self.db.execute(
                            "UPDATE sessions SET opencode_session_id = ? WHERE name = ?",
                            (content, self.session_name),
                        )
                        self.db.commit()
                    elif event_type == "text":
                        if isinstance(content, str):
                            accumulated += content
                            response_parts = [accumulated]
                    elif event_type == "tool":
                        if isinstance(content, str):
                            tool_summaries.append(content)
                        tool_count = len(tool_summaries)
                        if tool_count - last_progress_at >= 8:
                            last_progress_at = tool_count
                            recent = tool_summaries[-3:]
                            self.send_reply(f"... {' '.join(recent)}")
                    elif event_type == "result":
                        if not isinstance(content, OpenCodeResult):
                            continue
                        result = content
                        parts = []
                        if tool_summaries:
                            tools = " ".join(tool_summaries[:5])
                            if len(tool_summaries) > 5:
                                tools += f" +{len(tool_summaries) - 5}"
                            parts.append(tools)
                        if response_parts:
                            parts.append(response_parts[-1])
                        stats = (
                            f"[{result.tokens_in}/{result.tokens_out} tok"
                            f" r{result.tokens_reasoning} c{result.tokens_cache_read}/{result.tokens_cache_write}"
                            f" ${result.cost:.3f} {result.duration_s:.1f}s]"
                        )
                        parts.append(stats)
                        response = "\n\n".join(parts)
                        self.send_reply(response)
                        self.db.execute(
                            """INSERT INTO session_messages (session_name, role, content, engine, created_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                self.session_name,
                                "assistant",
                                response_parts[-1] if response_parts else "",
                                active_engine,
                                datetime.now().isoformat(),
                            ),
                        )
                        self.db.commit()
                    elif event_type == "error":
                        self.send_reply(f"Error: {content}")
                    elif event_type == "cancelled":
                        self.send_reply("Cancelled.")

        except Exception as e:
            self.log.exception("Error")
            self.send_reply(f"Error: {e}")

        finally:
            self.processing = False
