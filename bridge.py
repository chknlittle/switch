#!/usr/bin/env python3
"""
XMPP-OpenCode Bridge (Multi-Account)

Each OpenCode/Claude session gets its own XMPP account, appearing as a separate
chat contact in the client.

- Send to oc@... to create a new session (auto-named from message)
- Each session appears as its own contact (e.g., react-app@...)
- Reply directly to that contact to continue the conversation
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import secrets
import shlex
import signal
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

from utils import (
    BaseXMPPBot,
    get_xmpp_config,
    load_env,
    run_ejabberdctl as _run_ejabberdctl,
)

# Load environment
load_env()

# Configuration
_cfg = get_xmpp_config()
XMPP_SERVER = _cfg["server"]
XMPP_DOMAIN = _cfg["domain"]
XMPP_DISPATCHER_JID = _cfg["dispatcher_jid"]
XMPP_DISPATCHER_PASSWORD = _cfg["dispatcher_password"]
XMPP_RECIPIENT = _cfg["recipient"]
EJABBERD_CTL = _cfg["ejabberd_ctl"]
WORKING_DIR = os.getenv("CLAUDE_WORKING_DIR", str(Path.home()))
OPENCODE_WORKING_DIR = os.getenv("OPENCODE_WORKING_DIR", str(Path(__file__).parent))
DB_PATH = Path(__file__).parent / "sessions.db"
HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"
ACTIVITY_LOG_PATH = Path.home() / ".claude" / "activity.jsonl"
SESSION_OUTPUT_DIR = Path(__file__).parent / "output"  # Live output capture

# Calendar integration
CALENDAR_DIR = Path.home() / "calendar"  # type: ignore[reportMissingImports]
sys.path.insert(0, str(CALENDAR_DIR))
try:
    cal_store = importlib.import_module("cal_store")  # type: ignore[reportMissingImports]
    add_event = cal_store.add_event
    remove_event = cal_store.remove_event
    list_events = cal_store.list_events
    format_event_for_display = cal_store.format_event_for_display
    CALENDAR_AVAILABLE = True
except Exception:
    CALENDAR_AVAILABLE = False
    add_event = None
    remove_event = None
    list_events = None
    format_event_for_display = None

# Telegram integration
TELEGRAM_DIR = Path.home() / "telegram"  # type: ignore[reportMissingImports]
sys.path.insert(0, str(TELEGRAM_DIR))
try:
    tg = importlib.import_module("tg")  # type: ignore[reportMissingImports]
    tg_send = tg.send_message
    tg_history = tg.fetch_history
    tg_poll = tg.poll_and_save
    tg_typing = tg.send_typing
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False
    tg_send = None
    tg_history = None
    tg_poll = None
    tg_typing = None

# Telegram engagement (for bidirectional mirroring)
TELEGRAM_PENDING_DIR = Path(__file__).parent / "telegram_pending"
TELEGRAM_ENGAGEMENT_FILE = TELEGRAM_DIR / "engagement.json"
TELEGRAM_CANCEL_DIR = TELEGRAM_PENDING_DIR / "cancel"


def get_telegram_engagement() -> dict:
    """Get current Telegram engagement state."""
    if TELEGRAM_ENGAGEMENT_FILE.exists():
        try:
            return json.loads(TELEGRAM_ENGAGEMENT_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {"session": None, "chat_id": None}


def check_cancel_signal(session_name: str) -> bool:
    """Check if a cancel signal exists for this session and clear it."""
    TELEGRAM_CANCEL_DIR.mkdir(parents=True, exist_ok=True)
    cancel_file = TELEGRAM_CANCEL_DIR / f"{session_name}.cancel"
    if cancel_file.exists():
        cancel_file.unlink()
        return True
    return False


async def send_to_telegram(text: str, chat_id: int | None = None) -> bool:
    """Send message to Telegram (async wrapper)."""
    log.debug(f"send_to_telegram called: chat_id={chat_id}, text={text[:50]}...")
    if not TELEGRAM_AVAILABLE:
        log.warning("Telegram not available")
        return False
    try:
        result = await tg_send(text, chat_id)  # type: ignore[misc]
        success = any(r.get("ok") for r in result.values())
        log.info(f"Telegram send result: success={success}")
        return success
    except Exception as e:
        log.error(f"Failed to send to Telegram: {e}")
        return False


async def send_telegram_typing(chat_id: int | None = None) -> bool:
    """Send typing indicator to Telegram."""
    if not TELEGRAM_AVAILABLE:
        return False
    try:
        return await tg_typing(chat_id)  # type: ignore[misc]
    except Exception as e:
        log.error(f"Failed to send typing to Telegram: {e}")
        return False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")


# =============================================================================
# Database
# =============================================================================


def init_db() -> sqlite3.Connection:
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            name TEXT PRIMARY KEY,
            xmpp_jid TEXT UNIQUE NOT NULL,
            xmpp_password TEXT NOT NULL,
            claude_session_id TEXT,
            opencode_session_id TEXT,
            active_engine TEXT DEFAULT 'opencode',
            model_id TEXT DEFAULT 'openai/gpt-5.2-codex',
            reasoning_mode TEXT DEFAULT 'normal',
            tmux_name TEXT,
            created_at TEXT NOT NULL,
            last_active TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ralph_loops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            completion_promise TEXT,
            max_iterations INTEGER DEFAULT 0,
            current_iteration INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0,
            status TEXT DEFAULT 'running',
            started_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY (session_name) REFERENCES sessions(name)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            engine TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_name) REFERENCES sessions(name)
        )
    """
    )
    # Migration: add columns for existing databases
    for column in [
        ("opencode_session_id", "TEXT"),
        ("active_engine", "TEXT DEFAULT 'opencode'"),
        ("model_id", "TEXT DEFAULT 'openai/gpt-5.2-codex'"),
        ("reasoning_mode", "TEXT DEFAULT 'normal'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {column[0]} {column[1]}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


# =============================================================================
# Helpers
# =============================================================================


def slugify(text: str, max_len: int = 20) -> str:
    """Convert text to a safe session/username."""
    words = text.lower().split()[:4]
    slug = "-".join(words)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = slug[:max_len].rstrip("-")
    return slug or f"session-{secrets.token_hex(4)}"


def append_to_history(message: str, project: str, session_id: str | None = None):
    """Append a message to Claude's history.jsonl for session tracking."""
    try:
        entry = {
            "display": message,
            "pastedContents": {},
            "timestamp": int(datetime.now().timestamp() * 1000),
            "project": project,
            "sessionId": session_id or "xmpp-bridge",
        }
        with open(HISTORY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Don't break message flow if history write fails


def log_activity(message: str, session: str | None = None, source: str = "xmpp"):
    """
    Log activity to unified activity log.

    Args:
        message: The user message or activity description
        session: Session name (e.g., 'lets-get-back-to' or 'laptop')
        source: Where this came from ('xmpp', 'laptop', 'claude-response')
    """
    try:
        entry = {
            "ts": datetime.now().isoformat(),
            "source": source,
            "session": session,
            "message": message[:500],
        }
        with open(ACTIVITY_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def run_ejabberdctl(*args) -> tuple[bool, str]:
    """Run an ejabberdctl command via SSH or locally."""
    return _run_ejabberdctl(EJABBERD_CTL, *args)


def create_xmpp_account(username: str, password: str) -> tuple[bool, str]:
    """Create a new ejabberd account."""
    success, output = run_ejabberdctl("register", username, XMPP_DOMAIN, password)
    if success:
        log.info(f"Created XMPP account: {username}@{XMPP_DOMAIN}")
    else:
        log.error(f"Failed to create account {username}: {output}")
    return success, output


def register_unique_account(
    base_name: str,
    db: sqlite3.Connection,
    max_attempts: int = 5,
) -> tuple[str, str, str] | None:
    """Register a unique XMPP account, retrying on conflicts."""
    for idx in range(max_attempts):
        suffix = "" if idx == 0 else f"-{idx + 1}"
        trim_len = max(1, 20 - len(suffix))
        candidate = base_name[:trim_len].rstrip("-") + suffix

        exists = db.execute(
            "SELECT 1 FROM sessions WHERE name = ?", (candidate,)
        ).fetchone()
        if exists:
            continue

        password = secrets.token_urlsafe(16)
        success, output = create_xmpp_account(candidate, password)
        if success:
            return candidate, password, f"{candidate}@{XMPP_DOMAIN}"
        if "conflict" in output.lower():
            continue
        break

    return None


def add_roster_subscription(username: str, contact_jid: str, group: str) -> None:
    """Add mutual roster subscription between user and contact."""
    contact_user = contact_jid.split("@")[0]
    run_ejabberdctl(
        "add_rosteritem",
        username,
        XMPP_DOMAIN,
        contact_user,
        XMPP_DOMAIN,
        contact_user,
        group,
        "both",
    )


def delete_xmpp_account(username: str) -> bool:
    """Delete an ejabberd account."""
    success, output = run_ejabberdctl("unregister", username, XMPP_DOMAIN)
    if success:
        log.info(f"Deleted XMPP account: {username}@{XMPP_DOMAIN}")
    return success


def tmux_session_exists(name: str) -> bool:
    """Check if a tmux session exists."""
    result = subprocess.run(["tmux", "has-session", "-t", name], capture_output=True)
    return result.returncode == 0


def create_tmux_session(name: str) -> bool:
    """Create a new tmux session with interactive shell."""
    if tmux_session_exists(name):
        return True
    script_path = Path(__file__).parent / "session-shell.sh"
    result = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            name,
            "-c",
            WORKING_DIR,
            str(script_path),
            name,
        ],
        capture_output=True,
    )
    return result.returncode == 0


def is_session_locked(name: str) -> bool:
    """Check if session is being used interactively from terminal."""
    lock_file = Path(__file__).parent / "locks" / f"{name}.lock"
    return lock_file.exists()


def kill_tmux_session(name: str) -> bool:
    """Kill a tmux session."""
    result = subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)
    return result.returncode == 0


# =============================================================================
# OpenCode Runner
# =============================================================================


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
        working_dir: str = OPENCODE_WORKING_DIR,
        session_name: str | None = None,
        model: str | None = None,
        reasoning_mode: str = "normal",
    ):
        self.working_dir = working_dir
        self.process: asyncio.subprocess.Process | None = None
        self.session_name = session_name
        self.output_file: Path | None = None
        self.model = model
        self.reasoning_mode = reasoning_mode
        if session_name:
            SESSION_OUTPUT_DIR.mkdir(exist_ok=True)
            self.output_file = SESSION_OUTPUT_DIR / f"{session_name}.log"

    async def run(self, prompt: str, session_id: str | None = None):
        """Run OpenCode, yielding (event_type, content) tuples."""
        cmd = ["opencode", "run", "--format", "json", "--agent", "bridge"]
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
                f.write(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Prompt: {prompt[:100]}...\n"
                )

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
                if self.session_name and check_cancel_signal(self.session_name):
                    log.info(f"Cancel signal detected for {self.session_name}")
                    self.cancel()
                    yield ("cancelled", "Cancelled via signal")
                    return

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


# =============================================================================
# Claude Runner
# =============================================================================


class ClaudeRunner:
    """Runs Claude Code and parses stream-json output."""

    def __init__(self, working_dir: str = WORKING_DIR, session_name: str | None = None):
        self.working_dir = working_dir
        self.process: asyncio.subprocess.Process | None = None
        self.session_name = session_name
        self.output_file: Path | None = None
        if session_name:
            SESSION_OUTPUT_DIR.mkdir(exist_ok=True)
            self.output_file = SESSION_OUTPUT_DIR / f"{session_name}.log"

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
                f.write(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Prompt: {prompt[:100]}...\n"
                )

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
                if self.session_name and check_cancel_signal(self.session_name):
                    log.info(f"Cancel signal detected for {self.session_name}")
                    self.cancel()
                    yield ("cancelled", "Cancelled via signal")
                    return

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


# =============================================================================
# Ralph Loop (Autonomous iteration loop)
# =============================================================================


def parse_ralph_command(body: str) -> dict | None:
    """
    Parse /ralph command into components.

    Formats supported:
      /ralph <prompt> --max <N> --done "<promise>"
      /ralph <N> <prompt>  (shorthand: first number is max iterations)
      /ralph <prompt>  (infinite loop - dangerous!)

    Returns dict with: prompt, max_iterations, completion_promise
    Or None if not a ralph command.
    """
    if not body.lower().startswith("/ralph"):
        return None

    rest = body[6:].strip()
    if not rest:
        return None

    max_iterations = 0
    completion_promise = None

    try:
        parts = shlex.split(rest)
    except ValueError:
        parts = rest.split()

    prompt_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in ("--max", "--max-iterations", "-m"):
            if i + 1 < len(parts):
                try:
                    max_iterations = int(parts[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
        elif part in ("--done", "--completion-promise", "-d"):
            if i + 1 < len(parts):
                completion_promise = parts[i + 1]
                i += 2
                continue
        prompt_parts.append(part)
        i += 1

    if prompt_parts and prompt_parts[0].isdigit():
        max_iterations = int(prompt_parts[0])
        prompt_parts = prompt_parts[1:]

    prompt = " ".join(prompt_parts)
    if not prompt:
        return None

    return {
        "prompt": prompt,
        "max_iterations": max_iterations,
        "completion_promise": completion_promise,
    }


class RalphLoop:
    """Manages an autonomous iteration loop."""

    def __init__(
        self,
        session_bot,
        prompt: str,
        max_iterations: int = 0,
        completion_promise: str | None = None,
        db: sqlite3.Connection | None = None,
    ):
        self.session_bot = session_bot
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.completion_promise = completion_promise
        self.db = db
        self.current_iteration = 0
        self.total_cost = 0.0
        self.cancelled = False
        self.loop_id: int | None = None
        self.log = logging.getLogger(f"ralph.{session_bot.session_name}")

    def _save_state(self, status: str = "running"):
        """Save loop state to database."""
        if not self.db or not self.loop_id:
            return
        self.db.execute(
            """
            UPDATE ralph_loops
            SET current_iteration = ?, total_cost = ?, status = ?,
                finished_at = CASE WHEN ? != 'running' THEN ? ELSE NULL END
            WHERE id = ?
        """,
            (
                self.current_iteration,
                self.total_cost,
                status,
                status,
                datetime.now().isoformat(),
                self.loop_id,
            ),
        )
        self.db.commit()

    def cancel(self):
        """Signal the loop to stop after current iteration."""
        self.cancelled = True
        self._save_state("cancelled")

    async def run(self):
        """Run the autonomous loop."""
        if self.db:
            cursor = self.db.execute(
                """
                INSERT INTO ralph_loops
                (session_name, prompt, completion_promise, max_iterations, started_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    self.session_bot.session_name,
                    self.prompt,
                    self.completion_promise,
                    self.max_iterations,
                    datetime.now().isoformat(),
                ),
            )
            self.loop_id = cursor.lastrowid
            self.db.commit()

        max_str = str(self.max_iterations) if self.max_iterations > 0 else "unlimited"
        promise_str = (
            f'"{self.completion_promise}"' if self.completion_promise else "none"
        )
        self.session_bot.send_reply(
            f"Ralph loop started\n"
            f"Max: {max_str} | Done when: {promise_str}\n"
            f"Use /ralph-cancel to stop"
        )

        while True:
            if self.cancelled:
                self.session_bot.send_reply(
                    f"Ralph cancelled at iteration {self.current_iteration}\n"
                    f"Total cost: ${self.total_cost:.3f}"
                )
                break

            if (
                self.max_iterations > 0
                and self.current_iteration >= self.max_iterations
            ):
                self.session_bot.send_reply(
                    f"Ralph complete: hit max iterations ({self.max_iterations})\n"
                    f"Total cost: ${self.total_cost:.3f}"
                )
                self._save_state("max_iterations")
                break

            self.current_iteration += 1
            self._save_state()

            iter_info = f"[Ralph iteration {self.current_iteration}"
            if self.max_iterations > 0:
                iter_info += f"/{self.max_iterations}"
            iter_info += "]"

            full_prompt = f"{iter_info}\n\n{self.prompt}"
            if self.completion_promise:
                full_prompt += (
                    f"\n\nTo signal completion, output EXACTLY: "
                    f"<promise>{self.completion_promise}</promise>\n"
                    f"ONLY output this when the task is genuinely complete."
                )

            self.log.info(f"Ralph iteration {self.current_iteration}")

            runner = ClaudeRunner(
                WORKING_DIR, session_name=self.session_bot.session_name
            )
            self.session_bot.runner = runner

            row = (
                self.db.execute(
                    "SELECT claude_session_id FROM sessions WHERE name = ?",
                    (self.session_bot.session_name,),
                ).fetchone()
                if self.db
                else None
            )
            claude_session_id = row["claude_session_id"] if row else None

            response_text = ""
            iteration_cost = 0.0
            tool_count = 0

            try:
                async for event_type, content in runner.run(
                    full_prompt, claude_session_id
                ):
                    if event_type == "session_id" and self.db:
                        self.db.execute(
                            "UPDATE sessions SET claude_session_id = ? WHERE name = ?",
                            (content, self.session_bot.session_name),
                        )
                        self.db.commit()
                    elif event_type == "text":
                        response_text = content
                    elif event_type == "tool":
                        tool_count += 1
                    elif event_type == "result":
                        cost_match = re.search(r"\$(\d+\.?\d*)", content)
                        if cost_match:
                            iteration_cost = float(cost_match.group(1))
                        self.total_cost += iteration_cost
                    elif event_type == "error":
                        self.session_bot.send_reply(
                            f"Ralph error at iteration {self.current_iteration}: {content}\n"
                            f"Stopping loop. Total cost: ${self.total_cost:.3f}"
                        )
                        self._save_state("error")
                        return
            finally:
                self.session_bot.runner = None

            preview = (
                response_text[:200] + "..."
                if len(response_text) > 200
                else response_text
            )
            self.session_bot.send_reply(
                f"[Ralph {self.current_iteration}"
                f"{'/' + str(self.max_iterations) if self.max_iterations > 0 else ''}"
                f" | {tool_count}tools ${iteration_cost:.3f}]\n\n{preview}"
            )

            if self.completion_promise:
                promise_tag = f"<promise>{self.completion_promise}</promise>"
                if promise_tag in response_text:
                    self.session_bot.send_reply(
                        f"Ralph COMPLETE at iteration {self.current_iteration}\n"
                        f"Detected: {promise_tag}\n"
                        f"Total cost: ${self.total_cost:.3f}"
                    )
                    self._save_state("completed")
                    return

            self._save_state()
            await asyncio.sleep(2)

        self._save_state("finished")


# =============================================================================
# Session Bot (one per OpenCode/Claude session)
# =============================================================================


class SessionBot(BaseXMPPBot):
    """XMPP bot for a single session."""

    def __init__(
        self,
        session_name: str,
        jid: str,
        password: str,
        db: sqlite3.Connection,
        manager: "SessionManager | None" = None,
    ):
        super().__init__(jid, password, recipient=XMPP_RECIPIENT)
        self.session_name = session_name
        self.db = db
        self.manager = manager
        self.runner: OpenCodeRunner | ClaudeRunner | None = None
        self.processing = False
        self.ralph_loop: RalphLoop | None = None
        self.telegram_poll_task: asyncio.Task | None = None
        self.log = logging.getLogger(f"session.{session_name}")

        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

    async def on_start(self, event):
        self.send_presence()
        await self.get_roster()
        await self["xep_0280"].enable()  # type: ignore[attr-defined]
        self.log.info("Connected")
        self._connected = True
        self.telegram_poll_task = asyncio.ensure_future(self._poll_telegram_pending())

    def is_connected(self) -> bool:
        return getattr(self, "_connected", False)

    def is_telegram_engaged(self) -> tuple[bool, int | None]:
        """Check if this session is engaged via Telegram. Returns (engaged, chat_id)."""
        engagement = get_telegram_engagement()
        if engagement.get("session") == self.session_name:
            return True, engagement.get("chat_id")
        return False, None

    async def _poll_telegram_pending(self):
        """Poll for pending messages from Telegram."""
        TELEGRAM_PENDING_DIR.mkdir(exist_ok=True)
        pending_file = TELEGRAM_PENDING_DIR / f"{self.session_name}.jsonl"

        while True:
            try:
                await asyncio.sleep(2)

                if not pending_file.exists():
                    continue

                content = pending_file.read_text().strip()
                if not content:
                    continue

                pending_file.write_text("")

                for line in content.split("\n"):
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                        sender = msg.get("from", "telegram")
                        text = msg.get("text", "")

                        if text:
                            self.log.info(
                                f"Telegram message from {sender}: {text[:50]}..."
                            )

                            text_lower = text.strip().lower()
                            if text_lower in ("/cancel", "/c"):
                                self.send_reply(
                                    f"[TG/{sender}] {text}", mirror_to_telegram=False
                                )
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
                                continue

                            self.send_reply(
                                f"[TG/{sender}] {text}", mirror_to_telegram=False
                            )

                            if not self.processing:
                                await self.process_message(
                                    text, from_telegram=True, telegram_sender=sender
                                )
                            else:
                                self.send_reply(
                                    f"[Busy - message queued from {sender}]",
                                    mirror_to_telegram=False,
                                )

                    except json.JSONDecodeError:
                        continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error(f"Error polling Telegram pending: {e}")
                await asyncio.sleep(5)

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
        mirror_to_telegram: bool = True,
    ):
        """Send message to user, splitting into multiple messages if needed."""
        if mirror_to_telegram:
            engaged, chat_id = self.is_telegram_engaged()
            self.log.debug(
                f"Telegram mirror check: engaged={engaged}, chat_id={chat_id}, session={self.session_name}"
            )
            if engaged and chat_id:
                self.log.info(f"Mirroring to Telegram chat {chat_id}: {text[:50]}...")
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(send_to_telegram(text, chat_id))
                except RuntimeError:
                    self.log.warning(
                        "No running loop for Telegram mirror, using ensure_future"
                    )
                    asyncio.ensure_future(send_to_telegram(text, chat_id))

        target = recipient or XMPP_RECIPIENT
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
                cwd=WORKING_DIR,
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
        output_file = SESSION_OUTPUT_DIR / f"{self.session_name}.log"
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
        if msg["type"] not in ("chat", "normal"):
            return
        if not msg["body"]:
            return

        sender = str(msg["from"].bare)
        dispatcher_jid = f"oc@{XMPP_DOMAIN}"
        if sender != XMPP_RECIPIENT and sender != dispatcher_jid:
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

        if is_session_locked(self.session_name):
            self.send_reply(
                "Session in use from terminal. Attach with: tmux attach -t "
                + self.session_name
            )
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
        account = register_unique_account(base_name, self.db)
        if not account:
            self.send_reply("Failed to create sibling session")
            return

        name, password, jid = account

        recipient_user = XMPP_RECIPIENT.split("@")[0]
        add_roster_subscription(name, XMPP_RECIPIENT, "Clients")
        add_roster_subscription(recipient_user, jid, "Sessions")

        create_tmux_session(name)

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

    async def _telegram_typing_loop(self, chat_id: int):
        """Send typing indicator every 5 seconds until processing is done."""
        while self.processing:
            await send_telegram_typing(chat_id)
            await asyncio.sleep(5)

    async def process_message(
        self,
        body: str,
        from_telegram: bool = False,
        telegram_sender: str | None = None,
    ):
        """Send message to OpenCode or Claude and relay response."""
        self.processing = True
        self.send_typing()

        engaged, tg_chat_id = self.is_telegram_engaged()
        typing_task = None
        if engaged and tg_chat_id:
            typing_task = asyncio.ensure_future(self._telegram_typing_loop(tg_chat_id))

        if not from_telegram:
            if engaged and tg_chat_id:
                asyncio.ensure_future(send_to_telegram(f"[XMPP] {body}", tg_chat_id))

        try:
            row = self.db.execute(
                "SELECT * FROM sessions WHERE name = ?",
                (self.session_name,),
            ).fetchone()
            claude_session_id = row["claude_session_id"] if row else None
            opencode_session_id = row["opencode_session_id"] if row else None
            active_engine = row["active_engine"] if row else "opencode"
            model_id = row["model_id"] if row else "openai/gpt-5.2-codex"

            self.db.execute(
                "UPDATE sessions SET last_active = ? WHERE name = ?",
                (datetime.now().isoformat(), self.session_name),
            )
            self.db.commit()

            append_to_history(body, WORKING_DIR, claude_session_id)
            source = f"telegram/{telegram_sender}" if from_telegram else "xmpp"
            log_activity(body, session=self.session_name, source=source)

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
                self.runner = ClaudeRunner(WORKING_DIR, session_name=self.session_name)
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
                    OPENCODE_WORKING_DIR,
                    session_name=self.session_name,
                    model=model_id,
                    reasoning_mode=row["reasoning_mode"] if row else "normal",
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
            if typing_task:
                typing_task.cancel()


# =============================================================================
# Dispatcher Bot
# =============================================================================


class DispatcherBot(BaseXMPPBot):
    """Dispatcher bot that creates new session bots."""

    def __init__(self, jid: str, password: str, db: sqlite3.Connection, manager=None):
        super().__init__(jid, password)
        self.db = db
        self.manager: SessionManager | None = manager
        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

    async def on_start(self, event):
        self.send_presence()
        await self.get_roster()
        self.log = logging.getLogger("dispatcher")
        self.log.info("Dispatcher connected")

    def on_disconnected(self, event):
        self.log.warning("Dispatcher disconnected, reconnecting...")
        asyncio.ensure_future(self._reconnect())

    async def _reconnect(self):
        await asyncio.sleep(5)
        self.connect()

    async def on_message(self, msg):
        if msg["type"] not in ("chat", "normal"):
            return
        if not msg["body"]:
            return

        sender = str(msg["from"].bare)
        dispatcher_bare = XMPP_DISPATCHER_JID.split("/")[0]
        if sender != XMPP_RECIPIENT and sender != dispatcher_bare:
            return

        body = msg["body"].strip()
        if body.startswith("@"):
            body = "/" + body[1:]

        self.log.info(f"Dispatcher received: {body[:50]}...")

        if body.startswith("/"):
            await self.handle_command(body)
            return

        await self.create_session(body)

    async def handle_command(self, body: str):
        parts = body.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/list":
            rows = self.db.execute(
                "SELECT name, last_active FROM sessions ORDER BY last_active DESC LIMIT 15"
            ).fetchall()
            if rows:
                lines = ["Sessions (message the contact directly to continue):"]
                for row in rows:
                    lines.append(f"  {row['name']}@{XMPP_DOMAIN}")
                self.send_reply("\n".join(lines), recipient=XMPP_RECIPIENT)
            else:
                self.send_reply(
                    "No sessions yet. Send a message to start one!",
                    recipient=XMPP_RECIPIENT,
                )
            return

        if cmd == "/kill":
            if not arg:
                self.send_reply("Usage: /kill <session-name>", recipient=XMPP_RECIPIENT)
                return
            if not self.manager:
                self.send_reply(
                    "Session manager unavailable.", recipient=XMPP_RECIPIENT
                )
                return
            await self.manager.kill_session(arg)
            self.send_reply(f"Killed: {arg}", recipient=XMPP_RECIPIENT)
            return

        if cmd == "/cal":
            if not CALENDAR_AVAILABLE:
                self.send_reply(
                    "Calendar not available. Check ~/calendar/cal_store.py",
                    recipient=XMPP_RECIPIENT,
                )
                return

            parts = arg.split(None, 1) if arg else [""]
            subcmd = parts[0].lower()
            subarg = parts[1] if len(parts) > 1 else ""

            if subcmd == "add":
                add_parts = subarg.split()
                if len(add_parts) < 2:
                    self.send_reply(
                        "Usage: /cal add <title> <YYYY-MM-DD> [HH:MM]",
                        recipient=XMPP_RECIPIENT,
                    )
                    return
                title_parts = []
                date_str = None
                time_str = None
                for i, part in enumerate(add_parts):
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", part):
                        date_str = part
                        title_parts = add_parts[:i]
                        if i + 1 < len(add_parts) and re.match(
                            r"^\d{2}:\d{2}$", add_parts[i + 1]
                        ):
                            time_str = add_parts[i + 1]
                        break
                if not date_str or not title_parts:
                    self.send_reply(
                        "Usage: /cal add <title> <YYYY-MM-DD> [HH:MM]",
                        recipient=XMPP_RECIPIENT,
                    )
                    return
                title = " ".join(title_parts)
                event = cast(Callable[[str, str, str | None], Any], add_event)(
                    title, date_str, time_str
                )
                formatter = cast(Callable[[Any], str], format_event_for_display)
                self.send_reply(
                    f"Added: {formatter(event)}",
                    recipient=XMPP_RECIPIENT,
                )
                return

            if subcmd in ("rm", "remove"):
                if not subarg:
                    self.send_reply(
                        "Usage: /cal rm <event-id>",
                        recipient=XMPP_RECIPIENT,
                    )
                    return
                cast(Callable[[str], Any], remove_event)(subarg.strip())
                self.send_reply("Removed.", recipient=XMPP_RECIPIENT)
                return

            events = cast(Callable[[], list[Any]], list_events)()
            if not events:
                self.send_reply("No upcoming events.", recipient=XMPP_RECIPIENT)
                return
            formatter = cast(Callable[[Any], str], format_event_for_display)
            lines = ["Upcoming:"]
            for event in events[:10]:
                lines.append(f"  {formatter(event)}")
            self.send_reply("\n".join(lines), recipient=XMPP_RECIPIENT)
            return

        if cmd == "/tg":
            if not TELEGRAM_AVAILABLE:
                self.send_reply(
                    "Telegram not available. Check ~/telegram/tg.py",
                    recipient=XMPP_RECIPIENT,
                )
                return

            parts = arg.split(None, 1) if arg else [""]
            subcmd = parts[0].lower()
            subarg = parts[1] if len(parts) > 1 else ""

            if subcmd == "send":
                if not subarg:
                    self.send_reply(
                        "Usage: /tg send <message>", recipient=XMPP_RECIPIENT
                    )
                    return
                results = await cast(Callable[[str], Awaitable[dict]], tg_send)(subarg)
                ok_count = sum(
                    1 for r in results.values() if isinstance(r, dict) and r.get("ok")
                )
                self.send_reply(f"Sent to {ok_count} chat(s)", recipient=XMPP_RECIPIENT)
                return

            if subcmd in ("history", "h"):
                limit = int(subarg) if subarg else 10
                msgs = await cast(Callable[[int], Awaitable[list[dict]]], tg_history)(
                    limit
                )
                if not msgs:
                    self.send_reply(
                        "No messages yet. Send something to the bot first.",
                        recipient=XMPP_RECIPIENT,
                    )
                    return
                lines = []
                for msg in msgs[-limit:]:
                    date = msg.get("date", "")[-8:-3] if msg.get("date") else ""
                    text = msg.get("text", "")[:60]
                    lines.append(f"[{date}] {msg.get('from', '?')}: {text}")
                self.send_reply("\n".join(lines), recipient=XMPP_RECIPIENT)
                return

            if subcmd == "poll":
                count = await cast(Callable[[], Awaitable[int]], tg_poll)()
                self.send_reply(
                    f"Polled {count} new message(s)", recipient=XMPP_RECIPIENT
                )
                return

            self.send_reply(
                "Telegram commands:\n"
                "  /tg send <msg> - send to all chats\n"
                "  /tg history [n] - show last n messages\n"
                "  /tg poll - fetch new messages",
                recipient=XMPP_RECIPIENT,
            )
            return

        if cmd == "/recent":
            rows = self.db.execute(
                """SELECT name, status, last_active, created_at
                   FROM sessions ORDER BY last_active DESC LIMIT 10"""
            ).fetchall()
            if rows:
                lines = ["Recent sessions:"]
                for row in rows:
                    status = row["status"] or "active"
                    last = row["last_active"][5:16] if row["last_active"] else "?"
                    lines.append(f"  {row['name']} [{status}] {last}")
                self.send_reply("\n".join(lines), recipient=XMPP_RECIPIENT)
            else:
                self.send_reply("No sessions yet.", recipient=XMPP_RECIPIENT)
            return

        if cmd == "/help":
            self.send_reply(
                "Send any message to start a new session.\n"
                "Each session appears as a separate contact.\n\n"
                "Prefixes:\n"
                "  @cc <msg> - start session on Claude\n"
                "  @oc <msg> - start session on OpenCode\n\n"
                "Aliases (some clients rewrite @cc):\n"
                "  /cc <msg> - start session on Claude\n"
                "  /oc <msg> - start session on OpenCode\n"
                "  cc <msg> - start session on Claude\n"
                "  oc <msg> - start session on OpenCode\n\n"
                "Commands:\n"
                "  /list - show all sessions\n"
                "  /recent - 10 most recent with status\n"
                "  /kill <name> - end a session\n"
                "  /cal - calendar (add/rm/list)\n"
                "  /tg - telegram (send/history/poll)\n"
                "  /help - this message",
                recipient=XMPP_RECIPIENT,
            )
            return

        self.send_reply(f"Unknown: {cmd}. Try /help", recipient=XMPP_RECIPIENT)

    async def create_session(self, first_message: str):
        """Create a new session and send first message to the engine."""
        self.send_typing()

        engine = "opencode"
        message = first_message.strip()
        lowered = message.lower()
        if lowered.startswith("@cc"):
            engine = "claude"
            message = message[3:].lstrip()
        elif lowered.startswith("@oc"):
            engine = "opencode"
            message = message[3:].lstrip()
        elif lowered.startswith("/cc"):
            engine = "claude"
            message = message[3:].lstrip()
        elif lowered.startswith("/oc"):
            engine = "opencode"
            message = message[3:].lstrip()
        elif lowered.startswith("cc "):
            engine = "claude"
            message = message[2:].lstrip()
        elif lowered.startswith("oc "):
            engine = "opencode"
            message = message[2:].lstrip()

        base_name = slugify(message or first_message)
        account = register_unique_account(base_name, self.db)
        if not account:
            self.send_reply(
                f"Failed to create XMPP account for {base_name}",
                recipient=XMPP_RECIPIENT,
            )
            return

        name, password, jid = account

        self.send_reply(f"Creating session: {name}...", recipient=XMPP_RECIPIENT)

        recipient_user = XMPP_RECIPIENT.split("@")[0]
        add_roster_subscription(name, XMPP_RECIPIENT, "Clients")
        add_roster_subscription(recipient_user, jid, "Sessions")

        create_tmux_session(name)

        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO sessions
               (name, xmpp_jid, xmpp_password, tmux_name, created_at, last_active, model_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, jid, password, name, now, now, "openai/gpt-5.2-codex"),
        )
        self.db.commit()

        self.db.execute(
            "UPDATE sessions SET active_engine = ? WHERE name = ?",
            (engine, name),
        )
        self.db.commit()

        if not self.manager:
            self.send_reply("Session manager unavailable.", recipient=XMPP_RECIPIENT)
            return
        bot = await self.manager.start_session_bot(name, jid, password)
        if bot:
            for _ in range(50):
                if bot.is_connected():
                    break
                await asyncio.sleep(0.1)

            bot.send_reply(
                f"Session '{name}' started. Engine: {engine}. "
                f"Processing: {message[:50] or first_message[:50]}..."
            )
            await bot.process_message(message or first_message)
        else:
            self.send_reply(
                f"Failed to start session bot for {name}",
                recipient=XMPP_RECIPIENT,
            )


# =============================================================================
# Manager
# =============================================================================


class SessionManager:
    """Manages all session bots and dispatcher."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.sessions: dict[str, SessionBot] = {}
        self.dispatcher: DispatcherBot | None = None
        self.loop = asyncio.get_event_loop()

    async def start_session_bot(self, name: str, jid: str, password: str):
        """Start a session bot."""
        bot = SessionBot(name, jid, password, self.db, manager=self)
        bot.connect_to_server(XMPP_SERVER)
        self.sessions[name] = bot
        return bot

    async def create_session(self, message: str):
        """Create a new session from dispatcher message."""
        base_name = slugify(message)
        account = register_unique_account(base_name, self.db)
        if not account:
            log.error(f"Failed to create session {base_name}")
            return

        name, password, jid = account

        recipient_user = XMPP_RECIPIENT.split("@")[0]
        add_roster_subscription(name, XMPP_RECIPIENT, "Clients")
        add_roster_subscription(recipient_user, jid, "Sessions")

        create_tmux_session(name)

        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO sessions
               (name, xmpp_jid, xmpp_password, tmux_name, created_at, last_active, model_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, jid, password, name, now, now, "openai/gpt-5.2-codex"),
        )
        self.db.commit()

        bot = await self.start_session_bot(name, jid, password)
        if bot:
            for _ in range(50):
                if bot.is_connected():
                    break
                await asyncio.sleep(0.1)

            bot.send_reply(f"Session '{name}' created. Message here to continue.")
            await bot.process_message(message)
        else:
            log.error(f"Failed to start bot for {name}")

    async def kill_session(self, name: str) -> bool:
        """Kill a session and cleanup."""
        row = self.db.execute(
            "SELECT xmpp_jid, xmpp_password, status FROM sessions WHERE name = ?",
            (name,),
        ).fetchone()

        if not row:
            return False

        if row["status"] == "closed":
            return True

        username = row["xmpp_jid"].split("@")[0]
        delete_xmpp_account(username)
        kill_tmux_session(name)

        self.db.execute("UPDATE sessions SET status = 'closed' WHERE name = ?", (name,))
        self.db.commit()
        if name in self.sessions:
            del self.sessions[name]
        return True

    async def start_dispatcher(self):
        """Start dispatcher bot."""
        self.dispatcher = DispatcherBot(
            XMPP_DISPATCHER_JID, XMPP_DISPATCHER_PASSWORD, self.db, manager=self
        )
        self.dispatcher.connect_to_server(XMPP_SERVER)

    async def restore_sessions(self):
        """Restore existing sessions from DB on startup."""
        rows = self.db.execute(
            "SELECT name, xmpp_jid, xmpp_password FROM sessions WHERE status = 'active'"
        ).fetchall()
        for row in rows:
            await self.start_session_bot(
                row["name"], row["xmpp_jid"], row["xmpp_password"]
            )
        log.info(f"Started {len(rows)} existing session(s)")


# =============================================================================
# Entry
# =============================================================================


async def main():
    db = init_db()
    manager = SessionManager(db)
    await manager.restore_sessions()
    await manager.start_dispatcher()

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down...")
