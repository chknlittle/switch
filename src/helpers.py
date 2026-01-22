#!/usr/bin/env python3
"""Helper functions for XMPP bridge."""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from src.utils import run_ejabberdctl as _run_ejabberdctl

HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"
ACTIVITY_LOG_PATH = Path.home() / ".claude" / "activity.jsonl"


def slugify(text: str, max_len: int = 20) -> str:
    """Convert text to a safe session/username."""
    words = text.lower().split()[:4]
    slug = "-".join(words)
    slug = re.sub(r"[^a-z0-9\\-]", "", slug)
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
        pass


def log_activity(message: str, session: str | None = None, source: str = "xmpp"):
    """Log activity to unified activity log."""
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


def run_ejabberdctl(ejabberd_ctl: str, *args) -> tuple[bool, str]:
    """Run an ejabberdctl command via SSH or locally."""
    return _run_ejabberdctl(ejabberd_ctl, *args)


def create_xmpp_account(
    username: str, password: str, ejabberd_ctl: str, domain: str, log
) -> tuple[bool, str]:
    """Create a new ejabberd account."""
    success, output = run_ejabberdctl(ejabberd_ctl, "register", username, domain, password)
    if success:
        log.info(f"Created XMPP account: {username}@{domain}")
    else:
        log.error(f"Failed to create account {username}: {output}")
    return success, output


def register_unique_account(
    base_name: str,
    db: sqlite3.Connection,
    ejabberd_ctl: str,
    domain: str,
    log,
    max_attempts: int = 5,
) -> tuple[str, str, str] | None:
    """Register a unique XMPP account, retrying on conflicts.

    Note: This function still takes a raw db connection for backwards
    compatibility with code that hasn't been migrated to repositories yet.
    """
    # Import here to avoid circular imports
    from src.db import SessionRepository

    sessions = SessionRepository(db)

    for idx in range(max_attempts):
        suffix = "" if idx == 0 else f"-{idx + 1}"
        trim_len = max(1, 20 - len(suffix))
        candidate = base_name[:trim_len].rstrip("-") + suffix

        if sessions.exists(candidate):
            continue

        password = secrets.token_urlsafe(16)
        success, output = create_xmpp_account(candidate, password, ejabberd_ctl, domain, log)
        if success:
            return candidate, password, f"{candidate}@{domain}"
        if "conflict" in output.lower():
            continue
        break

    return None


def add_roster_subscription(
    username: str, contact_jid: str, group: str, ejabberd_ctl: str, domain: str
) -> None:
    """Add mutual roster subscription between user and contact."""
    contact_user = contact_jid.split("@")[0]
    run_ejabberdctl(
        ejabberd_ctl,
        "add_rosteritem",
        username,
        domain,
        contact_user,
        domain,
        contact_user,
        group,
        "both",
    )


def delete_xmpp_account(username: str, ejabberd_ctl: str, domain: str, log) -> bool:
    """Delete an ejabberd account."""
    success, output = run_ejabberdctl(ejabberd_ctl, "unregister", username, domain)
    if success:
        log.info(f"Deleted XMPP account: {username}@{domain}")
    return success


def tmux_session_exists(name: str) -> bool:
    """Check if a tmux session exists."""
    result = subprocess.run(["tmux", "has-session", "-t", name], capture_output=True)
    return result.returncode == 0


def create_tmux_session(name: str, working_dir: str) -> bool:
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
            working_dir,
            str(script_path),
            name,
        ],
        capture_output=True,
    )
    return result.returncode == 0


def kill_tmux_session(name: str) -> bool:
    """Kill a tmux session."""
    result = subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)
    return result.returncode == 0
