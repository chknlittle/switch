#!/usr/bin/env python3
"""
Send a message to a dispatcher to spawn a new session.

Usage:
    spawn-session.py [--dispatcher <name>] <message>
    spawn-session.py --list-dispatchers

Example:
    spawn-session.py "Continue moonshot-survival work. Context: ..."

    # Use a specific orchestrator/dispatcher (e.g. oc-gpt-or, oc-kimi-coding)
    spawn-session.py --dispatcher oc-kimi-coding "Reply only: ok"

Notes:
    - Default dispatcher can be set with SWITCH_DEFAULT_DISPATCHER.
    - Use -h/--help to print this message (does not spawn a session).
"""

import asyncio
import os
import secrets
import sqlite3
import sys
from pathlib import Path

# Allow running this script directly without needing `uv run`.
# Ensures `import src.*` works when invoked as `python3 scripts/spawn-session.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.delegation import find_spawned_session_for_token, get_latest_message_id, send_dispatcher_message
from src.utils import get_xmpp_config, load_env

# Load environment
load_env()
cfg = get_xmpp_config()

XMPP_SERVER = cfg["server"]


def _default_dispatcher_name() -> str:
    # Prefer explicit env override, fall back to a sane default.
    return (os.getenv("SWITCH_DEFAULT_DISPATCHER") or "oc-gpt").strip() or "oc-gpt"


def _parse_args(argv: list[str]) -> tuple[str, str] | None:
    if not argv:
        return None

    if argv[0] in {"-h", "--help"}:
        print(__doc__)
        return ("__exit__", "")

    if argv[0] == "--list-dispatchers":
        return ("__list__", "")

    dispatcher_name = _default_dispatcher_name()

    if argv[0].startswith("--dispatcher="):
        dispatcher_name = argv[0].split("=", 1)[1]
        argv = argv[1:]
    elif len(argv) >= 2 and argv[0] in {"--dispatcher", "-d"}:
        dispatcher_name = argv[1]
        argv = argv[2:]

    message = " ".join(argv).strip()
    if not message:
        return None

    return dispatcher_name, message

async def main():
    parsed = _parse_args(sys.argv[1:])
    if not parsed:
        print(__doc__)
        sys.exit(1)

    dispatcher_name, message = parsed

    if dispatcher_name == "__exit__":
        sys.exit(0)

    dispatchers = cfg.get("dispatchers", {})

    if dispatcher_name == "__list__":
        known = "\n".join(f"- {name}" for name in sorted(dispatchers.keys())) or "(none)"
        print("Known dispatchers:\n" + known)
        sys.exit(0)

    dispatcher = dispatchers.get(dispatcher_name)
    if not dispatcher:
        known = ", ".join(sorted(dispatchers.keys())) or "none"
        print(f"Error: unknown dispatcher '{dispatcher_name}'. Known: {known}")
        sys.exit(2)

    dispatcher_jid = dispatcher["jid"]
    dispatcher_password = dispatcher["password"]

    if not dispatcher_password:
        print(
            f"Error: no password configured for dispatcher '{dispatcher_name}'. "
            "Set the dispatcher-specific password env var (or XMPP_PASSWORD) in .env."
        )
        sys.exit(1)

    token = f"switch-spawn-{secrets.token_hex(6)}"
    envelope = f"[{token}]\n{message}"
    conn = sqlite3.connect(REPO_ROOT / "sessions.db")
    conn.row_factory = sqlite3.Row
    min_message_id = get_latest_message_id(conn)

    try:
        await send_dispatcher_message(
            server=XMPP_SERVER,
            dispatcher_jid=dispatcher_jid,
            dispatcher_password=dispatcher_password,
            body=envelope,
        )
        print(f"Sent spawn request to {dispatcher_jid}")

        for _ in range(30):
            conn.commit()
            spawned = find_spawned_session_for_token(
                conn,
                dispatcher_jid=dispatcher_jid,
                token=token,
                min_message_id=min_message_id,
            )
            if spawned:
                session_name, _message_id = spawned
                print(f"Spawned session: {session_name}@{cfg['domain']}")
                print("New session should be visible in your chat client shortly.")
                return
            await asyncio.sleep(1)

        print("Spawn request sent, but timed out waiting for DB confirmation.")
        print("The session may still appear shortly in XMPP/tmux.")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
