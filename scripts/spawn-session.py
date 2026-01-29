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
import sys
from pathlib import Path

# Allow running this script directly without needing `uv run`.
# Ensures `import src.*` works when invoked as `python3 scripts/spawn-session.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import BaseXMPPBot, get_xmpp_config, load_env

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


class SpawnBot(BaseXMPPBot):
    def __init__(self, jid, password, target_jid, message):
        super().__init__(jid, password, recipient=target_jid)
        self.spawn_message = message
        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("failed_auth", self.on_failed_auth)

    def on_failed_auth(self, event):
        pass  # Slixmpp sometimes fires this even on success

    async def on_start(self, event):
        self.send_presence()
        self.send_reply(self.spawn_message)
        print(f"Sent to {self.recipient}")
        print(f"Message: {self.spawn_message[:100]}...")

        await asyncio.sleep(2)
        self.disconnect()


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

    bot = SpawnBot(dispatcher_jid, dispatcher_password, dispatcher_jid, message)
    bot.connect_to_server(XMPP_SERVER)

    try:
        await asyncio.wait_for(bot.disconnected, timeout=10)
        print("New session should be spawning...")
    except asyncio.TimeoutError:
        print("Timeout")
        bot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
