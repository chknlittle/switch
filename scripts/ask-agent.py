#!/usr/bin/env python3
"""Ask another dispatcher/session and wait for first answer.

This is a delegation helper for agents running on the Switch box.
It sends a prompt to a dispatcher, waits for the spawned session to produce
its first assistant message, and prints that result to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.db import DB_PATH
from src.delegation import delegate_once
from src.utils import get_xmpp_config, load_env


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delegate prompt to dispatcher")
    parser.add_argument("prompt", nargs="+", help="prompt for delegated agent")
    parser.add_argument(
        "--dispatcher",
        "-d",
        default=None,
        help="dispatcher name (default: SWITCH_DEFAULT_DISPATCHER or oc-gpt)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="max seconds to wait for a delegated result",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="sqlite polling interval in seconds",
    )
    return parser.parse_args(argv)


def _default_dispatcher_name() -> str:
    import os

    return (os.getenv("SWITCH_DEFAULT_DISPATCHER") or "oc-gpt").strip() or "oc-gpt"


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    load_env()
    cfg = get_xmpp_config()

    dispatcher_name = (args.dispatcher or _default_dispatcher_name()).strip()
    dispatcher = cfg.get("dispatchers", {}).get(dispatcher_name)
    if not dispatcher:
        known = ", ".join(sorted((cfg.get("dispatchers") or {}).keys())) or "none"
        print(f"Error: unknown dispatcher '{dispatcher_name}'. Known: {known}", file=sys.stderr)
        return 2

    dispatcher_jid = str(dispatcher.get("jid") or "").strip()
    dispatcher_password = str(dispatcher.get("password") or "").strip()
    if not dispatcher_password:
        print(
            f"Error: dispatcher '{dispatcher_name}' has no password configured.",
            file=sys.stderr,
        )
        return 1

    prompt_text = " ".join(args.prompt).strip()
    if not prompt_text:
        print("Error: empty prompt", file=sys.stderr)
        return 1

    conn = _open_db()
    try:
        result = await delegate_once(
            conn,
            server=cfg["server"],
            dispatcher_jid=dispatcher_jid,
            dispatcher_password=dispatcher_password,
            prompt=prompt_text,
            parent_session="ask-agent",
            timeout_s=float(args.timeout or 0.0),
            poll_interval_s=float(args.poll_interval or 1.0),
        )
        print(result.content)
        return 0
    except TimeoutError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
