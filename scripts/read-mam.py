#!/usr/bin/env python3
"""Read ejabberd MAM archives for Switch sessions.

This is a convenience script for agents/operators to fetch archived messages
without relying on conversational intent interception inside session bots.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
_VENV_ROOT = REPO_ROOT / ".venv"
if _VENV_PYTHON.exists() and Path(sys.prefix).resolve() != _VENV_ROOT.resolve():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), __file__, *sys.argv[1:]])

from slixmpp.exceptions import IqError, IqTimeout

from src.db import DB_PATH, SessionRepository
from src.utils import BaseXMPPBot, get_xmpp_config, load_env


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read MAM archives for Switch sessions")
    p.add_argument(
        "--session",
        required=True,
        help="Switch session name whose XMPP account is used to query MAM",
    )
    p.add_argument(
        "--conversation-jid",
        "--archive-jid",
        default="",
        help="Archive/session/room JID to query directly",
    )
    p.add_argument(
        "--with-jid",
        default="",
        help="Peer bare JID filter (user-to-user scope, e.g. fil@domain)",
    )
    p.add_argument(
        "--related",
        action="store_true",
        help="Include related sibling sessions (e.g. foo, foo-2, foo-3)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Max number of messages to print (default: 40)",
    )
    p.add_argument(
        "--contains",
        default="",
        help="Optional case-insensitive body substring filter",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="XMPP connection timeout seconds (default: 20)",
    )
    return p.parse_args(argv)


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _infer_related_archive_jids(
    sessions: SessionRepository, *, session_name: str, owner_jid: str
) -> list[str]:
    import re

    m = re.match(r"^(?P<base>.+)-(?P<num>\d+)$", session_name)
    if not m:
        return []

    base = (m.group("base") or "").strip()
    if not base:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for s in sessions.list_recent_for_owner(owner_jid, limit=100):
        if s.name == session_name:
            continue
        if not (s.name == base or s.name.startswith(f"{base}-")):
            continue
        jid = (s.xmpp_jid or "").split("/", 1)[0]
        if not jid or jid in seen:
            continue
        seen.add(jid)
        out.append(jid)
        if len(out) >= 12:
            break
    return out


def _looks_like_room_jid(jid: str) -> bool:
    bare = (jid or "").split("/", 1)[0]
    if not bare or "@" not in bare:
        return False
    domain = bare.split("@", 1)[1]
    return domain.startswith("conference.") or ".conference." in domain


def _format_stanza(stanza: Any) -> str | None:
    try:
        forwarded = stanza["mam_result"]["forwarded"]
        archived = forwarded["stanza"]
    except Exception:
        return None

    body = str(archived.get("body") or "").strip()
    if not body:
        return None

    sender = str(getattr(archived["from"], "bare", archived.get("from") or "?"))
    stamp_obj = None
    with suppress(Exception):
        stamp_obj = forwarded["delay"]["stamp"]

    if isinstance(stamp_obj, datetime):
        ts = stamp_obj.strftime("%Y-%m-%d %H:%M:%SZ")
    else:
        ts = str(stamp_obj or "unknown-time")
    return f"[{ts}] {sender}: {' '.join(body.splitlines())}"


class MamReadBot(BaseXMPPBot):
    def __init__(
        self,
        jid: str,
        password: str,
        *,
        owner_jid: str,
        archive_jids: list[str | None],
        with_jid: str | None,
        limit: int,
        contains: str,
    ):
        super().__init__(jid, password)
        self.owner_jid = owner_jid
        self.archive_jids = archive_jids
        self.with_jid = with_jid
        self.limit = max(1, min(limit, 500))
        self.contains = (contains or "").lower().strip()
        self.results: list[str] = []
        self.error: str | None = None
        self.register_plugin("xep_0313")
        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("disconnected", self.on_disconnected)
        self.add_event_handler("failed_auth", self.on_failed_auth)

    def on_failed_auth(self, event):
        del event
        self.error = "authentication failed"
        self.disconnect()

    def on_disconnected(self, event):
        del event
        self.set_connected(False)

    async def on_start(self, event):
        del event
        self.set_connected(True)
        self.send_presence()
        seen: set[str] = set()
        try:
            mam = self["xep_0313"]
            mam = mam if mam is not False else None
            if mam is None:
                self.error = "xep_0313 plugin unavailable"
                return
            rsm_max = min(self.limit, 50)
            for archive_jid in self.archive_jids:
                if len(self.results) >= self.limit:
                    break
                try:
                    async for stanza in mam.iterate(
                        jid=archive_jid,
                        with_jid=self.with_jid,
                        reverse=True,
                        rsm={"max": rsm_max},
                        total=max(1, self.limit - len(self.results)),
                    ):
                        row = _format_stanza(stanza)
                        if not row or row in seen:
                            continue
                        if self.contains and self.contains not in row.lower():
                            continue
                        seen.add(row)
                        self.results.append(row)
                        if len(self.results) >= self.limit:
                            break
                except (IqTimeout, IqError):
                    continue
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    continue
        finally:
            await asyncio.sleep(0.2)
            self.disconnect()


async def _run(argv: list[str]) -> int:
    args = _parse_args(argv)
    load_env()
    cfg = get_xmpp_config()

    conn = _open_db()
    try:
        sessions = SessionRepository(conn)
        session = sessions.get(args.session)
        if not session:
            print(f"Error: session '{args.session}' not found", file=sys.stderr)
            return 2

        session_jid = (session.xmpp_jid or "").split("/", 1)[0]
        owner_jid = (session.owner_jid or cfg.get("recipient") or "").split("/", 1)[0]
        if not session_jid or not session.xmpp_password:
            print(f"Error: session '{args.session}' missing XMPP credentials", file=sys.stderr)
            return 2

        with_jid: str | None = owner_jid or None
        if args.with_jid:
            with_jid = args.with_jid.split("/", 1)[0].strip() or with_jid
        archive_jids: list[str | None] = []

        target = (args.conversation_jid or "").strip()
        if target:
            target_bare = target.split("/", 1)[0]
            if target_bare != session_jid and not _looks_like_room_jid(target_bare):
                print(
                    "Error: this script authenticates as the session account, so it cannot "
                    "query another user's private archive directly. "
                    "Use --with-jid to filter that session's own archive, or run with a "
                    "session that owns the target archive.",
                    file=sys.stderr,
                )
                return 2
            archive_jids = [target_bare]
        elif args.related:
            related = _infer_related_archive_jids(
                sessions, session_name=session.name, owner_jid=owner_jid
            )
            archive_jids = [*related] if related else [None]
        else:
            archive_jids = [None]

        bot = MamReadBot(
            session_jid,
            session.xmpp_password,
            owner_jid=owner_jid,
            archive_jids=archive_jids,
            with_jid=with_jid,
            limit=int(args.limit),
            contains=args.contains,
        )
        bot.connect_to_server(cfg["server"])
        ok = await bot.wait_connected(timeout=max(2.0, float(args.timeout)))
        if not ok:
            print("Error: timeout connecting to XMPP", file=sys.stderr)
            with suppress(Exception):
                bot.disconnect()
            return 3

        await asyncio.wait_for(bot.disconnected, timeout=max(5.0, float(args.timeout) + 5.0))
        if bot.error and not bot.results:
            print(f"Error: {bot.error}", file=sys.stderr)
            return 4

        if not bot.results:
            scopes = [s for s in archive_jids if s]
            scope_desc = ", ".join(scopes) if scopes else "with=<owner>"
            print(f"No archived messages found (scope: {scope_desc})")
            return 0

        for line in bot.results[: args.limit]:
            print(line)
        return 0
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    return asyncio.run(_run(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
