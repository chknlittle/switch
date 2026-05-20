#!/usr/bin/env python3
"""Offline smoke checks for Switch critical import and wiring paths.

Run from repo root via scripts/smoke.sh (sets PYTHONPATH + venv).
Exits non-zero on the first failing check.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_module(name: str) -> object:
    return importlib.import_module(name)


def _check_imports() -> None:
    for name in (
        "src.db",
        "src.bridge",
        "src.manager",
        "src.bots",
        "src.lifecycle.sessions",
        "src.delegation",
    ):
        _import_module(name)


def _check_init_db() -> None:
    import src.db.schema as schema

    old_path = schema.DB_PATH
    tmp_dir = Path(tempfile.mkdtemp(prefix="switch-smoke-db-"))
    schema.DB_PATH = tmp_dir / "sessions.db"
    try:
        conn = schema.init_db()
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not row:
            raise RuntimeError("init_db produced no tables")
        conn.close()
    finally:
        schema.DB_PATH = old_path


def _check_db_repos() -> None:
    from src.db import (
        DelegationTaskRepository,
        MessageRepository,
        RalphLoopRepository,
        SessionRepository,
    )
    import src.db.schema as schema

    old_path = schema.DB_PATH
    tmp_dir = Path(tempfile.mkdtemp(prefix="switch-smoke-repos-"))
    schema.DB_PATH = tmp_dir / "sessions.db"
    try:
        conn = schema.init_db()
        SessionRepository(conn)
        MessageRepository(conn)
        RalphLoopRepository(conn)
        DelegationTaskRepository(conn)
        conn.close()
    finally:
        schema.DB_PATH = old_path


def _check_engines() -> None:
    from src.engines import ENGINE_SPECS, normalize_engine

    assert normalize_engine("cc") == "claude"
    assert normalize_engine("OC") == "opencode"
    assert normalize_engine("vllm") == "vllm-direct"
    assert normalize_engine("nope") is None
    assert set(ENGINE_SPECS) == {
        "claude",
        "opencode",
        "pi",
        "cursor",
        "vllm-direct",
    }


def _check_ralph_parser() -> None:
    from src.ralph import parse_ralph_command

    parsed = parse_ralph_command("/ralph 3 fix the smoke test")
    if not parsed or parsed.get("max_iterations") != 3:
        raise RuntimeError(f"unexpected ralph parse: {parsed!r}")
    if parsed.get("prompt") != "fix the smoke test":
        raise RuntimeError(f"unexpected ralph prompt: {parsed!r}")

    swarm = parse_ralph_command("/ralph ship it --max 2 --swarm 3")
    if not swarm or swarm.get("swarm") != 3:
        raise RuntimeError(f"unexpected ralph swarm parse: {swarm!r}")

    assert parse_ralph_command("/not-ralph") is None


def _check_delegation_helpers() -> None:
    from src.delegation import build_envelope, parse_intent, resolve_dispatcher_name

    known = {"oc-gpt", "cc", "heretic"}
    if resolve_dispatcher_name("gpt", known) != "oc-gpt":
        raise RuntimeError("resolve_dispatcher_name alias failed")
    if resolve_dispatcher_name("unknown", known) is not None:
        raise RuntimeError("resolve_dispatcher_name should reject unknown")

    dispatchers = {name: {"jid": f"{name}@test"} for name in known}
    intent = parse_intent("ask oc-gpt summarize the refactor", dispatchers=dispatchers)
    if not intent or intent.dispatcher_name != "oc-gpt":
        raise RuntimeError(f"parse_intent failed: {intent!r}")

    envelope = build_envelope(token="tok", prompt="hi", parent_session="parent-1")
    if "tok" not in envelope or "parent-1" not in envelope:
        raise RuntimeError("build_envelope missing expected fields")


def _check_create_runners() -> None:
    from src.engines import ENGINE_SPECS
    from src.runners import create_runner

    tmp = Path(tempfile.mkdtemp(prefix="switch-smoke-runners-"))
    out = tmp / "output"
    out.mkdir()
    work = str(tmp / "work")
    Path(work).mkdir()

    for engine in ENGINE_SPECS:
        runner = create_runner(
            engine,
            working_dir=work,
            output_dir=out,
            session_name=f"smoke-{engine}",
        )
        if runner is None:
            raise RuntimeError(f"create_runner returned None for {engine!r}")


def _check_lifecycle_imports() -> None:
    from src.lifecycle.sessions import create_session, kill_session

    if not callable(create_session) or not callable(kill_session):
        raise RuntimeError("lifecycle session helpers not callable")


def _check_bots_imports() -> None:
    from src.bots import DirectoryBot, DispatcherBot, SessionBot

    for cls in (DirectoryBot, DispatcherBot, SessionBot):
        if not hasattr(cls, "__init__"):
            raise RuntimeError(f"{cls.__name__} missing __init__")


async def _check_bridge_wiring() -> None:
    """Exercise bridge lock + temp DB + SessionManager construct (no XMPP)."""
    import src.db.schema as schema
    from src.bridge import _SingleInstanceLock
    from src.manager import SessionManager
    from src.utils import get_xmpp_config

    lock_dir = Path(tempfile.mkdtemp(prefix="switch-smoke-bridge-"))
    lock_path = lock_dir / "bridge.lock"
    db_path = lock_dir / "sessions.db"

    old_db = schema.DB_PATH
    schema.DB_PATH = db_path
    lock = _SingleInstanceLock(lock_path)
    if not lock.acquire():
        raise RuntimeError("failed to acquire bridge single-instance lock")

    try:
        db = schema.init_db()
        cfg = get_xmpp_config()
        manager = SessionManager(
            db=db,
            working_dir=str(lock_dir / "work"),
            output_dir=lock_dir / "output",
            xmpp_server=cfg["server"],
            xmpp_domain=cfg["domain"],
            xmpp_recipient=cfg["recipient"],
            ejabberd_ctl=cfg["ejabberd_ctl"],
            dispatchers_config={},
        )
        if manager.sessions is None:
            raise RuntimeError("SessionManager missing sessions repo")
    finally:
        try:
            db.close()
        except Exception:
            pass
        lock.release()
        schema.DB_PATH = old_db


def _run(name: str, fn: Callable[[], None]) -> None:
    fn()


CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("imports (db, bridge, manager, bots, lifecycle, delegation)", _check_imports),
    ("init_db (temp sqlite)", _check_init_db),
    ("db repositories construct", _check_db_repos),
    ("engines.normalize_engine + ENGINE_SPECS", _check_engines),
    ("ralph.parse_ralph_command", _check_ralph_parser),
    ("delegation helpers (resolve, parse_intent, envelope)", _check_delegation_helpers),
    ("create_runner for each ENGINE_SPECS engine", _check_create_runners),
    ("lifecycle.sessions imports", _check_lifecycle_imports),
    ("bots (Directory, Dispatcher, Session)", _check_bots_imports),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Switch offline smoke harness")
    parser.add_argument(
        "--with-bridge",
        action="store_true",
        help="Also run bridge wiring check (lock, temp DB, SessionManager; no XMPP)",
    )
    args = parser.parse_args(argv)

    checks = list(CHECKS)
    if args.with_bridge:
        checks.append(
            (
                "bridge wiring (lock + SessionManager, offline)",
                lambda: asyncio.run(_check_bridge_wiring()),
            )
        )

    print(f"Switch smoke ({len(checks)} checks)\n")
    for name, fn in checks:
        try:
            _run(name, fn)
        except Exception as exc:
            print(f"FAIL  {name}")
            print(f"      {type(exc).__name__}: {exc}")
            return 1
        print(f"PASS  {name}")

    print(f"\nAll {len(checks)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
