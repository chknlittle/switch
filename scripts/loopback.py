#!/usr/bin/env python3
"""Loopback test for Switch XMPP dispatcher.

Creates a temporary XMPP user, sends a message to the dispatcher, and prints
any responses to stdout. The account is removed afterward.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import sys
from pathlib import Path
from typing import Iterable

from slixmpp import ClientXMPP
from slixmpp.jid import JID

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import get_xmpp_config, load_env, run_ejabberdctl


class LoopbackBot(ClientXMPP):
    def __init__(
        self, jid: str, password: str, target: str, message: str, timeout: int
    ):
        super().__init__(jid, password)
        self.target = target
        self.message = message
        self.timeout = timeout
        self.responses: list[tuple[str, str]] = []

        self.register_plugin("xep_0199")
        self.register_plugin("xep_0085")
        self.register_plugin("xep_0280")

        self.add_event_handler("session_start", self._on_start)
        self.add_event_handler("message", self._on_message)
        self.add_event_handler("failed_auth", self._on_failed_auth)
        self.add_event_handler("disconnected", self._on_disconnected)

    async def _on_start(self, event):
        self.send_presence()
        await self.get_roster()
        self.send_message(mto=JID(self.target), mbody=self.message, mtype="chat")
        self.loop.call_later(self.timeout, self.disconnect)

    def _on_message(self, msg):
        if msg["type"] not in ("chat", "normal"):
            return
        body = msg["body"] or ""
        sender = str(msg["from"].bare)
        self.responses.append((sender, body))
        print(f"[from {sender}] {body}")

    def _on_failed_auth(self, event):
        print("Authentication failed", file=sys.stderr)
        self.disconnect()

    def _on_disconnected(self, event):
        self.loop.stop()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("slixmpp").setLevel(level)
    logging.getLogger("slixmpp.xmlstream").setLevel(level)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Switch XMPP loopback test")
    parser.add_argument("message", nargs="?", default="test-session")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dispatcher",
        choices=["cc", "oc", "oc-gpt"],
        default="cc",
        help="Which dispatcher to test (default: cc)",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    load_env()
    cfg = get_xmpp_config()

    domain = cfg["domain"]
    dispatcher_jid = cfg["dispatchers"][args.dispatcher]["jid"]
    server = cfg["server"]

    username = f"switch-loopback-{secrets.token_hex(3)}"
    password = secrets.token_urlsafe(12)

    success, output = run_ejabberdctl(
        cfg["ejabberd_ctl"], "register", username, domain, password
    )
    if not success:
        print(f"Failed to register loopback user: {output}", file=sys.stderr)
        return 1

    jid = f"{username}@{domain}"
    bot = LoopbackBot(jid, password, dispatcher_jid, args.message, args.timeout)
    bot["feature_mechanisms"].unencrypted_plain = True  # type: ignore[attr-defined]
    bot.enable_starttls = False
    bot.enable_direct_tls = False
    bot.enable_plaintext = True

    try:
        bot.connect(host=server, port=5222)
        bot.loop.run_forever()
        if not bot.responses:
            print("No responses received.")
        return 0
    finally:
        run_ejabberdctl(cfg["ejabberd_ctl"], "unregister", username, domain)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
