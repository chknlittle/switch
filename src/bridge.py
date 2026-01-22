#!/usr/bin/env python3
"""
Switch - XMPP AI Assistant Bridge

Each OpenCode/Claude session gets its own XMPP account, appearing as a separate
chat contact in the client.

Orchestrator contacts:
- cc@... - Claude Code sessions
- oc@... - OpenCode with GLM 4.7 (local)
- oc-gpt@... - OpenCode with GPT 5.2

Send any message to an orchestrator to create a new session.
Each session appears as its own contact (e.g., react-app@...).
Reply directly to that contact to continue the conversation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from src.db import init_db
from src.manager import SessionManager
from src.utils import get_xmpp_config, load_env

# Load environment
load_env()

# Configuration
_cfg = get_xmpp_config()
XMPP_SERVER = _cfg["server"]
XMPP_DOMAIN = _cfg["domain"]
XMPP_RECIPIENT = _cfg["recipient"]
EJABBERD_CTL = _cfg["ejabberd_ctl"]
DISPATCHERS = _cfg["dispatchers"]
WORKING_DIR = os.getenv("SWITCH_WORKING_DIR", str(Path.home()))
SESSION_OUTPUT_DIR = Path(__file__).parent.parent / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")


async def main():
    db = init_db()
    manager = SessionManager(
        db=db,
        working_dir=WORKING_DIR,
        output_dir=SESSION_OUTPUT_DIR,
        xmpp_server=XMPP_SERVER,
        xmpp_domain=XMPP_DOMAIN,
        xmpp_recipient=XMPP_RECIPIENT,
        ejabberd_ctl=EJABBERD_CTL,
        dispatchers_config=DISPATCHERS,
    )
    await manager.restore_sessions()
    await manager.start_dispatchers()

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down...")
