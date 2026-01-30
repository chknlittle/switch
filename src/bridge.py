#!/usr/bin/env python3
"""
Switch - XMPP AI Assistant Bridge

Each OpenCode/Claude session gets its own XMPP account, appearing as a separate
chat contact in the client.

Orchestrator contacts:
- cc@... - Claude Code sessions
- oc@... - OpenCode with GLM 4.7 (local)
- oc-gpt@... - OpenCode with GPT 5.2
- oc-glm-zen@... - OpenCode with GLM 4.7 (Zen)
- oc-gpt-or@... - OpenCode with GPT 5.2 (OpenRouter)

Send any message to an orchestrator to create a new session.
Each session appears as its own contact (e.g., react-app@...).
Reply directly to that contact to continue the conversation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from src.attachments import AttachmentStore
from src.attachments.config import get_attachments_config
from src.attachments.server import start_attachments_server
from src.db import init_db
from src.helpers import create_xmpp_account
from src.manager import SessionManager
from src.utils import get_xmpp_config, load_env

# Load environment
load_env()

# Configuration
_cfg = get_xmpp_config()
XMPP_SERVER = _cfg["server"]
XMPP_DOMAIN = _cfg["domain"]
XMPP_RECIPIENT = _cfg["recipient"]
PUBSUB_SERVICE = _cfg["pubsub_service"]
DIRECTORY_CFG = _cfg["directory"]
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

    # Serve attachments over HTTP so chat clients can open `public_url` links.
    attachments_server = None
    attachments_cfg = get_attachments_config()
    attachments_store = AttachmentStore(
        base_dir=attachments_cfg.base_dir,
        public_base_url=attachments_cfg.public_base_url,
        token=attachments_cfg.token,
    )
    if os.getenv("SWITCH_ATTACHMENTS_ENABLE", "1").lower() in {"1", "true", "yes"}:
        try:
            attachments_server, host, port = await start_attachments_server(
                attachments_store.base_dir,
                token=attachments_cfg.token,
                host=attachments_cfg.host,
                port=attachments_cfg.port,
            )
            log.info(f"Attachments server listening on http://{host}:{port}")
        except Exception:
            log.exception("Failed to start attachments server")

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

    # Start directory service (XEP-0030 + pubsub refresh).
    try:
        directory_jid = DIRECTORY_CFG.get("jid")
        directory_password = DIRECTORY_CFG.get("password")
        if directory_jid and directory_password:
            if DIRECTORY_CFG.get("autocreate"):
                username = directory_jid.split("@")[0]
                # Best-effort: if account already exists, ejabberd will return conflict.
                create_xmpp_account(
                    username,
                    directory_password,
                    EJABBERD_CTL,
                    XMPP_DOMAIN,
                    log,
                    allow_conflict=True,
                )
            await manager.start_directory_service(
                jid=directory_jid,
                password=directory_password,
                pubsub_service_jid=PUBSUB_SERVICE,
            )
        else:
            log.info("Directory service disabled (missing SWITCH_DIRECTORY_JID/PASSWORD)")
    except Exception:
        log.exception("Failed to start directory service")
    await manager.restore_sessions()
    await manager.start_dispatchers()

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down...")
