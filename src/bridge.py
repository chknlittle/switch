#!/usr/bin/env python3
"""
Switch - XMPP AI Assistant Bridge

Each session gets its own XMPP account, appearing as a separate
chat contact in the client.

Dispatcher contacts are configured at runtime (supports 1..N dispatchers).

Send any message to a dispatcher to create a new session.
Each session appears as its own contact (e.g., react-app@...).
Reply directly to that contact to continue the conversation.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import signal
from pathlib import Path

from src.attachments import AttachmentStore
from src.attachments.config import get_attachments_config
from src.attachments.server import start_attachments_server
from src.db import init_db
from src.helpers import create_xmpp_account
from src.manager import SessionManager
from src.telemetry import init_telemetry
from src.settings.env import get_xmpp_config, load_env

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


class _SingleInstanceLock:
    def __init__(self, path: Path):
        self._path = path
        self._fh = None

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return True

    def release(self) -> None:
        if not self._fh:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None


async def main():
    # Error reporting (GlitchTip) — as early as possible so that even
    # lock/DB infrastructure failures are captured. No-op without DSN.
    init_telemetry()

    lock_path = Path(
        os.getenv("SWITCH_LOCK_FILE", "/tmp/switch-bridge.lock")
    ).expanduser()
    lock = _SingleInstanceLock(lock_path)
    if not lock.acquire():
        log.error(
            "Another Switch bridge instance is already running (lock: %s). Exiting.",
            lock_path,
        )
        return

    db = init_db()

    try:
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
                log.info(
                    "Directory service disabled (missing SWITCH_DIRECTORY_JID/PASSWORD)"
                )
        except Exception:
            log.exception("Failed to start directory service")
        await manager.restore_sessions()
        await manager.start_dispatchers()

        # Graceful shutdown on SIGINT/SIGTERM.
        shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)

        await shutdown_event.wait()
        log.info("Signal received, shutting down gracefully...")
        await manager.shutdown()
    finally:
        lock.release()
        log.info("Closing database connection")
        try:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            log.warning("WAL checkpoint failed", exc_info=True)
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
