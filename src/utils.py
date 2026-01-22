#!/usr/bin/env python3
"""
Shared utilities for XMPP bridge components.
"""

import os
import subprocess
from pathlib import Path

from slixmpp.clientxmpp import ClientXMPP

# =============================================================================
# Environment Loading
# =============================================================================


def load_env(env_path: Path | None = None) -> None:
    """Load .env file into os.environ. Handles quoted values and spaces."""
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            val = val.strip().strip('"').strip("'")
            os.environ[key.strip()] = val


# =============================================================================
# Configuration (call load_env() before accessing these)
# =============================================================================


def get_xmpp_config() -> dict:
    """Get XMPP configuration from environment."""
    server = os.getenv("XMPP_SERVER", "your.xmpp.server")
    return {
        "server": server,
        "domain": os.getenv("XMPP_DOMAIN", server),
        "dispatcher_jid": os.getenv("XMPP_JID", f"oc@{server}"),
        "dispatcher_password": os.getenv("XMPP_PASSWORD", ""),
        "recipient": os.getenv("XMPP_RECIPIENT", f"user@{server}"),
        "ejabberd_ctl": os.getenv(
            "EJABBERD_CTL",
            f"ssh user@{server} /path/to/ejabberdctl",
        ),
    }


# =============================================================================
# Ejabberd Commands
# =============================================================================


def run_ejabberdctl(ejabberd_ctl: str, *args) -> tuple[bool, str]:
    """Run an ejabberdctl command via SSH or locally."""
    if ejabberd_ctl.startswith("ssh "):
        parts = ejabberd_ctl.split(maxsplit=2)
        remote_cmd = parts[2] + " " + " ".join(args)
        cmd = ["ssh", parts[1], remote_cmd]
    else:
        cmd = ejabberd_ctl.split() + list(args)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output


# =============================================================================
# Base XMPP Bot
# =============================================================================


class BaseXMPPBot(ClientXMPP):
    """
    Base class for all XMPP bots with common setup.

    Provides:
    - Standard plugin registration (xep_0199, xep_0085, xep_0280)
    - Unencrypted plain auth setup
    - Common connect method
    - send_reply and send_typing helpers
    """

    def __init__(self, jid: str, password: str, recipient: str | None = None):
        super().__init__(jid, password)
        self.recipient = recipient

        # Common plugins
        self.register_plugin("xep_0199")  # Ping
        self.register_plugin("xep_0085")  # Chat State Notifications
        self.register_plugin("xep_0280")  # Message Carbons

    def connect_to_server(self, server: str, port: int = 5222):
        """Connect with standard settings (unencrypted, no TLS)."""
        self["feature_mechanisms"].unencrypted_plain = True  # type: ignore[attr-defined]
        self.enable_starttls = False
        self.enable_direct_tls = False
        self.connect(server, port)

    def send_reply(self, text: str, recipient: str | None = None):
        """Send a chat message to recipient."""
        to = recipient or self.recipient
        if not to:
            raise ValueError("No recipient specified")
        msg = self.make_message(mto=to, mbody=text, mtype="chat")
        msg["chat_state"] = "active"
        msg.send()

    def send_typing(self, recipient: str | None = None):
        """Send composing (typing) indicator."""
        to = recipient or self.recipient
        if not to:
            return
        msg = self.make_message(mto=to, mtype="chat")
        msg["chat_state"] = "composing"
        msg.send()
