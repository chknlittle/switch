#!/usr/bin/env python3
"""
Shared utilities for XMPP bridge components.
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path

from slixmpp.clientxmpp import ClientXMPP
from slixmpp.xmlstream import ET


SWITCH_META_NS = "urn:switch:message-meta"


def build_message_meta(
    meta_type: str,
    *,
    meta_tool: str | None = None,
    meta_attrs: dict[str, str] | None = None,
    meta_payload: object | None = None,
) -> ET.Element:
    """Build a Switch message meta extension element.

    This keeps structured data out of the message body, while remaining
    backward-compatible with clients that ignore unknown XML extensions.
    """

    meta = ET.Element(f"{{{SWITCH_META_NS}}}meta")
    meta.set("type", meta_type)
    if meta_tool:
        meta.set("tool", meta_tool)

    if meta_attrs:
        for k, v in meta_attrs.items():
            if not k or v is None:
                continue
            if k in ("type", "tool"):
                continue
            meta.set(str(k), str(v))

    if meta_payload is not None:
        payload = ET.SubElement(meta, f"{{{SWITCH_META_NS}}}payload")
        payload.set("format", "json")
        payload.text = json.dumps(
            meta_payload, ensure_ascii=True, separators=(",", ":")
        )

    return meta


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
    domain = os.getenv("XMPP_DOMAIN", server)
    raw_directory_jid = os.getenv(
        "SWITCH_DIRECTORY_JID", f"switch-dir@{domain}"
    ).strip()
    # ejabberd answers disco#items for bare user JIDs itself (PEP), so our
    # directory bot must be addressed as a *full* JID resource to reach the
    # connected client.
    if "/" not in raw_directory_jid:
        raw_directory_jid = f"{raw_directory_jid}/directory"

    return {
        "server": server,
        "domain": domain,
        "recipient": os.getenv("XMPP_RECIPIENT", f"user@{server}"),
        "pubsub_service": os.getenv("SWITCH_PUBSUB_JID", f"pubsub.{domain}"),
        "directory": {
            "jid": raw_directory_jid,
            "password": os.getenv(
                "SWITCH_DIRECTORY_PASSWORD", os.getenv("XMPP_PASSWORD", "")
            ),
            "autocreate": os.getenv("SWITCH_DIRECTORY_AUTOCREATE", "1")
            not in ("0", "false", "False"),
        },
        "ejabberd_ctl": os.getenv(
            "EJABBERD_CTL",
            f"ssh user@{server} /path/to/ejabberdctl",
        ),
        # Three separate dispatchers
        "dispatchers": {
            "cc": {
                "jid": os.getenv("CC_JID", f"cc@{domain}"),
                "password": os.getenv("CC_PASSWORD", ""),
                "engine": "claude",
                "agent": None,
                "label": "Claude Code",
            },
            "oc": {
                "jid": os.getenv("OC_JID", f"oc@{domain}"),
                "password": os.getenv("OC_PASSWORD", ""),
                "engine": "opencode",
                "agent": "bridge",
                "label": "GLM 4.7 Heretic",
            },
            "oc-gpt": {
                "jid": os.getenv("OC_GPT_JID", f"oc-gpt@{domain}"),
                "password": os.getenv("OC_GPT_PASSWORD", ""),
                "engine": "opencode",
                "agent": "bridge-gpt",
                "label": "GPT 5.2",
            },
            "oc-glm-zen": {
                "jid": os.getenv("OC_GLM_ZEN_JID", f"oc-glm-zen@{domain}"),
                "password": os.getenv(
                    "OC_GLM_ZEN_PASSWORD", os.getenv("XMPP_PASSWORD", "")
                ),
                "engine": "opencode",
                "agent": "bridge-zen",
                "label": "GLM 4.7 Zen",
            },
            "oc-gpt-or": {
                "jid": os.getenv("OC_GPT_OR_JID", f"oc-gpt-or@{domain}"),
                "password": os.getenv(
                    "OC_GPT_OR_PASSWORD", os.getenv("XMPP_PASSWORD", "")
                ),
                "engine": "opencode",
                "agent": "bridge-gpt-or",
                "label": "GPT 5.2 OR",
            },
            "oc-kimi-coding": {
                "jid": os.getenv("OC_KIMI_CODING_JID", f"oc-kimi-coding@{domain}"),
                "password": os.getenv(
                    "OC_KIMI_CODING_PASSWORD", os.getenv("XMPP_PASSWORD", "")
                ),
                "engine": "opencode",
                "agent": "bridge-kimi-coding",
                "label": "Kimi K2.5 Coding",
            },
        },
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
        self._connected_event = asyncio.Event()

        # Common plugins
        self.register_plugin("xep_0199")  # Ping
        self.register_plugin("xep_0085")  # Chat State Notifications
        self.register_plugin("xep_0280")  # Message Carbons

    def connect_to_server(self, server: str, port: int = 5222):
        """Connect with standard settings (unencrypted, no TLS)."""
        self["feature_mechanisms"].unencrypted_plain = True  # type: ignore[attr-defined]
        self.enable_starttls = False
        self.enable_direct_tls = False
        self.enable_plaintext = True
        # slixmpp.ClientXMPP.connect expects a single address tuple.
        # TLS behavior is governed by the enable_* flags above.
        self.connect((server, port))  # type: ignore[arg-type]

    def set_connected(self, connected: bool) -> None:
        if connected:
            self._connected_event.set()
        else:
            self._connected_event.clear()

    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    async def wait_connected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def send_reply(
        self,
        text: str,
        recipient: str | None = None,
        *,
        meta_type: str | None = None,
        meta_tool: str | None = None,
        meta_attrs: dict[str, str] | None = None,
        meta_payload: object | None = None,
    ):
        """Send a chat message to recipient."""
        to = recipient or self.recipient
        if not to:
            raise ValueError("No recipient specified")
        msg = self.make_message(mto=to, mbody=text, mtype="chat")
        msg["chat_state"] = "active"

        # Optional message metadata extension.
        if meta_type:
            meta = build_message_meta(
                meta_type,
                meta_tool=meta_tool,
                meta_attrs=meta_attrs,
                meta_payload=meta_payload,
            )
            msg.xml.append(meta)

        msg.send()

    def send_typing(self, recipient: str | None = None):
        """Send composing (typing) indicator."""
        to = recipient or self.recipient
        if not to:
            return
        msg = self.make_message(mto=to, mtype="chat")
        msg["chat_state"] = "composing"
        msg.send()

    def _format_exception_for_user(self, exc: BaseException) -> str:
        msg = str(exc).strip()
        if msg:
            return f"Error: {type(exc).__name__}: {msg}"
        return f"Error: {type(exc).__name__}"

    async def guard(
        self,
        coro,
        *,
        recipient: str | None = None,
        context: str | None = None,
    ):
        """Run a coroutine with a single error boundary.

        - Lets internal code raise normally.
        - Catches at the boundary, logs, and sends an error message to the
          relevant recipient.
        """

        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log = getattr(self, "log", logging.getLogger("xmpp"))
            if context:
                log.exception("Unhandled error (%s)", context)
            else:
                log.exception("Unhandled error")
            try:
                self.send_reply(
                    self._format_exception_for_user(exc), recipient=recipient
                )
            except Exception:
                pass
            return None

    def spawn_guarded(
        self,
        coro,
        *,
        recipient: str | None = None,
        context: str | None = None,
    ) -> asyncio.Task:
        """Create a task that reports exceptions to the user."""

        task = asyncio.create_task(
            self.guard(coro, recipient=recipient, context=context)
        )
        return task
