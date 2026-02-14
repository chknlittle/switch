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

from src.engines import (
    OPENCODE_MODEL_CODEX,
    OPENCODE_MODEL_DEFAULT,
    OPENCODE_MODEL_GPT,
    OPENCODE_MODEL_GPT_OR,
    OPENCODE_MODEL_KIMI_CODING,
    OPENCODE_MODEL_ZEN,
)
from slixmpp.clientxmpp import ClientXMPP
from slixmpp.xmlstream import ET


SWITCH_META_NS = "urn:switch:message-meta"
_log = logging.getLogger("utils")


def _parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _legacy_dispatchers(domain: str) -> dict[str, dict]:
    # Ordered to match hotkey assignments (Cmd+1..4), then the rest.
    return {
        "acorn": {
            "jid": os.getenv("ACORN_JID", f"acorn@{domain}"),
            "password": os.getenv("ACORN_PASSWORD", ""),
            "engine": "external",
            "agent": None,
            "label": "Acorn",
            "direct": True,
        },
        "oc-codex": {
            "jid": os.getenv("OC_CODEX_JID", f"oc-codex@{domain}"),
            "password": os.getenv("OC_CODEX_PASSWORD", os.getenv("XMPP_PASSWORD", "")),
            "engine": "opencode",
            "agent": "bridge-gpt",
            "model_id": os.getenv("OC_CODEX_MODEL_ID", OPENCODE_MODEL_CODEX),
            "label": "Codex 5.3",
        },
        "cc": {
            "jid": os.getenv("CC_JID", f"cc@{domain}"),
            "password": os.getenv("CC_PASSWORD", ""),
            "engine": "claude",
            "agent": None,
            "label": "Claude Code",
        },
        "oc-gpt": {
            "jid": os.getenv("OC_GPT_JID", f"oc-gpt@{domain}"),
            "password": os.getenv("OC_GPT_PASSWORD", ""),
            "engine": "opencode",
            "agent": "bridge-gpt",
            "model_id": os.getenv("OC_GPT_MODEL_ID", OPENCODE_MODEL_GPT),
            "label": "GPT 5.2",
        },
        "oc": {
            "jid": os.getenv("OC_JID", f"oc@{domain}"),
            "password": os.getenv("OC_PASSWORD", ""),
            "engine": "opencode",
            "agent": "bridge",
            "model_id": os.getenv("OC_MODEL_ID", OPENCODE_MODEL_DEFAULT),
            "label": "GLM 4.7 Heretic",
        },
        "oc-glm-zen": {
            "jid": os.getenv("OC_GLM_ZEN_JID", f"oc-glm-zen@{domain}"),
            "password": os.getenv(
                "OC_GLM_ZEN_PASSWORD", os.getenv("XMPP_PASSWORD", "")
            ),
            "engine": "opencode",
            "agent": "bridge-zen",
            "model_id": os.getenv("OC_GLM_ZEN_MODEL_ID", OPENCODE_MODEL_ZEN),
            "label": "GLM 4.7 Zen",
        },
        "oc-gpt-or": {
            "jid": os.getenv("OC_GPT_OR_JID", f"oc-gpt-or@{domain}"),
            "password": os.getenv("OC_GPT_OR_PASSWORD", os.getenv("XMPP_PASSWORD", "")),
            "engine": "opencode",
            "agent": "bridge-gpt-or",
            "model_id": os.getenv("OC_GPT_OR_MODEL_ID", OPENCODE_MODEL_GPT_OR),
            "label": "GPT 5.2 OR",
        },
        "oc-kimi-coding": {
            "jid": os.getenv("OC_KIMI_CODING_JID", f"oc-kimi-coding@{domain}"),
            "password": os.getenv(
                "OC_KIMI_CODING_PASSWORD", os.getenv("XMPP_PASSWORD", "")
            ),
            "engine": "opencode",
            "agent": "bridge-kimi-coding",
            "model_id": os.getenv(
                "OC_KIMI_CODING_MODEL_ID", OPENCODE_MODEL_KIMI_CODING
            ),
            "label": "Kimi K2.5 Coding",
        },
    }


def _normalize_dispatchers(payload: object, *, domain: str) -> dict[str, dict]:
    """Normalize dispatcher config from list/dict JSON to internal mapping."""

    entries: list[tuple[str, dict]] = []
    if isinstance(payload, list):
        for i, item in enumerate(payload):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("id") or f"dispatcher-{i + 1}")
                entries.append((name, item))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", str(key))
                entries.append((str(key), item))
    else:
        raise ValueError("dispatchers config must be a JSON list or object")

    out: dict[str, dict] = {}
    for fallback_name, item in entries:
        name = str(item.get("name") or fallback_name).strip() or fallback_name
        jid = str(item.get("jid") or f"{name}@{domain}").strip()
        if not jid:
            _log.warning("Skipping dispatcher %s: missing jid", name)
            continue

        password = ""
        if isinstance(item.get("password"), str):
            password = item.get("password", "").strip()
        elif isinstance(item.get("password_env"), str):
            password = os.getenv(item.get("password_env", ""), "").strip()

        engine = str(item.get("engine") or "opencode").strip().lower()
        agent = item.get("agent")
        if isinstance(agent, str):
            agent = agent.strip() or None
        elif agent is not None:
            agent = str(agent).strip() or None

        entry: dict[str, object] = {
            "jid": jid,
            "password": password,
            "engine": engine,
            "agent": agent,
            "label": str(item.get("label") or name),
        }

        model_id = item.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            entry["model_id"] = model_id.strip()

        if _parse_bool(item.get("direct"), default=False):
            entry["direct"] = True
        if _parse_bool(item.get("disabled"), default=False):
            entry["disabled"] = True

        out[name] = entry

    return out


def _load_dispatchers_config(domain: str) -> dict[str, dict]:
    """Load dispatchers from JSON env/file, with legacy fallback."""

    raw_json = (os.getenv("SWITCH_DISPATCHERS_JSON", "") or "").strip()
    raw_file = (os.getenv("SWITCH_DISPATCHERS_FILE", "") or "").strip()

    payload: object | None = None
    if raw_json:
        try:
            payload = json.loads(raw_json)
        except Exception as e:
            _log.warning(
                "Invalid SWITCH_DISPATCHERS_JSON; using legacy dispatchers: %s", e
            )
    elif raw_file:
        try:
            payload = json.loads(Path(raw_file).read_text())
        except Exception as e:
            _log.warning(
                "Invalid SWITCH_DISPATCHERS_FILE; using legacy dispatchers: %s", e
            )

    if payload is None:
        return _legacy_dispatchers(domain)

    try:
        cfg = _normalize_dispatchers(payload, domain=domain)
        if cfg:
            return cfg
        _log.warning(
            "Dispatcher config resolved to empty set; using legacy dispatchers"
        )
    except Exception as e:
        _log.warning(
            "Failed to normalize dispatchers config; using legacy dispatchers: %s", e
        )
    return _legacy_dispatchers(domain)


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
        "dispatchers": _load_dispatchers_config(domain),
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
