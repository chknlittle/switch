#!/usr/bin/env python3
"""
Shared utilities for XMPP bridge components.
"""

import json
import logging
import shlex
import subprocess

from slixmpp.xmlstream import ET

# Re-exported for callers outside the repo (skills, runbooks).
from src.settings.env import get_xmpp_config, load_env  # noqa: F401

SWITCH_META_NS = "urn:switch:message-meta"
_log = logging.getLogger("utils")


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
# Ejabberd Commands
# =============================================================================


def run_ejabberdctl(ejabberd_ctl: str, *args) -> tuple[bool, str]:
    """Run an ejabberdctl command via SSH or locally."""
    if ejabberd_ctl.startswith("ssh "):
        parts = ejabberd_ctl.split(maxsplit=2)
        remote_cmd = parts[2] + " " + " ".join(shlex.quote(a) for a in args)
        cmd = ["ssh", parts[1], remote_cmd]
    else:
        cmd = ejabberd_ctl.split() + list(args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        _log.warning("ejabberdctl timed out after 30s: %s", cmd)
        return False, "command timed out"
    except FileNotFoundError as e:
        _log.warning("ejabberdctl binary not found: %s", e)
        return False, str(e)
    except OSError as e:
        _log.warning("ejabberdctl OS error: %s", e)
        return False, str(e)
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output
