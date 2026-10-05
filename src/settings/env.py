"""
Environment loading and XMPP configuration.
"""

import os
from pathlib import Path

from src.settings.dispatchers import _load_dispatchers_config

# =============================================================================
# Environment Loading
# =============================================================================


def load_env(env_path: Path | None = None) -> None:
    """Load .env file into os.environ. Handles quoted values and spaces."""
    if env_path is None:
        env_path = Path(__file__).parents[2] / ".env"

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
