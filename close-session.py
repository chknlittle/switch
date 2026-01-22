#!/usr/bin/env python3
"""
Close a session: send XMPP goodbye, mark closed in DB, kill tmux & XMPP account.

Usage:
    close-session.py <session-name> [message]

Example:
    close-session.py old-session-name
    close-session.py old-session-name "Session archived. Bye!"
"""

import asyncio
import sqlite3
import subprocess
import sys
from pathlib import Path

from utils import load_env, get_xmpp_config, run_ejabberdctl, BaseXMPPBot

# Load environment
load_env()
cfg = get_xmpp_config()

XMPP_SERVER = cfg["server"]
XMPP_DOMAIN = cfg["domain"]
XMPP_RECIPIENT = cfg["recipient"]
EJABBERD_CTL = cfg["ejabberd_ctl"]
DB_PATH = Path(__file__).parent / "sessions.db"


class ClosureBot(BaseXMPPBot):
    def __init__(self, jid, password, message):
        super().__init__(jid, password, recipient=XMPP_RECIPIENT)
        self.closure_message = message
        self.sent = False
        self.add_event_handler("session_start", self.on_start)

    async def on_start(self, event):
        self.send_presence()
        self.send_reply(self.closure_message)
        self.sent = True
        print(f"Sent: {self.closure_message}")
        await asyncio.sleep(1)
        self.disconnect()


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    session_name = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else "Session closed. Goodbye!"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT xmpp_jid, xmpp_password, status FROM sessions WHERE name = ?",
        (session_name,),
    ).fetchone()

    if not row:
        print(f"Session '{session_name}' not found in database")
        result = subprocess.run(
            ["tmux", "kill-session", "-t", session_name], capture_output=True
        )
        if result.returncode == 0:
            print(f"Killed orphan tmux session: {session_name}")
        conn.close()
        return

    if row["status"] == "closed":
        print(f"Session '{session_name}' already marked closed")
        conn.close()
        return

    print(f"Closing session: {session_name}")

    bot = ClosureBot(row["xmpp_jid"], row["xmpp_password"], message)
    try:
        bot.connect_to_server(XMPP_SERVER)
        await asyncio.wait_for(bot.disconnected, timeout=5)
    except asyncio.TimeoutError:
        bot.disconnect()
    except Exception as e:
        print(f"XMPP send failed: {e}")

    username = row["xmpp_jid"].split("@")[0]
    success, _ = run_ejabberdctl(EJABBERD_CTL, "unregister", username, XMPP_DOMAIN)
    if success:
        print(f"Deleted XMPP account: {username}")

    result = subprocess.run(
        ["tmux", "kill-session", "-t", session_name], capture_output=True
    )
    if result.returncode == 0:
        print(f"Killed tmux: {session_name}")

    conn.execute(
        "UPDATE sessions SET status = 'closed' WHERE name = ?", (session_name,)
    )
    conn.commit()
    conn.close()

    print(f"Session '{session_name}' closed")


if __name__ == "__main__":
    asyncio.run(main())
