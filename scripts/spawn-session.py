#!/usr/bin/env python3
"""
Send a message to oc@ dispatcher to spawn a new session.

Usage:
    spawn-session.py <message>

Example:
    spawn-session.py "Continue moonshot-survival work. Context: ..."
"""

import asyncio
import sys

from src.utils import load_env, get_xmpp_config, BaseXMPPBot

# Load environment
load_env()
cfg = get_xmpp_config()

XMPP_SERVER = cfg["server"]


class SpawnBot(BaseXMPPBot):
    def __init__(self, jid, password, target_jid, message):
        super().__init__(jid, password, recipient=target_jid)
        self.spawn_message = message
        self.add_event_handler("session_start", self.on_start)
        self.add_event_handler("failed_auth", self.on_failed_auth)

    def on_failed_auth(self, event):
        pass  # Slixmpp sometimes fires this even on success

    async def on_start(self, event):
        self.send_presence()
        self.send_reply(self.spawn_message)
        print(f"Sent to {self.recipient}")
        print(f"Message: {self.spawn_message[:100]}...")

        await asyncio.sleep(2)
        self.disconnect()


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    message = " ".join(sys.argv[1:])

    dispatcher_jid = cfg["dispatchers"]["oc"]["jid"]
    dispatcher_password = cfg["dispatchers"]["oc"]["password"]

    if not dispatcher_password:
        print("Error: XMPP_PASSWORD not set in .env")
        sys.exit(1)

    bot = SpawnBot(dispatcher_jid, dispatcher_password, dispatcher_jid, message)
    bot.connect_to_server(XMPP_SERVER)

    try:
        await asyncio.wait_for(bot.disconnected, timeout=10)
        print("New session should be spawning...")
    except asyncio.TimeoutError:
        print("Timeout")
        bot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
