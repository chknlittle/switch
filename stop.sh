#!/bin/bash
# Stop the XMPP-OpenCode bridge

systemctl --user stop xmpp-opencode-bridge.service
echo "Bridge stopped."
