#!/bin/bash
# Start the XMPP-OpenCode bridge

systemctl --user start xmpp-opencode-bridge.service
systemctl --user status xmpp-opencode-bridge.service --no-pager

echo ""
echo "Commands:"
echo "  Logs:    ./logs.sh"
echo "  Stop:    ./stop.sh"
echo "  Status:  systemctl --user status xmpp-opencode-bridge"
