#!/bin/bash
# Tail the session output log in tmux
# Usage: session-shell.sh <session-name>

SESSION_NAME="$1"
LOG_DIR="$HOME/xmpp-opencode-bridge/output"
LOG_FILE="$LOG_DIR/$SESSION_NAME.log"

if [ -z "$SESSION_NAME" ]; then
    echo "Usage: $0 <session-name>"
    exit 1
fi

mkdir -p "$LOG_DIR"

if [ ! -f "$LOG_FILE" ]; then
    echo "No output yet for $SESSION_NAME."
    echo "Waiting for log file: $LOG_FILE"
    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done
fi

echo "=== XMPP-OpenCode session: $SESSION_NAME ==="
echo "Tailing: $LOG_FILE"
echo "(Ctrl+C to exit tmux pane)"
echo ""

exec tail -n 200 -f "$LOG_FILE"
