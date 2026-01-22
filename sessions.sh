#!/bin/bash
# List or manage OpenCode sessions

cd "$(dirname "$0")"

DB="sessions.db"

case "${1:-list}" in
    list)
        echo "=== OpenCode Sessions ==="
        if [ -f "$DB" ]; then
            sqlite3 -header -column "$DB" \
                "SELECT name, xmpp_jid, datetime(last_active) as last_active FROM sessions ORDER BY last_active DESC"
        else
            echo "No sessions yet."
        fi
        echo ""
        echo "=== tmux Sessions ==="
        tmux list-sessions 2>/dev/null | grep -v "xmpp-opencode-bridge" || echo "None"
        ;;

    kill)
        if [ -z "$2" ]; then
            echo "Usage: $0 kill <session-name>"
            exit 1
        fi
        NAME="$2"

        JID=$(sqlite3 "$DB" "SELECT xmpp_jid FROM sessions WHERE name='$NAME'" 2>/dev/null)
        if [ -z "$JID" ]; then
            echo "Session not found: $NAME"
            exit 1
        fi

        tmux kill-session -t "$NAME" 2>/dev/null

        USERNAME=$(echo "$JID" | cut -d@ -f1)
        source .env 2>/dev/null
        $EJABBERD_CTL unregister "$USERNAME" "your.xmpp.server" 2>/dev/null

        sqlite3 "$DB" "DELETE FROM sessions WHERE name='$NAME'"

        echo "Killed session: $NAME"
        ;;

    clean)
        echo "Cleaning all sessions..."
        sqlite3 "$DB" "SELECT name FROM sessions" 2>/dev/null | while read NAME; do
            "$0" kill "$NAME"
        done
        echo "Done."
        ;;

    *)
        echo "Usage: $0 [list|kill <name>|clean]"
        ;;
esac
