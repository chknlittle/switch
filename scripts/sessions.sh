#!/bin/bash
# List or manage Switch sessions

cd "$(dirname "$0")/.."

DB="sessions.db"

has_sqlite() {
    command -v sqlite3 >/dev/null 2>&1
}

require_sqlite() {
    if ! has_sqlite; then
        echo "sqlite3 not found; install it to manage sessions.db"
        exit 1
    fi
}

case "${1:-list}" in
    list)
        echo "=== Sessions ==="
        if [ -f "$DB" ]; then
            if has_sqlite; then
                sqlite3 -header -column "$DB" \
                    "SELECT name, xmpp_jid, datetime(last_active) as last_active FROM sessions ORDER BY last_active DESC"
            else
                echo "sqlite3 not found; install it to view sessions.db"
            fi
        else
            echo "No sessions yet."
        fi
        echo ""
        echo "=== tmux Sessions ==="
        tmux list-sessions 2>/dev/null | grep -v "switch" || echo "None"
        ;;

    kill)
        if [ -z "$2" ]; then
            echo "Usage: $0 kill <session-name>"
            exit 1
        fi
        NAME="$2"

        # Prefer in-bridge kill via dispatcher so the in-memory bot winds down
        # and doesn't immediately reconnect.
        DISPATCHER=${SWITCH_DEFAULT_DISPATCHER:-oc-gpt}
        if scripts/spawn-session --dispatcher "$DISPATCHER" "/kill $NAME" >/dev/null 2>&1; then
            echo "Requested kill via dispatcher ($DISPATCHER): $NAME"
            exit 0
        fi

        echo "Dispatcher kill failed; falling back to offline cleanup."
        require_sqlite

        JID=$(sqlite3 "$DB" "SELECT xmpp_jid FROM sessions WHERE name='$NAME'" 2>/dev/null)
        if [ -z "$JID" ]; then
            echo "Session not found: $NAME"
            exit 1
        fi

        tmux kill-session -t "$NAME" 2>/dev/null

        USERNAME=$(echo "$JID" | cut -d@ -f1)
        source .env 2>/dev/null
        $EJABBERD_CTL unregister "$USERNAME" "$XMPP_DOMAIN" 2>/dev/null

        # Archive semantics: don't delete session history.
        sqlite3 "$DB" "UPDATE sessions SET status='closed' WHERE name='$NAME'"

        echo "Closed session: $NAME"
        ;;

    clean)
        echo "Cleaning all sessions..."
        require_sqlite

        sqlite3 "$DB" "SELECT name FROM sessions" 2>/dev/null | while read NAME; do
            "$0" kill "$NAME"
        done
        echo "Done."
        ;;

    *)
        echo "Usage: $0 [list|kill <name>|clean]"
        ;;
esac
