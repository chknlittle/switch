#!/bin/bash
# Run the XMPP-OpenCode bridge

cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Check for password
if [ -z "$XMPP_PASSWORD" ]; then
    echo "Error: XMPP_PASSWORD not set"
    echo "Create .env file with XMPP_PASSWORD=yourpassword"
    exit 1
fi

# Check for opencode
if ! command -v opencode &> /dev/null; then
    if [ -x "$HOME/.opencode/bin/opencode" ]; then
        export PATH="$HOME/.opencode/bin:$PATH"
    fi
fi

if ! command -v opencode &> /dev/null; then
    echo "Error: opencode command not found"
    exit 1
fi

# Check for claude
if ! command -v claude &> /dev/null; then
    echo "Error: claude command not found"
    exit 1
fi

# Check for tmux
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux not found"
    exit 1
fi

# Install deps if needed
if ! python3 -c "import slixmpp" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting XMPP-OpenCode bridge..."
python3 bridge.py
