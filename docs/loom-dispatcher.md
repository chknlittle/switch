# Loom Dispatcher

The `loom` dispatcher is a new XMPP bot that uses the GLM 4.7 Flash model running on port 8081.

## Configuration

The loom dispatcher is configured in:
- `dispatchers.json` - Main dispatcher configuration
- `src/utils.py` - Internal configuration mapping

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOOM_JID` | XMPP JID for the loom bot | `loom@localhost` |
| `LOOM_PASSWORD` | Password for the loom bot | (required) |
| `LOOM_MODEL_ID` | Optional model ID override | (empty) |

## Usage

1. Set the environment variables:
   ```bash
   export LOOM_JID="loom@localhost"
   export LOOM_PASSWORD="buttpass"
   ```

2. Ensure the GLM 4.7 Flash model is running on port 8081:
   ```bash
   curl http://127.0.0.1:8081/v1/models
   ```

3. Start the switch bridge:
   ```bash
   python src/bridge.py
   ```

4. Use the loom dispatcher by sending messages to `loom@localhost`

## Features

- Uses the GLM 4.7 Flash model (port 8081)
- Same interface as other dispatchers (oc, debate, etc.)
- Supports all standard dispatcher commands:
  - `/list` - List active sessions
  - `/recent` - Show recent sessions
  - `/kill <name>` - Kill a session
  - `/new --with <jid> <prompt>` - Create a shared session
  - `/commit <repo>` - Commit and push changes
  - `/ralph <args>` - Run a Ralph loop
  - `/help` - Show help

## Differences from "oc"

- Uses port 8081 instead of 8080
- Configured for GLM 4.7 Flash instead of Qwen 122B
- Can be used alongside the existing "oc" dispatcher

## Setup Script

Run the setup script to configure environment variables:
```bash
./scripts/setup_loom.sh
```
