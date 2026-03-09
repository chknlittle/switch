#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Creating data directories..."
mkdir -p "$PROJECT_DIR/data/falkordb"
mkdir -p "$PROJECT_DIR/config/searxng"

echo "==> Starting FalkorDB + SearXNG containers..."
docker compose -f "$PROJECT_DIR/docker-compose.weaver.yml" up -d

echo "==> Waiting for FalkorDB (port 6379)..."
for i in $(seq 1 30); do
    if docker compose -f "$PROJECT_DIR/docker-compose.weaver.yml" exec -T falkordb redis-cli PING 2>/dev/null | grep -q PONG; then
        echo "    FalkorDB is ready."
        break
    fi
    sleep 1
done

echo "==> Waiting for SearXNG (port 8888)..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8888/healthz >/dev/null 2>&1 || curl -sf http://localhost:8888/ >/dev/null 2>&1; then
        echo "    SearXNG is ready."
        break
    fi
    sleep 1
done

echo "==> Installing Python dependencies..."
cd "$PROJECT_DIR"
uv add graphiti-core openai

echo ""
echo "Done! Services:"
echo "  FalkorDB: redis://localhost:6379  (Web UI: http://localhost:3000)"
echo "  SearXNG:  http://localhost:8888"
