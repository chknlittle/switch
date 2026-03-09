#!/bin/bash
# Setup script for loom dispatcher (GLM 4.7 Flash on port 8081)

echo "Setting up LOOM dispatcher environment variables..."

# Set default values (override with your actual values)
export LOOM_JID="loom@localhost"
export LOOM_PASSWORD="buttpass"

# Optional: Override model ID if needed
# export LOOM_MODEL_ID="opencode/glm-4.7"

echo "LOOM_JID: $LOOM_JID"
echo "LOOM_PASSWORD: $LOOM_PASSWORD"
echo ""
echo "Make sure you have the GLM 4.7 Flash model running on port 8081."
echo "You can test it with: curl http://127.0.0.1:8081/v1/models"
