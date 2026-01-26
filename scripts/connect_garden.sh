#!/bin/bash
set -e
CONTAINER="garden-production"

echo "🛑 Stopping disconnected container..."
podman stop $CONTAINER 2>/dev/null || true
podman rm $CONTAINER 2>/dev/null || true

echo "🔍 Detecting Model Storage..."
# Prefer host directory if populated, otherwise fallback to named volume
if [ -d "$HOME/.ollama/models" ]; then
    echo "   ✅ Found models in $HOME/.ollama"
    VOLUME_MAP="-v $HOME/.ollama:/root/.ollama:Z"
else
    echo "   ⚠️  Models not found in Home. Using 'ollama_data' volume."
    VOLUME_MAP="-v ollama_data:/root/.ollama"
fi

echo "🚀 Starting Garden with TURBO settings..."
podman run -d \
  --name $CONTAINER \
  --restart always \
  --network host \
  --security-opt label=disable \
  --device /dev/kfd --device /dev/dri \
  $VOLUME_MAP \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e HSA_ENABLE_SDMA=0 \
  -e OLLAMA_NUM_PARALLEL=4 \
  -e OLLAMA_MAX_LOADED_MODELS=2 \
  ollama/ollama:latest

echo "✅ Container Connected!"
echo "👉 Verify models with: curl http://localhost:11434/api/tags"
