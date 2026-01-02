#!/bin/bash
# Complete startup script for LLIX - starts Qdrant and Streamlit

echo "🚀 LLIX Complete Startup"
echo "========================"
echo ""

# Navigate to LLIX directory
cd /home/gerrit/Antigravity/LLIX

# Check if Qdrant is already running
if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "✅ Qdrant is already running"
else
    echo "📦 Starting Qdrant container..."
    podman run -d \
        --name qdrant \
        -p 6333:6333 \
        -p 6334:6334 \
        -v qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant:latest
    
    echo "⏳ Waiting for Qdrant to be ready..."
    sleep 5
    
    if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
        echo "✅ Qdrant started successfully"
    else
        echo "❌ Failed to start Qdrant"
        exit 1
    fi
fi

echo ""
echo "🌐 Starting Streamlit UI..."
echo ""

# Run the start script
bash start.sh
