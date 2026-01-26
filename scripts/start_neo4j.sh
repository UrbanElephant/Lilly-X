#!/bin/bash
set -e
CONTAINER="garden-neo4j"
DATA_DIR="$HOME/neo4j_data"

# 1. Prepare Storage
mkdir -p "$DATA_DIR"

# 2. Cleanup Old Container
echo "🧹 Cleaning up old graph container..."
podman stop $CONTAINER 2>/dev/null || true
podman rm $CONTAINER 2>/dev/null || true

# 3. Start Neo4j
# Matches src/config.py credentials (neo4j/password)
echo "🚀 Starting Neo4j 5.15..."
podman run -d \
    --name $CONTAINER \
    --restart always \
    --network host \
    --security-opt label=disable \
    -v "$DATA_DIR":/data:Z \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_dbms_memory_heap_initial__size=1G \
    -e NEO4J_dbms_memory_heap_max__size=2G \
    neo4j:5.15

echo "⏳ Waiting for Bolt port (7687)..."
# Simple wait loop
for i in {1..30}; do
    if nc -z localhost 7687 2>/dev/null; then
        echo "✅ Neo4j is UP!"
        exit 0
    fi
    sleep 1
done

echo "⚠️  Neo4j started but port 7687 is not yet reachable. It might take a moment."
