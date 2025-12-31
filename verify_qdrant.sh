#!/bin/bash
# Qdrant Connection Verification Script

echo "==================================="
echo "Qdrant Connection Test"
echo "==================================="
echo ""

# Check container status
echo "1. Container Status:"
podman ps --filter name=qdrant --format "   {{.Names}}: {{.Status}}"
echo ""

# Check logs for startup
echo "2. Recent Logs:"
podman logs qdrant 2>&1 | tail -3
echo ""

# Test health endpoint
echo "3. Health Check (127.0.0.1:6333/healthz):"
HEALTH=$(curl -s http://127.0.0.1:6333/healthz 2>&1)
if [ $? -eq 0 ]; then
    echo "   ✓ Connected successfully"
    echo "   Response: $HEALTH"
else
    echo "   ✗ Connection failed"
    echo "   Error: $HEALTH"
fi
echo ""

# Test main endpoint
echo "4. Main API (127.0.0.1:6333):"
MAIN=$(curl -s -w "\n   HTTP Status: %{http_code}" http://127.0.0.1:6333 2>&1 | head -2)
echo "$MAIN"
echo ""

# Check collections
echo "5. Collections Endpoint:"
COLLECTIONS=$(curl -s http://127.0.0.1:6333/collections 2>&1)
echo "   $COLLECTIONS"
echo ""

echo "==================================="
echo "Test Complete"
echo "==================================="
