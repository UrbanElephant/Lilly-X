#!/bin/bash
# Robust, idempotent startup script for LLIX
# Starts Qdrant (via Podman) and Streamlit UI

echo "🚀 LLIX Complete Startup"
echo "========================"
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Define Qdrant health URL (force IPv4)
QDRANT_HEALTH_URL="http://127.0.0.1:6333/healthz"

# === STEP 1: Check if Qdrant is already healthy ===
echo "🔍 Checking Qdrant health..."
if curl -s "$QDRANT_HEALTH_URL" > /dev/null 2>&1; then
    echo "✅ Qdrant is already running and healthy"
else
    HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" "$QDRANT_HEALTH_URL")
    echo "⚠️  Qdrant health check failed (HTTP status: $HTTP_STATUS)"
    
    # === STEP 2: Check if container exists ===
    echo "🔍 Checking if Qdrant container exists..."
    if podman ps -a --format "{{.Names}}" | grep -q "^qdrant$"; then
        echo "📦 Container 'qdrant' exists, attempting to start it..."
        podman start qdrant
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to start existing container"
            echo "💡 Try: podman rm qdrant (to remove and recreate)"
            exit 1
        fi
    else
        echo "📦 Creating new Qdrant container..."
        podman run -d \
            --name qdrant \
            -p 6333:6333 \
            -p 6334:6334 \
            -v qdrant_storage:/qdrant/storage:z \
            qdrant/qdrant:latest
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create Qdrant container"
            exit 1
        fi
    fi
    
    # === STEP 3: Wait for Qdrant to become healthy ===
    echo "⏳ Waiting for Qdrant to become healthy..."
    MAX_RETRIES=10
    RETRY_COUNT=0
    SLEEP_TIME=2
    HEALTH_OK=false
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" "$QDRANT_HEALTH_URL")
        
        if [ "$HTTP_STATUS" = "200" ]; then
            echo "✅ Qdrant is now healthy (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES, HTTP 200)"
            HEALTH_OK=true
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "   Attempt $RETRY_COUNT/$MAX_RETRIES failed (HTTP $HTTP_STATUS), waiting ${SLEEP_TIME}s..."
            sleep $SLEEP_TIME
        fi
    done
    
    # === STEP 4: Soft fail logic ===
    if [ "$HEALTH_OK" = "false" ]; then
        HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" "$QDRANT_HEALTH_URL")
        echo "⚠️  Health check still failing after $MAX_RETRIES attempts (HTTP $HTTP_STATUS)"
        
        # Check if container is actually running
        CONTAINER_RUNNING=$(podman inspect -f '{{.State.Running}}' qdrant 2>/dev/null)
        
        if [ "$CONTAINER_RUNNING" = "true" ]; then
            echo "⚠️  Health check failed but container is running. Proceeding..."
            echo "💡 This may indicate a network binding issue (IPv4/IPv6)"
        else
            echo "❌ Qdrant container is not running"
            echo "💡 Check logs with: podman logs qdrant"
            exit 1
        fi
    fi
fi

echo ""
echo "=" * 50
echo "🔵 Neo4j Graph Database Health Check"
echo "=" * 50
echo ""

# Define Neo4j health URL (force IPv4)
NEO4J_HEALTH_URL="http://127.0.0.1:7474"

# === STEP 1: Check if Neo4j is already healthy ===
echo "🔍 Checking Neo4j health..."
if curl -s "$NEO4J_HEALTH_URL" > /dev/null 2>&1; then
    echo "✅ Neo4j is already running and healthy"
else
    HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" "$NEO4J_HEALTH_URL")
    echo "⚠️  Neo4j health check failed (HTTP status: $HTTP_STATUS)"
    
    # === STEP 2: Check if container exists ===
    echo "🔍 Checking if Neo4j container exists..."
    if podman ps -a --format "{{.Names}}" | grep -q "^neo4j$"; then
        echo "📦 Container 'neo4j' exists, attempting to start it..."
        podman start neo4j
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to start existing container"
            echo "💡 Try: podman rm neo4j (to remove and recreate)"
            exit 1
        fi
    else
        echo "📦 Creating new Neo4j container..."
        podman run -d \
            --name neo4j \
            -p 7474:7474 \
            -p 7687:7687 \
            -v ./neo4j_data:/data:z \
            -e NEO4J_AUTH=neo4j/password \
            -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
            -e NEO4J_dbms_security_procedures_unrestricted='apoc.*,gds.*' \
            -e NEO4J_dbms_memory_heap_max__size=2G \
            neo4j:5.15
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create Neo4j container"
            exit 1
        fi
    fi
    
    # === STEP 3: Wait for Neo4j to become healthy ===
    echo "⏳ Waiting for Neo4j to become healthy (this may take up to 60s)..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    SLEEP_TIME=2
    HEALTH_OK=false
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" "$NEO4J_HEALTH_URL")
        
        if [ "$HTTP_STATUS" = "200" ]; then
            echo "✅ Neo4j is now healthy (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES, HTTP 200)"
            HEALTH_OK=true
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "   Attempt $RETRY_COUNT/$MAX_RETRIES failed (HTTP $HTTP_STATUS), waiting ${SLEEP_TIME}s..."
            sleep $SLEEP_TIME
        fi
    done
    
    # === STEP 4: Soft fail logic ===
    if [ "$HEALTH_OK" = "false" ]; then
        HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" "$NEO4J_HEALTH_URL")
        echo "⚠️  Health check still failing after $MAX_RETRIES attempts (HTTP $HTTP_STATUS)"
        
        # Check if container is actually running
        CONTAINER_RUNNING=$(podman inspect -f '{{.State.Running}}' neo4j 2>/dev/null)
        
        if [ "$CONTAINER_RUNNING" = "true" ]; then
            echo "⚠️  Health check failed but container is running. Proceeding..."
            echo "💡 Neo4j may still be initializing. Check logs: podman logs neo4j"
        else
            echo "❌ Neo4j container is not running"
            echo "💡 Check logs with: podman logs neo4j"
            exit 1
        fi
    fi
fi

echo ""
echo "🌐 Starting Streamlit UI..."
echo ""

# === STEP 5: Launch Streamlit ===
bash start.sh
