#!/bin/bash
# Verify Phase 2: Dual-Store Architecture Implementation

echo "=========================================="
echo "Phase 2 Implementation Verification"
echo "=========================================="
echo ""

echo "1. Checking compose.yaml for Neo4j service..."
if grep -q "neo4j" compose.yaml && grep -q "7474:7474" compose.yaml; then
    echo "   ✅ Neo4j service configured"
else
    echo "   ❌ Neo4j service not found"
fi

echo ""
echo "2. Checking .gitignore for Neo4j data..."
if grep -q "neo4j_data/" .gitignore; then
    echo "   ✅ neo4j_data/ in .gitignore"
else
    echo "   ❌ neo4j_data/ not found in .gitignore"
fi

echo ""
echo "3. Checking .env.template for Neo4j credentials..."
if grep -q "NEO4J_URL" .env.template && grep -q "NEO4J_PASSWORD" .env.template; then
    echo "   ✅ Neo4j configuration in .env.template"
else
    echo "   ❌ Neo4j configuration missing"
fi

echo ""
echo "4. Checking src/config.py for Neo4j settings..."
if grep -q "neo4j_url" src/config.py && grep -q "neo4j_password" src/config.py; then
    echo "   ✅ Neo4j settings in config.py"
else
    echo "   ❌ Neo4j settings missing"
fi

echo ""
echo "5. Checking requirements.txt for Neo4j dependencies..."
if grep -q "neo4j" requirements.txt && grep -q "llama-index-graph-stores-neo4j" requirements.txt; then
    echo "   ✅ Neo4j dependencies added"
else
    echo "   ❌ Neo4j dependencies missing"
fi

echo ""
echo "6. Checking src/graph_database.py exists..."
if [ -f "src/graph_database.py" ]; then
    echo "   ✅ src/graph_database.py created"
else
    echo "   ❌ src/graph_database.py not found"
fi

echo ""
echo "7. Checking src/graph_schema.py exists..."
if [ -f "src/graph_schema.py" ]; then
    echo "   ✅ src/graph_schema.py created"
else
    echo "   ❌ src/graph_schema.py not found"
fi

echo ""
echo "8. Checking start_all.sh for Neo4j health checks..."
if grep -q "NEO4J_HEALTH_URL" start_all.sh; then
    echo "   ✅ Neo4j health checks in start_all.sh"
else
    echo "   ❌ Neo4j health checks missing"
fi

echo ""
echo "=========================================="
echo "✅ Phase 2 Implementation Verification Complete"
echo "=========================================="
echo ""
echo "To test Neo4j connectivity:"
echo "  1. Ensure Neo4j container is running: podman ps | grep neo4j"
echo "  2. Wait ~30s for initialization"
echo "  3. Run: python -m src.graph_database"
echo ""
