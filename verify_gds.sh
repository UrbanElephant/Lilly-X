#!/bin/bash
# Neo4j GDS Plugin Verification Script
# This script verifies that the Graph Data Science plugin is properly installed and accessible

echo "=================================================="
echo "  Neo4j GDS Plugin Verification"
echo "=================================================="
echo ""

# Check if Neo4j container is running
echo "1️⃣ Checking Neo4j container status..."
if podman ps | grep -q neo4j; then
    echo "   ✅ Neo4j container is running"
else
    echo "   ❌ Neo4j container is NOT running!"
    echo "   Run: podman-compose up -d neo4j"
    exit 1
fi

echo ""
echo "2️⃣ Checking Neo4j connectivity..."
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "   ✅ Neo4j HTTP endpoint is accessible"
else
    echo "   ❌ Cannot reach Neo4j at http://localhost:7474"
    echo "   Wait a few seconds for Neo4j to start, then retry"
    exit 1
fi

echo ""
echo "3️⃣ Verifying GDS plugin installation..."
echo "   Testing via Python..."

python3 - <<'EOF'
import sys
from src.graph_database import get_neo4j_driver

try:
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # Check for GDS procedures
        result = session.run(
            "CALL dbms.procedures() YIELD name "
            "WHERE name STARTS WITH 'gds' "
            "RETURN count(name) as gds_count"
        )
        gds_count = result.single()["gds_count"]
        
        if gds_count > 0:
            print(f"   ✅ GDS plugin is installed ({gds_count} procedures available)")
            
            # List some key GDS procedures
            result = session.run(
                "CALL dbms.procedures() YIELD name "
                "WHERE name STARTS WITH 'gds.leiden' OR name STARTS WITH 'gds.louvain' "
                "RETURN name ORDER BY name LIMIT 10"
            )
            
            procedures = [record["name"] for record in result]
            
            if procedures:
                print()
                print("   Key GDS procedures found:")
                for proc in procedures:
                    print(f"     - {proc}")
                
                print()
                print("   ✅ Community detection algorithms are available!")
                sys.exit(0)
            else:
                print()
                print("   ⚠️ GDS is installed but Leiden/Louvain not found")
                print("      This might be a version mismatch")
                sys.exit(1)
        else:
            print("   ❌ GDS plugin is NOT installed!")
            print()
            print("   TROUBLESHOOTING:")
            print("   1. Check compose.yaml has: NEO4J_PLUGINS=[\"apoc\",\"graph-data-science\"]")
            print("   2. Restart Neo4j: podman-compose restart neo4j")
            print("   3. Check logs: podman logs neo4j")
            sys.exit(1)
            
except Exception as e:
    print(f"   ❌ Error checking GDS: {e}")
    print()
    print("   TROUBLESHOOTING:")
    print("   - Ensure Neo4j is fully started (wait 30-60 seconds)")
    print("   - Check credentials in .env (NEO4J_USER, NEO4J_PASSWORD)")
    print("   - Verify connection: podman logs neo4j")
    sys.exit(1)
EOF

GDS_STATUS=$?

echo ""
echo "=================================================="
if [ $GDS_STATUS -eq 0 ]; then
    echo "  ✅ GDS VERIFICATION SUCCESSFUL"
    echo "=================================================="
    echo ""
    echo "You can now run community detection:"
    echo "  python3 run_community_summarization.py"
else
    echo "  ❌ GDS VERIFICATION FAILED"
    echo "=================================================="
    echo ""
    echo "See troubleshooting steps above"
fi
echo ""

exit $GDS_STATUS
