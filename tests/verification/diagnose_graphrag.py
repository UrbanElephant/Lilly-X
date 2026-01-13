#!/usr/bin/env python3
"""
Comprehensive verification of all GraphRAG components.
This script tests each component individually to pinpoint any issues.
"""

import sys
import traceback

print("=" * 70)
print("COMPREHENSIVE GRAPHRAG VERIFICATION")
print("=" * 70)

# Test 1: Check imports
print("\n[1/6] Testing imports...")
try:
    from src.schemas import CommunitySummary, QueryIntent
    print("  ✓ CommunitySummary imported")
    print(f"  ✓ GLOBAL_DISCOVERY = {QueryIntent.GLOBAL_DISCOVERY}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.graph_ops import GraphOperations  
    print("  ✓ GraphOperations imported")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.community_pipeline import CommunitySummarizationPipeline
    print("  ✓ CommunitySummarizationPipeline imported")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from src.rag_engine import RAGEngine
    print("  ✓ RAGEngine imported")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check GraphOperations methods
print("\n[2/6] Checking GraphOperations methods...")
try:
    ops = GraphOperations()
    methods = ['run_community_detection', 'get_nodes_in_community', 
               'store_community_summary', 'get_community_context']
    for method in methods:
        if hasattr(ops, method):
            print(f"  ✓ {method} exists")
        else:
            print(f"  ✗ {method} MISSING!")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check RAGEngine methods
print("\n[3/6] Checking RAGEngine methods...")
try:
    # Check if methods exist without instantiating (avoids long init)
    if hasattr(RAGEngine, 'global_search'):
        print("  ✓ global_search exists")
    else:
        print("  ✗ global_search MISSING!")
        sys.exit(1)
    
    if hasattr(RAGEngine, '_extract_keywords_for_global_search'):
        print("  ✓ _extract_keywords_for_global_search exists")
    else:
        print("  ✗ _extract_keywords_for_global_search MISSING!")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Neo4j Connection
print("\n[4/6] Testing Neo4j connection...")
try:
    from src.graph_database import get_neo4j_driver
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        assert result.single()["test"] == 1
    print("  ✓ Neo4j connection successful")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    print("  Make sure Neo4j container is running: podman ps | grep neo4j")
    sys.exit(1)

# Test 5: Check for entities
print("\n[5/6] Checking for Entity nodes...")
try:
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN count(e) as count")
        count = result.single()["count"]
        print(f"  ✓ Found {count} Entity nodes")
        if count == 0:
            print("  ⚠ WARNING: No entities found. Run: python3 -m src.ingest")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check GDS availability
print("\n[6/6] Checking Neo4j GDS plugin...")
try:
    with driver.session() as session:
        result = session.run(
            "CALL dbms.procedures() YIELD name "
            "WHERE name STARTS WITH 'gds' "
            "RETURN count(name) as count"
        )
        gds_count = result.single()["count"]
        if gds_count > 0:
            print(f"  ✓ GDS plugin available ({gds_count} procedures)")
        else:
            print("  ✗ GDS plugin NOT available")
            print("  Check compose.yaml: NEO4J_PLUGINS=[\"apoc\",\"graph-data-science\"]")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL VERIFICATION TESTS PASSED!")
print("=" * 70)
print("\nYou can now run:")
print("  1. python3 run_community_summarization.py")
print("  2. python3 test_global.py")
print()
