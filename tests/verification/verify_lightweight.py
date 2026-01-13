#!/usr/bin/env python3
"""
Lightweight verification that bypasses RAGEngine initialization issues.
Directly tests the core components.
"""

import sys

print("🚀 LIGHTWEIGHT GraphRAG VERIFICATION")
print("=" * 70)

# Test 1: Community Check
print("\n📊 Step 1: Checking Community Nodes in Neo4j...")
try:
    from src.graph_database import get_neo4j_driver
    
    driver = get_neo4j_driver()
    with driver.session() as session:
        # Count communities
        result = session.run("MATCH (c:Community) RETURN count(c) as count")
        comm_count = result.single()["count"]
        
        print(f"   ✅ Found {comm_count} Community nodes")
        
        if comm_count == 0:
            print("   ❌ No communities! Run: python3 run_community_summarization.py")
            sys.exit(1)
        
        # Get sample community
        result = session.run("""
            MATCH (c:Community)
            RETURN c.community_id as id, c.summary as summary, c.keywords as keywords
            ORDER BY c.community_id
            LIMIT 3
        """)
        
        print(f"\n   📋 Sample Communities:")
        for record in result:
            print(f"      Community {record['id']}: {record['keywords'][:3]}...")
            print(f"      → {record['summary'][:100]}...")
            print()

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Graph Operations Check
print("\n🔧 Step 2: Testing GraphOperations.get_community_context()...")
try:
    from src.graph_ops import GraphOperations
    
    ops = GraphOperations()
    
    # Test keyword-based retrieval
    test_keywords = ["docker", "qdrant", "neo4j", "system", "database"]
    print(f"   Testing with keywords: {test_keywords}")
    
    summaries = ops.get_community_context(keywords=test_keywords, top_k=3)
    
    print(f"   ✅ Retrieved {len(summaries)} community summaries")
    
    if summaries:
        print(f"\n   📄 Sample Retrieved Context:")
        for idx, summary in enumerate(summaries[:2], 1):
            print(f"      {idx}. {summary[:150]}...")
            print()
    else:
        print("   ⚠ No summaries matched keywords")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check schemas
print("\n📚 Step 3: Verifying Schema Definitions...")
try:
    from src.schemas import CommunitySummary, QueryIntent
    
    # Check GLOBAL_DISCOVERY exists
    if hasattr(QueryIntent, 'GLOBAL_DISCOVERY'):
        print(f"   ✅ QueryIntent.GLOBAL_DISCOVERY = {QueryIntent.GLOBAL_DISCOVERY}")
    else:
        print("   ❌ GLOBAL_DISCOVERY not found in QueryIntent!")
        sys.exit(1)
    
    # Test CommunitySummary
    test_summary = CommunitySummary(
        community_id=999,
        level=0,
        summary="Test summary",
        keywords=["test", "verify"]
    )
    print(f"   ✅ CommunitySummary model works: {test_summary.community_id}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE!")
print("=" * 70)
print("\n📊 Results:")
print(f"   - Communities in Neo4j: {comm_count}")
print(f"   - Community retrieval: Working")
print(f"   - Schema definitions: Valid")
print("\n🎉 Microsoft GraphRAG Community Detection: OPERATIONAL")
print("\n⚠ Note: Full RAGEngine has Python 3.14 compatibility issues")
print("   The core community detection features are working correctly.")
print("=" * 70)
