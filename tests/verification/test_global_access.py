#!/usr/bin/env python3
"""Simple test to check if global_search is accessible."""

import sys
import traceback

print("Testing Global Search Access...")

try:
    print("1. Importing RAGEngine...")
    from src.rag_engine import RAGEngine
    print("   ✓ RAGEngine imported")
    
    print("2. Checking for global_search method...")
    if hasattr(RAGEngine, 'global_search'):
        print("   ✓ global_search method exists")
    else:
        print("   ✗ global_search method NOT FOUND!")
        sys.exit(1)
    
    print("3. Importing GraphOperations...")
    from src.graph_ops import GraphOperations
    print("   ✓ GraphOperations imported")
    
    print("4. Checking community count...")
    ops = GraphOperations()
    with ops._driver.session() as session:
        count = session.run("MATCH (c:Community) RETURN count(c) as n").single()['n']
    print(f"   ✓ Found {count} communities")
    
    if count == 0:
        print("   ⚠ WARNING: No communities found!")
        print("   Run: python3 run_community_summarization.py")
        sys.exit(1)
    
    print("\n✅ All checks passed! Ready to test global_search")
    print("\nTo test manually:")
    print(">>> from src.rag_engine import RAGEngine")
    print(">>> engine = RAGEngine()")
    print(">>> response = engine.global_search('Summarize the main themes')")
    print(">>> print(response)")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
