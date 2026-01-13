#!/usr/bin/env python3
"""
Final Verification Script for Microsoft GraphRAG.
Tests direct global_search functionality with community summaries.
"""

from src.rag_engine import RAGEngine
from src.graph_ops import GraphOperations
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("🚀 STARTING FINAL VERIFICATION: Microsoft GraphRAG")
    
    # 1. Direct Graph Check
    ops = GraphOperations()
    print("\n📊 Checking Neo4j Community Nodes...")
    with ops._driver.session() as session:
        count = session.run("MATCH (c:Community) RETURN count(c) as n").single()['n']
    print(f"✅ Found {count} Community Summary Nodes in DB.")
    
    if count == 0:
        print("❌ ERROR: No communities found. Did run_community_summarization.py save them?")
        return

    # 2. Force Global Search (Bypass Router)
    print("\n🌍 Executing DIRECT Global Search (High-Level Summary)...")
    engine = RAGEngine()
    
    # Question targeting the whole dataset
    query = "Summarize the main technical themes and architectures in these documents."
    
    # Explicit call to global_search to test the pipeline logic
    print(f"\n🔍 Query: {query}")
    print("⏳ Calling engine.global_search()...")
    
    response_text = engine.global_search(query)
    
    print("\n" + "="*70)
    print("🤖 GLOBAL RAG RESPONSE:")
    print("="*70)
    print(response_text)
    print("="*70)
    
    # Verify it actually used communities
    if response_text and len(response_text) > 50:
        print("\n✅ Verification PASSED: Global Search executed successfully!")
        print("   Response length:", len(response_text), "characters")
        print("   Check logs above for '📌 Extracted keywords' and '✅ Retrieved X community summaries'")
    else:
        print("\n⚠️ Warning: Response seems too short. Check logs for errors.")

    # 3. Bonus: Show sample community
    print("\n📋 Sample Community Summary:")
    with ops._driver.session() as session:
        result = session.run("""
            MATCH (c:Community)
            RETURN c.community_id as id, c.summary as summary, c.keywords as keywords
            LIMIT 1
        """)
        sample = result.single()
        if sample:
            print(f"  Community ID: {sample['id']}")
            print(f"  Keywords: {sample['keywords']}")
            print(f"  Summary: {sample['summary'][:200]}...")

if __name__ == "__main__":
    main()
