"""Performance Test: Ollama Concurrency Verification

Tests OLLAMA_NUM_PARALLEL=4 configuration by running concurrent queries
and verifying parallel execution improves throughput.

Expected: 4 concurrent queries complete in < 3x baseline time
(proving parallel execution vs sequential).
"""

import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_engine import RAGEngine


async def run_single_query(engine: RAGEngine, query: str, query_id: int) -> float:
    """Run a single query and measure its execution time.
    
    Args:
        engine: RAG engine instance
        query: Query string
        query_id: Identifier for logging
        
    Returns:
        Execution time in seconds
    """
    start = time.time()
    try:
        print(f"[Query {query_id}] Starting...")
        response = await engine.aquery(query, top_n=3)
        elapsed = time.time() - start
        print(f"[Query {query_id}] Completed in {elapsed:.2f}s")
        return elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"[Query {query_id}] Error after {elapsed:.2f}s: {e}")
        return elapsed


async def test_concurrency():
    """Test Ollama concurrency by comparing sequential vs parallel query execution."""
    
    print("=" * 80)
    print("OLLAMA CONCURRENCY TEST")
    print("=" * 80)
    print("\nInitializing RAG Engine...")
    
    try:
        # Initialize engine
        engine = RAGEngine(
            enable_decomposition=False,  # Disable for simpler testing
            enable_hyde=False,
            enable_rewriting=False,
            verbose=False,
        )
        print("✅ RAG Engine initialized\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG Engine: {e}")
        print("\nPlease ensure:")
        print("  1. Qdrant is running (http://localhost:6333)")
        print("  2. Ollama is running (http://localhost:11434)")
        print("  3. Knowledge base has been ingested")
        return
    
    # Test query - complex enough to take some time
    test_query = "Explain the architecture and key components of a RAG system"
    
    # =========================================================================
    # Phase 1: Baseline (Single Query)
    # =========================================================================
    print("-" * 80)
    print("PHASE 1: Baseline - Single Query")
    print("-" * 80)
    
    baseline_time = await run_single_query(engine, test_query, 1)
    
    print(f"\n📊 Baseline Time: {baseline_time:.2f}s\n")
    
    # =========================================================================
    # Phase 2: Concurrent Queries (4 parallel)
    # =========================================================================
    print("-" * 80)
    print("PHASE 2: Concurrent Execution - 4 Parallel Queries")
    print("-" * 80)
    
    concurrent_start = time.time()
    
    # Run 4 queries concurrently
    tasks = [
        run_single_query(engine, test_query, i)
        for i in range(1, 5)
    ]
    
    query_times = await asyncio.gather(*tasks)
    concurrent_total = time.time() - concurrent_start
    
    print(f"\n📊 Concurrent Execution Results:")
    print(f"   Total Time: {concurrent_total:.2f}s")
    print(f"   Individual Query Times: {[f'{t:.2f}s' for t in query_times]}")
    print(f"   Average Query Time: {sum(query_times) / len(query_times):.2f}s")
    
    # =========================================================================
    # Phase 3: Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    sequential_estimate = baseline_time * 4
    speedup = sequential_estimate / concurrent_total
    success_threshold = baseline_time * 3  # Should be < 3x for parallel success
    
    print(f"\nSequential Estimate: {sequential_estimate:.2f}s (4 × {baseline_time:.2f}s)")
    print(f"Concurrent Actual:   {concurrent_total:.2f}s")
    print(f"Speedup Factor:      {speedup:.2f}x")
    print(f"Success Threshold:   < {success_threshold:.2f}s")
    
    print("\n" + "=" * 80)
    
    if concurrent_total < success_threshold:
        print("✅ SUCCESS: Ollama is processing requests in parallel!")
        print(f"   Concurrent execution is {speedup:.2f}x faster than sequential")
        print(f"   OLLAMA_NUM_PARALLEL=4 is working correctly")
        return True
    else:
        print("❌ FAILURE: Requests appear to be processed sequentially")
        print(f"   Expected: < {success_threshold:.2f}s")
        print(f"   Actual:   {concurrent_total:.2f}s")
        print("\nPossible issues:")
        print("  1. OLLAMA_NUM_PARALLEL not set or set to 1")
        print("  2. Model not loaded in VRAM (loading on each request)")
        print("  3. System resource constraints")
        print("\nVerify Ollama configuration:")
        print("  podman inspect garden-production | grep OLLAMA_NUM_PARALLEL")
        return False


def main():
    """Main entry point."""
    try:
        result = asyncio.run(test_concurrency())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
