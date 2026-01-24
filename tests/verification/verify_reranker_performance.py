#!/usr/bin/env /usr/bin/python3.12
"""Reranker Performance Verification Script.

Measures baseline CPU performance for reranking to establish metrics
before iGPU optimization. Tests inference time for 50 documents.

Usage:
    python verify_reranker_performance.py
"""

import sys
import time
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def generate_test_documents(n: int = 50) -> List[str]:
    """Generate synthetic test documents.
    
    Args:
        n: Number of documents to generate
        
    Returns:
        List of document texts
    """
    templates = [
        "Vector databases are specialized systems for storing embeddings with similarity search capabilities.",
        "Machine learning models can be fine-tuned on domain-specific data for better performance.",
        "Retrieval Augmented Generation combines information retrieval with language model generation.",
        "Knowledge graphs represent entities and relationships in a structured format.",
        "Semantic search uses embeddings to find documents by meaning rather than keywords.",
        "Transformers use self-attention mechanisms to process sequential data effectively.",
        "RAG systems retrieve relevant context before generating responses for factual accuracy.",
        "Cross-encoders provide more accurate relevance scoring than bi-encoders but are slower.",
        "BM25 is a classical keyword-based ranking function widely used in information retrieval.",
        "Graph neural networks can learn representations of nodes and edges in graph structures.",
    ]
    
    docs = []
    for i in range(n):
        template_idx = i % len(templates)
        doc = f"Document {i+1}: {templates[template_idx]} "
        doc += f"Additional context for document {i+1} with more details about the topic."
        docs.append(doc)
    
    return docs


def benchmark_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    num_docs: int = 50,
    top_n: int = 10,
    device: str = "cpu",
) -> dict:
    """Benchmark reranker performance.
    
    Args:
        model_name: HuggingFace model identifier
        num_docs: Number of documents to rerank
        top_n: Number of top results to return
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with benchmark results
    """
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    
    print(f"\n{'=' * 80}")
    print(f"ReRanker Performance Benchmark")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Documents: {num_docs}")
    print(f"  Top-N: {top_n}")
    print(f"  Device: {device}")
    
    # Generate test documents
    print(f"\n{'─' * 80}")
    print("Generating test documents...")
    print(f"{'─' * 80}")
    docs = generate_test_documents(num_docs)
    print(f"✓ Generated {len(docs)} test documents")
    
    # Create nodes
    nodes = [
        NodeWithScore(
            node=TextNode(text=doc, id_=f"node_{i}"),
            score=1.0 - (i * 0.01)  # Decreasing scores
        )
        for i, doc in enumerate(docs)
    ]
    
    # Test query
    query = "What is Retrieval Augmented Generation and how does it work?"
    
    # Initialize reranker
    print(f"\n{'─' * 80}")
    print("Initializing ReRanker model...")
    print(f"{'─' * 80}")
    
    init_start = time.time()
    
    try:
        from advanced_rag.rerank import ReRanker
        
        reranker = ReRanker(
            model=model_name,
            top_n=top_n,
            device=device,
        )
        
        init_time = time.time() - init_start
        print(f"✓ Model initialized in {init_time:.2f}s")
        
        # Check if reranker is ready
        info = reranker.get_model_info()
        if not info['initialized']:
            return {
                "success": False,
                "error": "Reranker failed to initialize",
                "init_time": init_time,
            }
        
    except Exception as e:
        init_time = time.time() - init_start
        print(f"✗ Failed to initialize reranker: {e}")
        return {
            "success": False,
            "error": str(e),
            "init_time": init_time,
        }
    
    # Warm-up run (not timed)
    print(f"\n{'─' * 80}")
    print("Running warm-up pass...")
    print(f"{'─' * 80}")
    
    try:
        _ = reranker.rerank(nodes[:5], query, top_n=3)
        print("✓ Warm-up completed")
    except Exception as e:
        print(f"✗ Warm-up failed: {e}")
        return {
            "success": False,
            "error": f"Warm-up failed: {e}",
            "init_time": init_time,
        }
    
    # Benchmark runs
    print(f"\n{'─' * 80}")
    print(f"Running benchmark ({num_docs} documents)...")
    print(f"{'─' * 80}")
    
    num_runs = 3
    inference_times = []
    
    for run in range(num_runs):
        print(f"\n  Run {run + 1}/{num_runs}...", end=" ", flush=True)
        
        start_time = time.time()
        
        try:
            results = reranker.rerank(nodes, query, top_n=top_n)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            print(f"✓ {inference_time:.3f}s ({len(results)} results)")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            return {
                "success": False,
                "error": f"Inference failed: {e}",
                "init_time": init_time,
            }
    
    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    
    docs_per_sec = num_docs / avg_time
    
    results = {
        "success": True,
        "model": model_name,
        "device": device,
        "num_documents": num_docs,
        "top_n": top_n,
        "init_time": init_time,
        "inference_times": inference_times,
        "avg_inference_time": avg_time,
        "min_inference_time": min_time,
        "max_inference_time": max_time,
        "throughput_docs_per_sec": docs_per_sec,
    }
    
    return results


def print_results(results: dict):
    """Print benchmark results in a formatted table.
    
    Args:
        results: Results dictionary from benchmark_reranker
    """
    if not results["success"]:
        print(f"\n{'=' * 80}")
        print("❌ BENCHMARK FAILED")
        print(f"{'=' * 80}")
        print(f"\nError: {results.get('error', 'Unknown error')}")
        print(f"Initialization Time: {results.get('init_time', 0):.2f}s")
        return
    
    print(f"\n{'=' * 80}")
    print("✅ BENCHMARK RESULTS")
    print(f"{'=' * 80}")
    
    print(f"\nModel Information:")
    print(f"  Name: {results['model']}")
    print(f"  Device: {results['device']}")
    print(f"  Initialization Time: {results['init_time']:.2f}s")
    
    print(f"\nTest Configuration:")
    print(f"  Documents: {results['num_documents']}")
    print(f"  Top-N: {results['top_n']}")
    print(f"  Runs: {len(results['inference_times'])}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average Inference Time: {results['avg_inference_time']:.3f}s")
    print(f"  Min Inference Time: {results['min_inference_time']:.3f}s")
    print(f"  Max Inference Time: {results['max_inference_time']:.3f}s")
    print(f"  Throughput: {results['throughput_docs_per_sec']:.1f} docs/sec")
    
    print(f"\nPer-Document Metrics:")
    avg_per_doc = results['avg_inference_time'] / results['num_documents'] * 1000
    print(f"  Average Time per Document: {avg_per_doc:.1f}ms")
    
    print(f"\n{'=' * 80}")
    print("Baseline established for iGPU optimization comparison")
    print(f"{'=' * 80}")


def main():
    """Main execution function."""
    
    print(f"\n{'#' * 80}")
    print("# ReRanker Performance Verification")
    print("# Establishing CPU Baseline for iGPU Optimization")
    print(f"{'#' * 80}")
    
    # Check Python version
    print(f"\nPython Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Run benchmark
    results = benchmark_reranker(
        model_name="BAAI/bge-reranker-v2-m3",
        num_docs=50,
        top_n=10,
        device="cpu",
    )
    
    # Print results
    print_results(results)
    
    # Save results to file
    output_file = Path(__file__).parent / "reranker_baseline_results.txt"
    
    try:
        with open(output_file, "w") as f:
            f.write("ReRanker Performance Baseline (CPU)\n")
            f.write("=" * 80 + "\n\n")
            
            if results["success"]:
                f.write(f"Model: {results['model']}\n")
                f.write(f"Device: {results['device']}\n")
                f.write(f"Documents: {results['num_documents']}\n")
                f.write(f"Top-N: {results['top_n']}\n\n")
                
                f.write(f"Initialization Time: {results['init_time']:.2f}s\n")
                f.write(f"Average Inference Time: {results['avg_inference_time']:.3f}s\n")
                f.write(f"Throughput: {results['throughput_docs_per_sec']:.1f} docs/sec\n")
                f.write(f"\nIndividual Runs:\n")
                for i, t in enumerate(results['inference_times'], 1):
                    f.write(f"  Run {i}: {t:.3f}s\n")
            else:
                f.write(f"FAILED: {results.get('error', 'Unknown error')}\n")
        
        print(f"\n📄 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    # Return exit code
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
