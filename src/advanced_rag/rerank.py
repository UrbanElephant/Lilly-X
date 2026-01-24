"""Reranking Module.

Implements cross-encoder based reranking to refine retrieval results.
Uses model like BAAI/bge-reranker-v2-m3 for more accurate relevance scoring.
"""

from typing import List, Optional

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle


class ReRanker:
    """Cross-encoder reranker for refining retrieval results.
    
    Reranking uses a cross-encoder model that scores query-document pairs
    more accurately than bi-encoder embeddings. This is computationally
    expensive but provides significant quality improvements.
    """
    
    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
        device: str = "cpu",
    ):
        """Initialize reranker.
        
        Args:
            model: HuggingFace model identifier for cross-encoder
            top_n: Number of top results to return after reranking
            device: Device to run model on ('cpu', 'cuda', or 'cuda:0')
        """
        self.model_name = model
        self.top_n = top_n
        self.device = device
        
        print(f"[ReRanker] Initializing with model: {model}")
        print(f"[ReRanker] Device: {device}")
        
        try:
            # Try to use FlagEmbeddingReranker (preferred)
            from llama_index.postprocessor.flag_embedding_reranker import (
                FlagEmbeddingReranker,
            )
            
            # FlagEmbeddingReranker doesn't accept 'device' parameter
            # It auto-detects or uses internal settings
            self._reranker = FlagEmbeddingReranker(
                model=model,
                top_n=top_n,
                # device parameter not supported - removed
            )
            print(f"[ReRanker] Using FlagEmbeddingReranker")
            print(f"[ReRanker] Note: FlagEmbeddingReranker auto-detects device")
            
        except (ImportError, TypeError) as e:
            # ImportError: package not installed
            # TypeError: incompatible parameters
            if isinstance(e, TypeError):
                print(f"[ReRanker] FlagEmbeddingReranker parameter error: {e}")
            else:
                print(f"[ReRanker] FlagEmbeddingReranker not available")
                print("[ReRanker] Install with: pip install llama-index-postprocessor-flag-embedding-reranker")
            
            # Fallback to SentenceTransformerRerank
            print(f"[ReRanker] Falling back to SentenceTransformerRerank")
            
            try:
                self._reranker = SentenceTransformerRerank(
                    model=model,
                    top_n=top_n,
                )
                print(f"[ReRanker] Using SentenceTransformerRerank")
            except Exception as fallback_error:
                print(f"[ReRanker] Warning: Could not initialize reranker: {fallback_error}")
                self._reranker = None
    
    def rerank(
        self,
        nodes: List[NodeWithScore],
        query: str,
        top_n: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """Rerank nodes using cross-encoder model.
        
        Args:
            nodes: List of nodes to rerank
            query: Query string for relevance scoring
            top_n: Number of results to return (overrides init value if provided)
            
        Returns:
            Reranked list of nodes with updated scores
        """
        if not nodes:
            return []
        
        if self._reranker is None:
            print("[ReRanker] Reranker not initialized, returning original nodes")
            return nodes[:top_n] if top_n else nodes[:self.top_n]
        
        print(f"[ReRanker] Reranking {len(nodes)} nodes for query: '{query[:50]}...'")
        
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Update top_n if provided
            if top_n is not None:
                original_top_n = self._reranker.top_n
                self._reranker.top_n = top_n
            
            # Rerank using postprocessor
            reranked_nodes = self._reranker.postprocess_nodes(
                nodes=nodes,
                query_bundle=query_bundle,
            )
            
            # Restore original top_n
            if top_n is not None:
                self._reranker.top_n = original_top_n
            
            print(f"[ReRanker] Reranked to top {len(reranked_nodes)} results")
            
            return reranked_nodes
            
        except Exception as e:
            print(f"[ReRanker] Error during reranking: {e}")
            # Fallback: return original nodes
            return nodes[:top_n] if top_n else nodes[:self.top_n]
    
    def get_model_info(self) -> dict:
        """Get information about the reranker model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model_name,
            "top_n": self.top_n,
            "device": self.device,
            "initialized": self._reranker is not None,
        }


# Self-contained testing
if __name__ == "__main__":
    """Quick testing of reranker."""
    
    print("=" * 80)
    print("ReRanker Module - Self Test")
    print("=" * 80)
    
    from llama_index.core.schema import TextNode
    
    # Create test nodes with varying relevance
    test_nodes = [
        NodeWithScore(
            node=TextNode(
                text="Vector databases are specialized systems for storing and querying high-dimensional embeddings.",
                id_="node_1"
            ),
            score=0.75
        ),
        NodeWithScore(
            node=TextNode(
                text="Machine learning models require large amounts of training data.",
                id_="node_2"
            ),
            score=0.72
        ),
        NodeWithScore(
            node=TextNode(
                text="Embeddings are dense vector representations of text that capture semantic meaning.",
                id_="node_3"
            ),
            score=0.70
        ),
        NodeWithScore(
            node=TextNode(
                text="Python is a popular programming language for data science.",
                id_="node_4"
            ),
            score=0.68
        ),
        NodeWithScore(
            node=TextNode(
                text="Semantic search uses embeddings to find documents by meaning rather than keywords.",
                id_="node_5"
            ),
            score=0.65
        ),
    ]
    
    query = "What are vector databases and embeddings?"
    
    print(f"\n📝 Query: {query}")
    print(f"\n📦 Testing with {len(test_nodes)} nodes")
    
    print("\n🔧 Original Ranking (by initial score):")
    for i, node in enumerate(test_nodes, 1):
        print(f"  {i}. [Score: {node.score:.3f}] {node.node.text[:80]}...")
    
    try:
        print("\n" + "─" * 80)
        print("Initializing ReRanker...")
        print("─" * 80)
        
        # Initialize reranker
        reranker = ReRanker(
            model="BAAI/bge-reranker-v2-m3",
            top_n=3,
            device="cpu"
        )
        
        # Get model info
        info = reranker.get_model_info()
        print(f"\n📊 Model Info:")
        print(f"   Model: {info['model']}")
        print(f"   Device: {info['device']}")
        print(f"   Top N: {info['top_n']}")
        print(f"   Initialized: {info['initialized']}")
        
        if info['initialized']:
            print("\n" + "─" * 80)
            print("Reranking nodes...")
            print("─" * 80)
            
            # Rerank
            reranked = reranker.rerank(
                nodes=test_nodes,
                query=query,
                top_n=3
            )
            
            print(f"\n✅ Reranked Results (top {len(reranked)}):")
            for i, node in enumerate(reranked, 1):
                print(f"\n  {i}. [Score: {node.score:.4f}]")
                print(f"     {node.node.text}")
            
            print("\n" + "=" * 80)
            print("✅ Reranker test completed successfully")
            print("=" * 80)
            
            print("\n💡 Note: Reranked scores should better reflect semantic relevance to query.")
            print("   Nodes 1, 3, and 5 should rank higher as they discuss vectors/embeddings.")
        
        else:
            print("\n⚠️  Reranker not initialized, skipping reranking test")
            print("   Install with: pip install llama-index-postprocessor-flag-embedding-reranker")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 If model download fails, ensure:")
        print("   1. Internet connection is available")
        print("   2. HuggingFace transformers is installed")
        print("   3. Sufficient disk space for model (~500MB)")
