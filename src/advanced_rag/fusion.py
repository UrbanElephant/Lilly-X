"""Reciprocal Rank Fusion Module.

Implements Reciprocal Rank Fusion (RRF) to combine ranked results from multiple
retrieval strategies into a single unified ranking.

RRF Formula: score(doc) = Σ 1 / (k + rank_i(doc))
where rank_i is the rank of the document in retrieval strategy i.
"""

from collections import defaultdict
from typing import List, Optional

from llama_index.core.schema import NodeWithScore


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining multiple ranked result lists.
    
    RRF is effective for combining results from different retrieval strategies
    (vector, BM25, graph) because it:
    - Is parameter-free (except for k)
    - Doesn't require score normalization
    - Gives higher weight to consistently high-ranked items
    """
    
    def __init__(self, k: int = 60):
        """Initialize RRF.
        
        Args:
            k: Constant for RRF formula (typical value: 60). Higher k gives
               less weight to rank differences.
        """
        self.k = k
    
    def fuse(
        self, 
        results_list: List[List[NodeWithScore]],
        top_n: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """Fuse multiple ranked result lists using RRF.
        
        Args:
            results_list: List of ranked result lists from different retrievers
            top_n: Optional limit on number of results to return
            
        Returns:
            Fused and re-ranked list of nodes with updated scores
        """
        if not results_list:
            return []
        
        # Handle empty result lists
        results_list = [r for r in results_list if r]
        if not results_list:
            return []
        
        print(f"[RRF] Fusing {len(results_list)} result lists")
        
        # Track RRF scores by node ID
        rrf_scores = defaultdict(float)
        node_map = {}  # Map node_id -> NodeWithScore (keep first occurrence)
        
        # Process each result list
        for result_idx, results in enumerate(results_list):
            print(f"[RRF]   List {result_idx + 1}: {len(results)} results")
            
            for rank, node_with_score in enumerate(results):
                node = node_with_score.node
                
                # Use node_id as unique identifier
                # If node doesn't have id_, use hash of text
                node_id = getattr(node, 'node_id', None) or getattr(node, 'id_', None)
                if not node_id:
                    node_id = hash(node.get_content())
                
                # Calculate RRF score contribution: 1 / (k + rank)
                # rank is 0-indexed, so rank 0 gets highest score
                rrf_score = 1.0 / (self.k + rank + 1)
                rrf_scores[node_id] += rrf_score
                
                # Store node if not seen before
                if node_id not in node_map:
                    node_map[node_id] = node_with_score
        
        # Create fused results with RRF scores
        fused_results = []
        for node_id, rrf_score in rrf_scores.items():
            node_with_score = node_map[node_id]
            # Create new NodeWithScore with RRF score
            fused_node = NodeWithScore(
                node=node_with_score.node,
                score=rrf_score
            )
            fused_results.append(fused_node)
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to top_n if specified
        if top_n is not None:
            fused_results = fused_results[:top_n]
        
        print(f"[RRF] Fused to {len(fused_results)} unique results")
        
        return fused_results


# Self-contained testing
if __name__ == "__main__":
    """Quick testing of RRF fusion."""
    
    print("=" * 80)
    print("Reciprocal Rank Fusion - Self Test")
    print("=" * 80)
    
    from llama_index.core.schema import TextNode
    
    # Create test nodes
    node_a = TextNode(text="Document A about vectors", id_="node_a")
    node_b = TextNode(text="Document B about databases", id_="node_b")
    node_c = TextNode(text="Document C about embeddings", id_="node_c")
    node_d = TextNode(text="Document D about search", id_="node_d")
    
    # Create mock result lists from different retrievers
    # List 1: Vector search results
    vector_results = [
        NodeWithScore(node=node_a, score=0.95),
        NodeWithScore(node=node_b, score=0.85),
        NodeWithScore(node=node_c, score=0.75),
    ]
    
    # List 2: BM25 results (different ranking)
    bm25_results = [
        NodeWithScore(node=node_b, score=12.5),  # Different scoring scheme
        NodeWithScore(node=node_d, score=10.2),
        NodeWithScore(node=node_a, score=8.7),
    ]
    
    # List 3: Graph results (partial overlap)
    graph_results = [
        NodeWithScore(node=node_c, score=1.0),
        NodeWithScore(node=node_a, score=1.0),
    ]
    
    print("\n📊 Input Result Lists:")
    print("\n  Vector Search (3 results):")
    for i, r in enumerate(vector_results, 1):
        print(f"    {i}. {r.node.id_} (score: {r.score:.2f})")
    
    print("\n  BM25 Search (3 results):")
    for i, r in enumerate(bm25_results, 1):
        print(f"    {i}. {r.node.id_} (score: {r.score:.2f})")
    
    print("\n  Graph Search (2 results):")
    for i, r in enumerate(graph_results, 1):
        print(f"    {i}. {r.node.id_} (score: {r.score:.2f})")
    
    # Test fusion
    print("\n" + "─" * 80)
    print("Applying Reciprocal Rank Fusion...")
    print("─" * 80)
    
    try:
        rrf = ReciprocalRankFusion(k=60)
        fused_results = rrf.fuse(
            [vector_results, bm25_results, graph_results],
            top_n=5
        )
        
        print("\n✅ Fused Results (ranked by RRF score):")
        for i, result in enumerate(fused_results, 1):
            print(f"\n  {i}. {result.node.id_}")
            print(f"     RRF Score: {result.score:.6f}")
            print(f"     Text: {result.node.text}")
        
        # Verify node_a has highest score (appears in all 3 lists)
        assert fused_results[0].node.id_ == "node_a", "Node A should rank highest (in all lists)"
        
        print("\n" + "=" * 80)
        print("✅ RRF fusion test completed successfully")
        print("=" * 80)
        
        # Show RRF calculation explanation
        print("\n📚 RRF Calculation Example:")
        print("   Node A appears at ranks: [0, 2, 1] in [Vector, BM25, Graph]")
        print(f"   RRF(A) = 1/(60+0+1) + 1/(60+2+1) + 1/(60+1+1)")
        print(f"          = 1/61 + 1/63 + 1/62")
        print(f"          = 0.0164 + 0.0159 + 0.0161")
        print(f"          = 0.0484")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
