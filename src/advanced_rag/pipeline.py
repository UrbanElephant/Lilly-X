"""Advanced RAG Pipeline Integration.

Orchestrates the complete Advanced RAG flow:
1. Query Decomposition → Sub-questions
2. Hybrid Retrieval → Multi-strategy search
3. Reciprocal Rank Fusion → Combine results
4. Reranking → Final relevance scoring
5. Answer Synthesis → Generate response (stub)
"""

import asyncio
from typing import List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, QueryBundle

from .fusion import ReciprocalRankFusion
from .query_transform import HyDEGenerator, QueryDecomposer, QueryRewriter
from .rerank import ReRanker
from .retrieval import HybridRetriever


class AdvancedRAGPipeline:
    """End-to-end Advanced RAG pipeline.
    
    Orchestrates multiple retrieval and reasoning strategies to provide
    high-quality, contextually relevant results for complex queries.
    
    Pipeline Flow:
    1. Query Transformation: Decompose into sub-questions
    2. Multi-Query Retrieval: Search with original + sub-questions
    3. Result Fusion: Combine using Reciprocal Rank Fusion
    4. Reranking: Cross-encoder scoring for final relevance
    5. Answer Synthesis: Generate response from top results
    
    Example:
        >>> pipeline = AdvancedRAGPipeline(vector_index=index)
        >>> results = await pipeline.run("What is RAG?", top_n=5)
        >>> for node in results:
        ...     print(node.text)
    """
    
    def __init__(
        self,
        vector_index: Optional[VectorStoreIndex] = None,
        graph_retriever: Optional[BaseRetriever] = None,
        enable_decomposition: bool = True,
        enable_hyde: bool = False,
        enable_rewriting: bool = False,
        fusion_k: int = 60,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker_device: str = "cpu",
        verbose: bool = True,
    ):
        """Initialize the Advanced RAG pipeline.
        
        Args:
            vector_index: LlamaIndex vector store index
            graph_retriever: Optional knowledge graph retriever
            enable_decomposition: Enable query decomposition
            enable_hyde: Enable Hypothetical Document Embeddings
            enable_rewriting: Enable query rewriting/expansion
            fusion_k: RRF k parameter (default: 60)
            reranker_model: Cross-encoder model for reranking
            reranker_device: Device for reranker ('cpu' or 'cuda')
            verbose: Enable verbose logging
        """
        self.vector_index = vector_index
        self.graph_retriever = graph_retriever
        self.verbose = verbose
        
        # Feature flags
        self.enable_decomposition = enable_decomposition
        self.enable_hyde = enable_hyde
        self.enable_rewriting = enable_rewriting
        
        # Initialize components
        self._log("Initializing Advanced RAG Pipeline components...")
        
        # Query transformation
        if enable_decomposition or enable_hyde or enable_rewriting:
            self.query_decomposer = QueryDecomposer() if enable_decomposition else None
            self.hyde_generator = HyDEGenerator() if enable_hyde else None
            self.query_rewriter = QueryRewriter() if enable_rewriting else None
        
        # Hybrid retrieval
        if vector_index is not None:
            self.hybrid_retriever = HybridRetriever(
                vector_index=vector_index,
                vector_top_k=10,
                bm25_top_k=10,
                graph_retriever=graph_retriever,
                graph_top_k=5,
                enable_vector=True,
                enable_bm25=True,
                enable_graph=graph_retriever is not None,
            )
        else:
            self.hybrid_retriever = None
            self._log("⚠️  No vector index provided, retrieval will be skipped")
        
        # Fusion
        self.fusion = ReciprocalRankFusion(k=fusion_k)
        
        # Reranking
        self.reranker = ReRanker(
            model=reranker_model,
            top_n=5,  # Will be overridden by run() parameter
            device=reranker_device,
        )
        
        self._log("✅ Pipeline initialized")
    
    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[AdvancedRAGPipeline] {message}")
    
    async def run(
        self,
        query: str,
        top_k: int = 50,
        top_n: int = 5,
        max_subqueries: int = 3,
    ) -> List[NodeWithScore]:
        """Run the complete Advanced RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of results to retrieve in fusion stage
            top_n: Number of final results after reranking
            max_subqueries: Maximum sub-questions to generate
            
        Returns:
            List of top-ranked NodeWithScore objects
            
        Raises:
            ValueError: If no retriever is configured
        """
        if self.hybrid_retriever is None:
            raise ValueError("No retriever configured. Provide vector_index during initialization.")
        
        self._log(f"Processing query: '{query}'")
        self._log("=" * 80)
        
        # =====================================================================
        # STEP 1: Query Transformation
        # =====================================================================
        self._log("\n📝 Step 1: Query Transformation")
        
        queries_to_retrieve = [query]  # Always include original query
        
        # 1a. Query Decomposition
        if self.enable_decomposition and self.query_decomposer:
            try:
                self._log("   Decomposing query into sub-questions...")
                sub_queries = self.query_decomposer.decompose(query, max_subqueries=max_subqueries)
                
                # Add sub-queries (avoid duplicates)
                for sq in sub_queries:
                    if sq != query and sq not in queries_to_retrieve:
                        queries_to_retrieve.append(sq)
                
                self._log(f"   ✅ Generated {len(sub_queries)} sub-questions")
                for i, sq in enumerate(sub_queries, 1):
                    self._log(f"      {i}. {sq}")
            
            except Exception as e:
                self._log(f"   ⚠️  Decomposition failed: {e}")
                self._log("   → Continuing with original query")
        
        # 1b. HyDE Generation
        if self.enable_hyde and self.hyde_generator:
            try:
                self._log("   Generating hypothetical document...")
                hyde_doc = self.hyde_generator.generate(query)
                queries_to_retrieve.append(hyde_doc)
                self._log(f"   ✅ HyDE document: {hyde_doc[:100]}...")
            except Exception as e:
                self._log(f"   ⚠️  HyDE generation failed: {e}")
        
        # 1c. Query Rewriting
        if self.enable_rewriting and self.query_rewriter:
            try:
                self._log("   Rewriting query variations...")
                variations = self.query_rewriter.rewrite(query, include_original=False)
                for var in variations[:2]:  # Limit to 2 variations
                    if var not in queries_to_retrieve:
                        queries_to_retrieve.append(var)
                self._log(f"   ✅ Added {len(variations[:2])} variations")
            except Exception as e:
                self._log(f"   ⚠️  Query rewriting failed: {e}")
        
        self._log(f"\n   → Total queries for retrieval: {len(queries_to_retrieve)}")
        
        # =====================================================================
        # STEP 2: Multi-Query Hybrid Retrieval
        # =====================================================================
        self._log("\n🔍 Step 2: Hybrid Retrieval")
        
        all_results = []
        
        for i, q in enumerate(queries_to_retrieve, 1):
            try:
                self._log(f"   Retrieving for query {i}/{len(queries_to_retrieve)}: '{q[:60]}...'")
                
                # Create query bundle
                query_bundle = QueryBundle(query_str=q)
                
                # Retrieve using hybrid strategy
                results = self.hybrid_retriever.retrieve(query_bundle)
                
                all_results.append(results)
                self._log(f"   ✅ Retrieved {len(results)} results")
                
            except Exception as e:
                self._log(f"   ⚠️  Retrieval failed for query {i}: {e}")
                # Continue with other queries
        
        if not all_results:
            self._log("   ❌ No results retrieved from any query")
            return []
        
        self._log(f"\n   → Total result sets: {len(all_results)}")
        
        # =====================================================================
        # STEP 3: Reciprocal Rank Fusion
        # =====================================================================
        self._log("\n🔗 Step 3: Reciprocal Rank Fusion")
        
        try:
            fused_results = self.fusion.fuse(all_results, top_n=top_k)
            self._log(f"   ✅ Fused to {len(fused_results)} unique results")
        except Exception as e:
            self._log(f"   ⚠️  Fusion failed: {e}")
            # Fallback: use first result set
            fused_results = all_results[0][:top_k] if all_results else []
            self._log(f"   → Fallback: using {len(fused_results)} results from first query")
        
        if not fused_results:
            self._log("   ❌ No results after fusion")
            return []
        
        # =====================================================================
        # STEP 4: Cross-Encoder Reranking
        # =====================================================================
        self._log("\n⚡ Step 4: Cross-Encoder Reranking")
        
        try:
            final_results = self.reranker.rerank(
                nodes=fused_results,
                query=query,  # Use original query for reranking
                top_n=top_n,
            )
            self._log(f"   ✅ Reranked to top {len(final_results)} results")
        except Exception as e:
            self._log(f"   ⚠️  Reranking failed: {e}")
            # Fallback: use top-k from fused results
            final_results = fused_results[:top_n]
            self._log(f"   → Fallback: using top {len(final_results)} from fusion")
        
        # =====================================================================
        # STEP 5: Answer Synthesis (Stub)
        # =====================================================================
        self._log("\n📊 Step 5: Results Summary")
        
        self._log(f"   Final result count: {len(final_results)}")
        self._log(f"   Top result scores:")
        for i, node in enumerate(final_results[:3], 1):
            score = node.score if hasattr(node, 'score') else 0.0
            self._log(f"      {i}. Score: {score:.4f} | {node.node.text[:80]}...")
        
        self._log("\n" + "=" * 80)
        self._log("✅ Pipeline execution complete\n")
        
        return final_results
    
    def run_sync(self, query: str, top_k: int = 50, top_n: int = 5) -> List[NodeWithScore]:
        """Synchronous wrapper for run().
        
        Args:
            query: User query
            top_k: Number of results for fusion
            top_n: Number of final results
            
        Returns:
            List of top-ranked nodes
        """
        return asyncio.run(self.run(query, top_k, top_n))


# Self-contained testing
if __name__ == "__main__":
    """Quick test of pipeline structure."""
    
    print("=" * 80)
    print("Advanced RAG Pipeline - Structure Test")
    print("=" * 80)
    
    # Note: This requires a vector index which we don't have in standalone mode
    print("\n⚠️  This module requires a VectorStoreIndex for testing.")
    print("   See tests/verification/verify_advanced_retrieval.py for full test.")
    
    # Test initialization without index (should handle gracefully)
    try:
        pipeline = AdvancedRAGPipeline(
            vector_index=None,
            enable_decomposition=True,
            enable_hyde=False,
            verbose=True,
        )
        print("\n✅ Pipeline initialization: SUCCESS")
        print(f"   - Decomposition: {pipeline.enable_decomposition}")
        print(f"   - HyDE: {pipeline.enable_hyde}")
        print(f"   - Rewriting: {pipeline.enable_rewriting}")
        
    except Exception as e:
        print(f"\n❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
