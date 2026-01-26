"""Hybrid Retrieval Module.

Implements multi-strategy retrieval combining:
- Vector search (semantic similarity via embeddings)
- Keyword search (BM25 lexical matching)
- Graph search (knowledge graph traversal for entities and relationships)
"""

import asyncio
import functools
from typing import List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import (
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import VectorStoreQuery


# Helper function for empty async results
async def _empty_results() -> List[NodeWithScore]:
    """Return empty results list for disabled retrievers."""
    return []


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector, keyword (BM25), and graph search.
    
    Coordinates multiple retrieval strategies and returns combined results
    for subsequent fusion and reranking.
    
    Now includes async retrieval for parallel execution of multiple strategies.
    """
    
    def __init__(
        self,
        vector_index: Optional[VectorStoreIndex] = None,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        graph_retriever: Optional[BaseRetriever] = None,
        graph_top_k: int = 5,
        enable_vector: bool = True,
        enable_bm25: bool = True,
        enable_graph: bool = True,
    ):
        """Initialize hybrid retriever.
        
        Args:
            vector_index: LlamaIndex VectorStoreIndex for vector search
            vector_top_k: Number of results from vector search
            bm25_top_k: Number of results from BM25 search
            graph_retriever: Knowledge graph retriever (e.g., from Neo4j)
            graph_top_k: Number of results from graph search
            enable_vector: Enable vector search
            enable_bm25: Enable BM25 keyword search
            enable_graph: Enable graph search
        """
        self._vector_index = vector_index
        self._vector_top_k = vector_top_k
        self._bm25_top_k = bm25_top_k
        self._graph_retriever = graph_retriever
        self._graph_top_k = graph_top_k
        
        self._enable_vector = enable_vector
        self._enable_bm25 = enable_bm25
        self._enable_graph = enable_graph
        
        # Initialize BM25 retriever from vector index nodes if enabled
        self._bm25_retriever = None
        if enable_bm25 and vector_index is not None:
            try:
                from llama_index.retrievers.bm25 import BM25Retriever
                # Get all nodes from the index
                nodes = list(vector_index.docstore.docs.values())
                if nodes:
                    self._bm25_retriever = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=bm25_top_k,
                    )
                    print(f"[HybridRetriever] Initialized BM25 with {len(nodes)} nodes")
                else:
                    print("[HybridRetriever] Warning: No nodes available for BM25 initialization")
            except ImportError:
                print("[HybridRetriever] Warning: BM25Retriever not available, install llama-index-retrievers-bm25")
                self._enable_bm25 = False
            except Exception as e:
                print(f"[HybridRetriever] Warning: BM25 initialization failed: {e}")
                self._enable_bm25 = False
        
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Synchronous wrapper for hybrid retrieval.
        
        Handles nested event loops using nest_asyncio for robust operation
        in both sync and async contexts.
        
        Args:
            query_bundle: Query bundle containing query string
            
        Returns:
            Combined list of nodes from all enabled retrieval strategies
        """
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in a running loop - need nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    print("[HybridRetriever] Applied nest_asyncio for nested loop support")
                except ImportError:
                    print("[HybridRetriever] Warning: nest_asyncio not available, falling back to sequential retrieval")
                    # Fallback to sequential retrieval
                    return self._retrieve_sequential(query_bundle)
                
                # Now we can safely run async in nested loop
                return asyncio.run(self._aretrieve(query_bundle))
                
            except RuntimeError:
                # No running loop - safe to create new one
                return asyncio.run(self._aretrieve(query_bundle))
                
        except Exception as e:
            print(f"[HybridRetriever] Error in async retrieval, falling back to sequential: {e}")
            return self._retrieve_sequential(query_bundle)
    
    def _retrieve_sequential(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Fallback sequential retrieval for compatibility.
        
        Args:
            query_bundle: Query bundle containing query string
            
        Returns:
            Combined list of nodes from all enabled retrieval strategies
        """
        all_results = []
        
        # 1. Vector Search (Semantic Similarity)
        if self._enable_vector and self._vector_index is not None:
            try:
                vector_retriever = VectorIndexRetriever(
                    index=self._vector_index,
                    similarity_top_k=self._vector_top_k,
                )
                vector_results = vector_retriever.retrieve(query_bundle)
                all_results.extend(vector_results)
                print(f"[HybridRetriever] Vector search returned {len(vector_results)} results")
            except Exception as e:
                print(f"[HybridRetriever] Vector search error: {e}")
        
        # 2. BM25 Search (Keyword Matching)
        if self._enable_bm25 and self._bm25_retriever is not None:
            try:
                bm25_results = self._bm25_retriever.retrieve(query_bundle)
                all_results.extend(bm25_results)
                print(f"[HybridRetriever] BM25 search returned {len(bm25_results)} results")
            except Exception as e:
                print(f"[HybridRetriever] BM25 search error: {e}")
        
        # 3. Graph Search (Entity-based Traversal)
        if self._enable_graph and self._graph_retriever is not None:
            try:
                graph_results = self._graph_retriever.retrieve(query_bundle)
                # Limit graph results to top_k
                graph_results = graph_results[:self._graph_top_k]
                all_results.extend(graph_results)
                print(f"[HybridRetriever] Graph search returned {len(graph_results)} results")
            except Exception as e:
                print(f"[HybridRetriever] Graph search error: {e}")
        
        print(f"[HybridRetriever] Total combined results: {len(all_results)}")
        return all_results
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using hybrid strategy (asynchronous, parallel execution).
        
        This async version executes all retrieval strategies in parallel using
        asyncio.gather for maximum performance.
        
        Args:
            query_bundle: Query bundle containing query string
            
        Returns:
            Combined list of nodes from all enabled retrieval strategies
        """
        tasks = []
        loop = asyncio.get_event_loop()
        
        # 1. Vector Search (Async) - Use aretrieve if available
        if self._enable_vector and self._vector_index is not None:
            async def vector_task():
                try:
                    vector_retriever = VectorIndexRetriever(
                        index=self._vector_index,
                        similarity_top_k=self._vector_top_k,
                    )
                    # Use async retrieval
                    vector_results = await vector_retriever.aretrieve(query_bundle)
                    print(f"[HybridRetriever] Vector search returned {len(vector_results)} results")
                    return vector_results
                except Exception as e:
                    print(f"[HybridRetriever] Vector search error: {e}")
                    return []
            
            tasks.append(vector_task())
        
        # 2. BM25 Search (Sync wrapped in executor)
        if self._enable_bm25 and self._bm25_retriever is not None:
            async def bm25_task():
                try:
                    # Wrap synchronous BM25 retrieval in executor
                    bm25_results = await loop.run_in_executor(
                        None,
                        functools.partial(self._bm25_retriever.retrieve, query_bundle)
                    )
                    print(f"[HybridRetriever] BM25 search returned {len(bm25_results)} results")
                    return bm25_results
                except Exception as e:
                    print(f"[HybridRetriever] BM25 search error: {e}")
                    return []
            
            tasks.append(bm25_task())
        
        # 3. Graph Search (Sync wrapped in executor)
        if self._enable_graph and self._graph_retriever is not None:
            async def graph_task():
                try:
                    # Wrap synchronous graph retrieval in executor
                    graph_results = await loop.run_in_executor(
                        None,
                        functools.partial(self._graph_retriever.retrieve, query_bundle)
                    )
                    # Limit graph results to top_k
                    graph_results = graph_results[:self._graph_top_k]
                    print(f"[HybridRetriever] Graph search returned {len(graph_results)} results")
                    return graph_results
                except Exception as e:
                    print(f"[HybridRetriever] Graph search error: {e}")
                    return []
            
            tasks.append(graph_task())
        
        # Execute all tasks in parallel
        if tasks:
            results_lists = await asyncio.gather(*tasks)
            # Flatten results
            all_results = []
            for results in results_lists:
                all_results.extend(results)
            
            print(f"[HybridRetriever] Total combined results (async): {len(all_results)}")
            return all_results
        else:
            return []


class SimpleGraphRetriever(BaseRetriever):
    """Simple knowledge graph retriever for Neo4j.
    
    Extracts entities from query and retrieves neighboring nodes from the graph.
    This is a simplified version - for production use KnowledgeGraphRAGRetriever.
    """
    
    def __init__(
        self,
        neo4j_driver,
        llm: Optional = None,
        similarity_top_k: int = 5,
        depth: int = 1,
    ):
        """Initialize graph retriever.
        
        Args:
            neo4j_driver: Neo4j driver instance
            llm: LLM for entity extraction (defaults to Settings.llm)
            similarity_top_k: Number of results to return
            depth: Graph traversal depth
        """
        self._driver = neo4j_driver
        self._llm = llm or Settings.llm
        self._similarity_top_k = similarity_top_k
        self._depth = depth
        super().__init__()
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query using LLM.
        
        Args:
            query: User query
            
        Returns:
            List of extracted entities
        """
        prompt = f"""Extract the main named entities, concepts, or technical terms from this query.
Return only a JSON array of entity names, no other text.

Query: {query}

Entities:"""
        
        try:
            response = self._llm.complete(prompt)
            response_text = response.text.strip()
            
            import json
            import re
            
            # Try to parse JSON
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                entities = json.loads(match.group(0))
            else:
                # Fallback: split by commas
                entities = [e.strip(' "\'') for e in response_text.split(',')]
            
            return [e for e in entities if e][:5]  # Limit to 5 entities
            
        except Exception as e:
            print(f"[SimpleGraphRetriever] Entity extraction error: {e}")
            # Fallback: use simple word extraction
            words = query.split()
            return [w for w in words if len(w) > 3][:3]
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve from knowledge graph.
        
        Args:
            query_bundle: Query bundle
            
        Returns:
            List of nodes from graph traversal
        """
        query = query_bundle.query_str
        
        # Extract entities
        entities = self._extract_entities(query)
        if not entities:
            return []
        
        print(f"[SimpleGraphRetriever] Extracted entities: {entities}")
        
        # Query Neo4j for entity neighbors
        nodes = []
        
        try:
            with self._driver.session() as session:
                for entity in entities:
                    # Cypher query to find nodes and their neighbors
                    # FIXED: Use labels(n) instead of n.label
                    cypher = """
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS toLower($entity)
                    OPTIONAL MATCH (n)-[r]-(neighbor)
                    RETURN n, labels(n) as node_labels, collect(neighbor) as neighbors
                    LIMIT $limit
                    """
                    
                    try:
                        result = session.run(
                            cypher, 
                            entity=entity, 
                            limit=self._similarity_top_k
                        )
                        
                        for record in result:
                            node_data = record["n"]
                            node_labels = record.get("node_labels", [])
                            neighbors = record.get("neighbors", [])
                            
                            if node_data:
                                # Create text representation
                                text = f"Entity: {node_data.get('name', 'Unknown')}\n"
                                
                                # Use labels list instead of property
                                if node_labels:
                                    text += f"Type: {node_labels[0]}\n"
                                
                                if neighbors:
                                    text += f"Related: {', '.join([n.get('name', 'Unknown') for n in neighbors[:5]])}\n"
                                
                                # Create node
                                text_node = TextNode(
                                    text=text,
                                    metadata={
                                        "source": "knowledge_graph",
                                        "entity": entity,
                                        "labels": node_labels,
                                    }
                                )
                                
                                nodes.append(NodeWithScore(node=text_node, score=1.0))
                    
                    except Exception as query_error:
                        # Log query-specific errors but continue with other entities
                        print(f"[SimpleGraphRetriever] Query error for entity '{entity}': {query_error}")
                        continue
            
            print(f"[SimpleGraphRetriever] Retrieved {len(nodes)} graph nodes")
            return nodes[:self._similarity_top_k]
            
        except Exception as e:
            print(f"[SimpleGraphRetriever] Graph session error: {e}")
            return []


# Self-contained testing
if __name__ == "__main__":
    """Quick testing of hybrid retrieval."""
    
    print("=" * 80)
    print("Hybrid Retrieval Module - Self Test")
    print("=" * 80)
    
    # Create mock nodes for testing
    from llama_index.core.schema import Document
    
    test_docs = [
        Document(text="Vector databases store embeddings for semantic search.", metadata={"source": "doc1"}),
        Document(text="RAG combines retrieval with generation for better LLM responses.", metadata={"source": "doc2"}),
        Document(text="Knowledge graphs represent entities and relationships.", metadata={"source": "doc3"}),
    ]
    
    try:
        print("\n📦 Creating test index...")
        from llama_index.core import VectorStoreIndex
        
        # Create in-memory index
        index = VectorStoreIndex.from_documents(test_docs)
        
        print("\n🔍 Initializing HybridRetriever...")
        retriever = HybridRetriever(
            vector_index=index,
            vector_top_k=5,
            bm25_top_k=5,
            enable_vector=True,
            enable_bm25=True,
            enable_graph=False,  # No graph for basic test
        )
        
        print("\n📝 Testing synchronous retrieval...")
        query = "What are vector databases?"
        results = retriever.retrieve(query)
        
        print(f"\n✅ Retrieved {len(results)} results (sync):")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"   Text: {result.node.text[:100]}...")
            print(f"   Source: {result.node.metadata.get('source', 'Unknown')}")
        
        # Test async retrieval
        print("\n📝 Testing asynchronous retrieval...")
        async def test_async():
            results = await retriever.aretrieve(QueryBundle(query_str=query))
            print(f"\n✅ Retrieved {len(results)} results (async):")
            for i, result in enumerate(results[:3], 1):
                print(f"\n{i}. Score: {result.score:.4f}")
                print(f"   Text: {result.node.text[:100]}...")
            return results
        
        asyncio.run(test_async())
        
        print("\n" + "=" * 80)
        print("✅ Hybrid retrieval test completed (sync + async)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
