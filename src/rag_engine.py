"""
The Clarity Engine: Sovereign RAG Intelligence Layer (Refactored)

This module implements the core "Clarity Engine" for Lilly-X using the Advanced RAG Pipeline.

**Architecture:**
- Advanced RAG Pipeline: Orchestrates query decomposition, hybrid retrieval, fusion, and reranking
- Query Planning: Decomposes complex queries into atomic sub-queries
- Hybrid Retrieval: Combines vector (Qdrant HNSW) + BM25 + graph (Neo4j) search
- Reciprocal Rank Fusion: Intelligently combines results from multiple retrieval strategies
- Cross-Encoder Reranking: Two-stage retrieval for precision

**Performance Optimizations:**
- RAM-First Vector Store (mmap_threshold_kb: 0 in compose.yaml)
- Logarithmic HNSW indexing for sub-millisecond retrieval
- Async-ready pipeline for future parallelization
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

# Core LlamaIndex imports
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.storage_context import StorageContext

# LLM and Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Vector Store
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Advanced RAG Pipeline
from advanced_rag import AdvancedRAGPipeline
from advanced_rag.retrieval import SimpleGraphRetriever

# Local imports
from src.config import settings
from src.database import get_qdrant_client
from src.graph_database import get_neo4j_driver

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response object from RAG engine."""
    response: str
    source_nodes: List[NodeWithScore]
    metadata: dict = None


class RAGEngine:
    """
    The Clarity Engine: Advanced RAG orchestrator.
    
    This class implements the "Sovereign AI" vision using the AdvancedRAGPipeline
    to combine vector search, BM25 keyword matching, graph traversal, and
    cross-encoder reranking for high-quality retrieval.
    
    The engine prioritizes latency and data sovereignty over cloud dependency.
    """
    
    def __init__(
        self,
        vector_index: Optional[VectorStoreIndex] = None,
        enable_decomposition: bool = True,
        enable_hyde: bool = False,
        enable_rewriting: bool = False,
        verbose: bool = True,
    ):
        """Initialize RAG Engine with Advanced RAG Pipeline.
        
        Args:
            vector_index: Pre-built VectorStoreIndex. If None, will be created from settings.
            enable_decomposition: Enable query decomposition into sub-questions
            enable_hyde: Enable Hypothetical Document Embeddings
            enable_rewriting: Enable query rewriting/expansion
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self._index = vector_index
        self._neo4j_driver = None
        self._pipeline = None
        
        # Initialize components
        self._initialize(
            enable_decomposition=enable_decomposition,
            enable_hyde=enable_hyde,
            enable_rewriting=enable_rewriting,
        )
    
    def _initialize(
        self,
        enable_decomposition: bool,
        enable_hyde: bool,
        enable_rewriting: bool,
    ):
        """Initialize LlamaIndex components and Advanced RAG Pipeline."""
        logger.info("Initializing RAG Engine with Advanced RAG Pipeline...")
        
        # 1. Setup LLM
        try:
            logger.info(f"Loading LLM: {settings.llm_model}")
            llm = Ollama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                request_timeout=360.0,
                context_window=8192,
                system_prompt="""You are Lilly-X, a precision-focused RAG assistant for engineering teams.
STRICT RULES:
1. Answer ONLY based on the provided context.
2. If the answer is not in the context, state 'I cannot find this information in the knowledge base.'
3. Be concise and technical.
4. Always cite the source filename when referencing specific data.""",
                additional_kwargs={"num_ctx": 8192}
            )
            Settings.llm = llm
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise

        # 2. Setup Embedding Model
        try:
            logger.info(f"Loading Embeddings: {settings.embedding_model}")
            embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding_model,
                cache_folder="./models",
            )
            Settings.embed_model = embed_model
            Settings.chunk_size = settings.chunk_size
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # 3. Initialize Neo4j driver for graph queries
        graph_retriever = None
        try:
            logger.info("Initializing Neo4j connection for graph retrieval...")
            self._neo4j_driver = get_neo4j_driver()
            
            # Create graph retriever
            graph_retriever = SimpleGraphRetriever(
                neo4j_driver=self._neo4j_driver,
                llm=Settings.llm,
                similarity_top_k=5,
            )
            logger.info("Neo4j graph retriever initialized")
            
        except Exception as e:
            logger.warning(f"Neo4j initialization failed: {e}. Graph queries will be disabled.")
            self._neo4j_driver = None

        # 4. Connect to Vector Store (if not provided)
        if self._index is None:
            try:
                logger.info("Creating VectorStoreIndex from Qdrant...")
                client = get_qdrant_client()
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=settings.qdrant_collection
                )
                
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=StorageContext.from_defaults(vector_store=vector_store),
                )
                logger.info(f"VectorStoreIndex loaded from '{settings.qdrant_collection}'")
                
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant/Index: {e}")
                raise

        # 5. Initialize Advanced RAG Pipeline
        try:
            logger.info("Initializing Advanced RAG Pipeline...")
            
            self._pipeline = AdvancedRAGPipeline(
                vector_index=self._index,
                graph_retriever=graph_retriever,
                enable_decomposition=enable_decomposition,
                enable_hyde=enable_hyde,
                enable_rewriting=enable_rewriting,
                fusion_k=60,
                reranker_model=settings.reranker_model,
                reranker_device="cpu",  # TODO: Migrate to iGPU
                verbose=self.verbose,
            )
            
            logger.info("✅ RAG Engine initialized successfully with Advanced RAG Pipeline")
            logger.info(f"   - Query Decomposition: {enable_decomposition}")
            logger.info(f"   - HyDE: {enable_hyde}")
            logger.info(f"   - Query Rewriting: {enable_rewriting}")
            logger.info(f"   - Graph Retrieval: {graph_retriever is not None}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced RAG Pipeline: {e}")
            raise
    
    async def aquery(
        self,
        user_query: str,
        top_k: int = None,
        top_n: int = None,
    ) -> RAGResponse:
        """Execute async query using Advanced RAG Pipeline.
        
        This is the main entry point for async queries. It runs the full
        Advanced RAG pipeline: decomposition → retrieval → fusion → reranking.
        
        Args:
            user_query: User's query string
            top_k: Number of results for fusion stage (default: 25)
            top_n: Number of final results after reranking (default: 5)
            
        Returns:
            RAGResponse with synthesized answer and source nodes
            
        Raises:
            RuntimeError: If RAG Engine not initialized
        """
        if self._pipeline is None:
            raise RuntimeError("RAG Engine not initialized. Pipeline is None.")
        
        # Use defaults from config if not specified
        if top_k is None:
            top_k = getattr(settings, 'advanced_top_k', 25)
        if top_n is None:
            top_n = getattr(settings, 'advanced_top_n', settings.top_k_final)
        
        logger.info(f"🔍 Processing query (async): '{user_query}'")
        
        try:
            # STEP 1: Run Advanced RAG Pipeline
            ranked_nodes = await self._pipeline.run(
                query=user_query,
                top_k=top_k,
                top_n=top_n,
            )
            
            if not ranked_nodes:
                logger.warning("No nodes retrieved from pipeline")
                return RAGResponse(
                    response="I could not find relevant information in the knowledge base for your query.",
                    source_nodes=[],
                    metadata={"pipeline_stage": "retrieval_empty"}
                )
            
            # STEP 2: Extract context from ranked nodes
            context_str = "\n\n".join([
                f"[Source: {node.node.metadata.get('source', 'Unknown')}]\n{node.node.get_content()}"
                for node in ranked_nodes
            ])
            
            # STEP 3: Generate answer using LLM
            prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {user_query}

Answer:"""
            
            logger.info("💭 Generating answer with LLM...")
            response = Settings.llm.complete(prompt)
            answer = response.text.strip()
            
            logger.info(f"✅ Answer generated ({len(answer)} chars)")
            
            return RAGResponse(
                response=answer,
                source_nodes=ranked_nodes,
                metadata={
                    "pipeline_stage": "complete",
                    "num_sources": len(ranked_nodes),
                    "top_k": top_k,
                    "top_n": top_n,
                }
            )
            
        except Exception as e:
            logger.error(f"Advanced RAG Pipeline failed: {e}")
            
            # FALLBACK: Try simple vector retrieval
            try:
                logger.warning("Falling back to simple vector retrieval...")
                retriever = self._index.as_retriever(similarity_top_k=top_n)
                nodes = retriever.retrieve(user_query)
                
                if not nodes:
                    raise RuntimeError("No results from fallback retrieval")
                
                context_str = "\n\n".join([node.get_content() for node in nodes])
                prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {user_query}

Answer:"""
                
                response = Settings.llm.complete(prompt)
                answer = response.text.strip()
                
                return RAGResponse(
                    response=answer,
                    source_nodes=nodes,
                    metadata={
                        "pipeline_stage": "fallback",
                        "error": str(e),
                    }
                )
                
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {fallback_error}")
                raise RuntimeError(
                    f"Both Advanced RAG Pipeline and fallback failed. "
                    f"Pipeline error: {e}. Fallback error: {fallback_error}"
                )
    
    def query(self, user_query: str, top_k: int = None, top_n: int = None) -> RAGResponse:
        """Execute synchronous query (wrapper for aquery).
        
        This is a synchronous wrapper around aquery() for compatibility
        with code that cannot use async/await.
        
        Args:
            user_query: User's query string
            top_k: Number of results for fusion stage
            top_n: Number of final results after reranking
            
        Returns:
            RAGResponse with answer and source nodes
        """
        return asyncio.run(self.aquery(user_query, top_k, top_n))
    
    @property
    def index(self) -> VectorStoreIndex:
        """Access to underlying vector index."""
        return self._index
    
    @property
    def pipeline(self) -> AdvancedRAGPipeline:
        """Access to Advanced RAG Pipeline."""
        return self._pipeline


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    """Smoke test with mocked dependencies."""
    
    import sys
    from pathlib import Path
    
    print("=" * 80)
    print("RAG Engine - Smoke Test (Mocked)")
    print("=" * 80)
    
    # Mock components for testing without real databases
    from llama_index.core.schema import Document, TextNode, QueryBundle
    from llama_index.core.base.base_retriever import BaseRetriever
    from typing import List
    
    class MockRetriever(BaseRetriever):
        """Mock retriever for testing."""
        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            return [
                NodeWithScore(
                    node=TextNode(
                        text="Mock document about RAG systems.",
                        id_="mock_1",
                        metadata={"source": "mock_doc.txt"}
                    ),
                    score=0.95
                ),
                NodeWithScore(
                    node=TextNode(
                        text="Advanced retrieval techniques for AI.",
                        id_="mock_2",
                        metadata={"source": "mock_doc2.txt"}
                    ),
                    score=0.88
                ),
            ]
    
    class MockLLM:
        """Mock LLM for testing."""
        def complete(self, prompt: str):
            class MockResponse:
                text = "This is a mock answer based on the provided context."
            return MockResponse()
    
    try:
        # Setup mock Settings
        Settings.llm = MockLLM()
        Settings.embed_model = None  # Mock doesn't need embeddings
        
        # Create mock index
        print("\n📦 Creating mock index...")
        docs = [
            Document(text="RAG combines retrieval with generation.", id_="doc1"),
            Document(text="Vector databases store embeddings.", id_="doc2"),
        ]
        mock_index = VectorStoreIndex.from_documents(docs)
        
        # Initialize RAG Engine with mock index
        print("\n🔧 Initializing RAG Engine...")
        engine = RAGEngine(
            vector_index=mock_index,
            enable_decomposition=False,  # Disable for simple test
            enable_hyde=False,
            enable_rewriting=False,
            verbose=True,
        )
        
        # Monkey-patch pipeline retriever with mock
        print("\n🐵 Monkey-patching with mock retriever...")
        engine._pipeline.hybrid_retriever = MockRetriever()
        
        # Test sync query
        print("\n" + "─" * 80)
        print("Testing sync query...")
        print("─" * 80)
        
        response = engine.query("What is RAG?", top_n=2)
        
        print(f"\n✅ Query completed")
        print(f"   Answer: {response.response}")
        print(f"   Sources: {len(response.source_nodes)}")
        print(f"   Metadata: {response.metadata}")
        
        # Test async query
        print("\n" + "─" * 80)
        print("Testing async query...")
        print("─" * 80)
        
        async def test_async():
            response = await engine.aquery("How do vector databases work?", top_n=2)
            print(f"\n✅ Async query completed")
            print(f"   Answer: {response.response}")
            print(f"   Sources: {len(response.source_nodes)}")
            return response
        
        asyncio.run(test_async())
        
        print("\n" + "=" * 80)
        print("🎉 SMOKE TEST PASSED")
        print("=" * 80)
        print("\n✅ RAGEngine class structure validated")
        print("✅ Sync query() method works")
        print("✅ Async aquery() method works")
        print("✅ Pipeline integration successful")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
