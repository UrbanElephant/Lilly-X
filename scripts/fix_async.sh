#!/bin/bash
set -e

echo "🔧 PATCHING ASYNC QDRANT CLIENT..."
echo "==================================="

# 1. Upgrade Database Module (Add Async Client)
echo "📝 Updating src/database.py..."
cat << 'EOF' > src/database.py
import logging
from qdrant_client import QdrantClient, AsyncQdrantClient
from src.config import settings

logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    """Get synchronous Qdrant client."""
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            # Add API key here if needed in future
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Qdrant client: {e}")
        raise

def get_async_qdrant_client() -> AsyncQdrantClient:
    """Get asynchronous Qdrant client for Turbo RAG."""
    try:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            # Add API key here if needed in future
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Async Qdrant client: {e}")
        raise

def close_qdrant_client(client):
    """Close the client connection."""
    if client:
        client.close()
EOF

# 2. Upgrade RAG Engine (Inject Async Client)
echo "🧠 Updating src/rag_engine.py..."
cat << 'EOF' > src/rag_engine.py
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Absolute Imports
from src.advanced_rag.pipeline import AdvancedRAGPipeline
from src.advanced_rag.retrieval import SimpleGraphRetriever
from src.config import settings
# [FIX] Import Async Client Factory
from src.database import get_qdrant_client, get_async_qdrant_client
from src.graph_database import get_neo4j_driver

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    response: str
    source_nodes: List[NodeWithScore]
    metadata: dict = None

class RAGEngine:
    def __init__(self, vector_index=None, enable_decomposition=True, enable_hyde=False, enable_rewriting=False, verbose=True):
        self.verbose = verbose
        self._index = vector_index
        self._neo4j_driver = None
        self._pipeline = None
        self._initialize(enable_decomposition, enable_hyde, enable_rewriting)
    
    def _initialize(self, enable_decomposition, enable_hyde, enable_rewriting):
        # 1. LLM & Embeddings (Loaded from Global Settings)
        if not Settings.llm:
            from src.config import setup_environment
            setup_environment()

        # 2. Graph
        try:
            self._neo4j_driver = get_neo4j_driver()
            graph_retriever = SimpleGraphRetriever(self._neo4j_driver, Settings.llm) if self._neo4j_driver else None
        except:
            graph_retriever = None

        # 3. Vector Store
        if self._index is None:
            logger.info("🔌 Connecting to Qdrant (Sync & Async)...")
            client = get_qdrant_client()
            # [FIX] Initialize Async Client for Parallel Retrieval
            aclient = get_async_qdrant_client()
            
            vector_store = QdrantVectorStore(
                client=client,
                aclient=aclient,  # <--- CRITICAL FIX
                collection_name=settings.qdrant_collection
            )
            self._index = VectorStoreIndex.from_vector_store(vector_store)

        # 4. Pipeline
        self._pipeline = AdvancedRAGPipeline(
            vector_index=self._index,
            graph_retriever=graph_retriever,
            enable_decomposition=enable_decomposition,
            enable_hyde=enable_hyde,
            enable_rewriting=enable_rewriting,
            reranker_model=settings.reranker_model,
            reranker_device="cpu"
        )

    async def aquery(self, user_query: str, top_k=25, top_n=5) -> RAGResponse:
        try:
            # Step 1: Run Pipeline (Retrieval -> Fusion -> Rerank)
            ranked_nodes = await self._pipeline.run(query=user_query, top_k=top_k, top_n=top_n)
            
            if not ranked_nodes: 
                return RAGResponse("I could not find relevant information in the knowledge base.", [])
            
            # Step 2: Context Construction
            context = "\n".join([f"[Source: {n.node.metadata.get('source', 'Unknown')}]\n{n.node.get_content()}" for n in ranked_nodes])
            
            # Step 3: Synthesis
            prompt = f"""You are Lilly-X. Answer based ONLY on the context below.

Context:
{context}

Question: {user_query}

Answer:"""
            
            response = await Settings.llm.acomplete(prompt)
            return RAGResponse(response.text.strip(), ranked_nodes, {"top_k": top_k})
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            # Fallback: Simple Vector Search
            retriever = self._index.as_retriever(similarity_top_k=top_n)
            nodes = await retriever.aretrieve(user_query)
            context = "\n".join([n.get_content() for n in nodes])
            response = await Settings.llm.acomplete(f"Context:\n{context}\n\nQ: {user_query}\n\nA:")
            return RAGResponse(response.text.strip(), nodes, {"error": str(e)})

    def query(self, *args, **kwargs):
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(self.aquery(*args, **kwargs))
EOF

chmod +x src/database.py src/rag_engine.py
echo "✅ ASYNC FIX APPLIED."
echo "👉 Please restart Streamlit: 'streamlit run src/app.py'"
