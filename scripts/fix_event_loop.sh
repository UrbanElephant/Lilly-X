#!/bin/bash
set -e

echo "🔧 FIXING ASYNC EVENT LOOP CRASH..."
echo "==================================="

# 1. Patch RAG Engine (The Bridge)
echo "🧠 Updating src/rag_engine.py..."
cat << 'EOF' > src/rag_engine.py
import asyncio
import logging
import nest_asyncio
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.advanced_rag.pipeline import AdvancedRAGPipeline
from src.advanced_rag.retrieval import SimpleGraphRetriever
from src.config import settings
from src.database import get_qdrant_client, get_async_qdrant_client
from src.graph_database import get_neo4j_driver

logger = logging.getLogger(__name__)

# Apply nested asyncio patch globally
nest_asyncio.apply()

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
        if not Settings.llm:
            from src.config import setup_environment
            setup_environment()

        try:
            self._neo4j_driver = get_neo4j_driver()
            graph_retriever = SimpleGraphRetriever(self._neo4j_driver, Settings.llm) if self._neo4j_driver else None
        except:
            graph_retriever = None

        if self._index is None:
            logger.info("🔌 Connecting to Qdrant (Sync & Async)...")
            client = get_qdrant_client()
            aclient = get_async_qdrant_client()
            vector_store = QdrantVectorStore(
                client=client,
                aclient=aclient,
                collection_name=settings.qdrant_collection
            )
            self._index = VectorStoreIndex.from_vector_store(vector_store)

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
            ranked_nodes = await self._pipeline.run(query=user_query, top_k=top_k, top_n=top_n)
            
            if not ranked_nodes: 
                return RAGResponse("I could not find relevant information in the knowledge base.", [])
            
            context = "\n".join([f"[Source: {n.node.metadata.get('source', 'Unknown')}]\n{n.node.get_content()}" for n in ranked_nodes])
            
            prompt = f"""You are Lilly-X. Answer based ONLY on the context below.

Context:
{context}

Question: {user_query}

Answer:"""
            
            response = await Settings.llm.acomplete(prompt)
            return RAGResponse(response.text.strip(), ranked_nodes, {"top_k": top_k})
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            retriever = self._index.as_retriever(similarity_top_k=top_n)
            nodes = await retriever.aretrieve(user_query)
            context = "\n".join([n.get_content() for n in nodes])
            response = await Settings.llm.acomplete(f"Context:\n{context}\n\nQ: {user_query}\n\nA:")
            return RAGResponse(response.text.strip(), nodes, {"error": str(e)})

    def query(self, *args, **kwargs):
        """
        Synchronous wrapper that safely uses the existing event loop.
        Avoids asyncio.run() to prevent closing loops used by cached clients.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.aquery(*args, **kwargs))
EOF

# 2. Patch App (The Consumer)
echo "🖥️  Updating src/app.py..."
cat << 'EOF' > src/app.py
import sys
import os
import time
import asyncio
import streamlit as st
from pathlib import Path
from requests.exceptions import ConnectionError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, setup_environment
from src.rag_engine import RAGEngine, RAGResponse

st.set_page_config(page_title="Lilly-X | Garden Edition", page_icon="🌿", layout="wide")

@st.cache_resource
def initialize_system():
    try:
        setup_environment()
        engine = RAGEngine(enable_decomposition=True, verbose=True)
        return engine, "success"
    except Exception as e:
        return None, str(e)

engine, status = initialize_system()

if not engine:
    st.error(f"System Failed: {status}")
    st.stop()

st.title("🌿 Lilly-X @ The Garden")
st.caption(f"Brain: {settings.llm_model} | Graph: Neo4j | Vector: Qdrant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask the Garden..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking (Async GraphRAG)..."):
            # SIMPLIFIED CALL: Trust the engine's safe loop handling
            response_obj = engine.query(prompt)
            
            response_text = response_obj.response
            st.markdown(response_text)
            
            # Metadata
            sources = len(response_obj.source_nodes)
            st.caption(f"📚 Sources: {sources} | ⚡ Strategy: Hybrid")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text
            })
EOF

chmod +x src/rag_engine.py src/app.py
echo "✅ EVENT LOOP FIX APPLIED."
echo "👉 Restart Streamlit: 'streamlit run src/app.py'"
