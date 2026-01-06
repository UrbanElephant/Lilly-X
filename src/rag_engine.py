"""
Core RAG Engine logic.
Encapsulates LlamaIndex setup and querying.
"""

import logging
from typing import Tuple, List, Any
from dataclasses import dataclass

# Conditional imports
try:
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.schema import NodeWithScore
    from llama_index.postprocessor.huggingface_rerank import HuggingFaceRerank
    from qdrant_client import QdrantClient
except ImportError:
    # Allow import for type checking/linting even if missing
    pass

from src.config import settings
from src.database import get_qdrant_client

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    response: str
    source_nodes: List[NodeWithScore]


class RAGEngine:
    """Singleton-style class to handle RAG operations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._index = None
        self._query_engine = None
        self._initialize()
        self._initialized = True

    def _initialize(self):
        """Initialize LlamaIndex components."""
        logger.info("Initializing RAG Engine...")
        
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
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # 3. Configure Global Settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = settings.chunk_size

        # 4. Initialize Reranker for Two-Stage Retrieval
        try:
            logger.info(f"Loading Reranker: {settings.reranker_model}")
            reranker = HuggingFaceRerank(
                model=settings.reranker_model,
                top_n=settings.top_k_final  # Final top 5 results after reranking
            )
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Falling back to single-stage retrieval.")
            reranker = None

        # 5. Connect to Vector Store
        try:
            client = get_qdrant_client()
            vector_store = QdrantVectorStore(
                client=client, 
                collection_name=settings.qdrant_collection
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self._index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )
            
            # Configure query engine with two-stage retrieval
            # Stage 1: Fetch 25 candidates (broad net)
            # Stage 2: Rerank to top 5 (high precision)
            if reranker:
                logger.info(f"Two-Stage Retrieval enabled: Fetch {settings.top_k_retrieval} → Rerank to {settings.top_k_final}")
                self._query_engine = self._index.as_query_engine(
                    similarity_top_k=settings.top_k_retrieval,  # Fetch 25 candidates
                    node_postprocessors=[reranker]  # Rerank to top 5
                )
            else:
                logger.info(f"Single-stage retrieval: top_k={settings.top_k}")
                self._query_engine = self._index.as_query_engine(
                    similarity_top_k=settings.top_k
                )
            
            logger.info("RAG Engine initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant/Index: {e}")
            raise

    def query(self, query_text: str) -> RAGResponse:
        """Execute a query and return response with sources."""
        if not self._query_engine:
            raise RuntimeError("RAG Engine not initialized.")
            
        logger.info(f"Querying: {query_text}")
        
        # Get response
        response_obj = self._query_engine.query(query_text)
        
        return RAGResponse(
            response=str(response_obj),
            source_nodes=response_obj.source_nodes
        )

    def retrieve(self, query_text: str) -> List[NodeWithScore]:
        """Retrieve relevant context nodes without generating an answer."""
        if not self._index:
            raise RuntimeError("RAG Engine not initialized.")
            
        retriever = self._index.as_retriever(similarity_top_k=settings.top_k)
        return retriever.retrieve(query_text)
