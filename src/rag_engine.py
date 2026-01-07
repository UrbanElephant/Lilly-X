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
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder
except ImportError:
    # Allow import for type checking/linting even if missing
    pass

from src.config import settings
from src.database import get_qdrant_client
from src.graph_database import get_neo4j_driver

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
        self._reranker = None  # Lazy-loaded reranker model
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

        # 4. Initialize Neo4j driver for graph queries
        try:
            logger.info("Initializing Neo4j connection for hybrid retrieval...")
            self._neo4j_driver = get_neo4j_driver()
            logger.info("Neo4j connection ready")
        except Exception as e:
            logger.warning(f"Neo4j initialization failed: {e}. Graph queries will be disabled.")
            self._neo4j_driver = None

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
            
            # Configure query engine for vector retrieval
            logger.info(f"Vector retrieval configured: top_k={settings.top_k}")
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=settings.top_k
            )
            
            logger.info("RAG Engine initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant/Index: {e}")
            raise

    def query(self, query_text: str, top_k: int = None, use_reranker: bool = True) -> RAGResponse:
        """
        Execute a query and return response with sources.
        
        Args:
            query_text: Query string
            top_k: Number of final results to return (defaults to settings.top_k_final)
            use_reranker: Whether to use reranker for two-stage retrieval
            
        Returns:
            RAGResponse with answer and source nodes
        """
        if not self._query_engine:
            raise RuntimeError("RAG Engine not initialized.")
            
        logger.info(f"Querying: {query_text}")
        
        # Use retrieve for configurable retrieval, then query engine for response generation
        # This is a simpler interface that uses the built-in query engine
        # For more control, use retrieve() + custom prompting
        response_obj = self._query_engine.query(query_text)
        
        return RAGResponse(
            response=str(response_obj),
            source_nodes=response_obj.source_nodes[:top_k] if top_k else response_obj.source_nodes
        )

    def query_graph_store(self, query_text: str) -> str:
        """
        Query Neo4j graph database for entities and relationships related to the query.
        
        Args:
            query_text: User query string
            
        Returns:
            Formatted string with graph context, or empty string if no results/errors
        """
        if not self._neo4j_driver:
            logger.debug("Neo4j driver not available, skipping graph query")
            return ""
        
        try:
            # Extract key entities from query using LLM
            extract_prompt = f"""Extract the key entities, topics, or concepts from this query.
Return only a comma-separated list of 2-4 key terms, nothing else.

Query: {query_text}

Key terms:"""
            
            response = Settings.llm.complete(extract_prompt)
            keywords = str(response).strip().split(",")
            keywords = [k.strip().lower() for k in keywords if k.strip()]
            
            if not keywords:
                logger.debug("No keywords extracted from query")
                return ""
            
            logger.info(f"Graph query keywords: {keywords[:3]}")
            
            # Query Neo4j for related entities and relationships
            with self._neo4j_driver.session() as session:
                # Build Cypher query to find related nodes
                cypher_query = """
                MATCH (n)-[r]->(m)
                WHERE toLower(n.name) CONTAINS $keyword
                RETURN n.name AS source, type(r) AS relationship, m.name AS target, 
                       n.entity_type AS source_type, m.entity_type AS target_type
                LIMIT 20
                """
                
                all_results = []
                for keyword in keywords[:3]:  # Limit to first 3 keywords
                    result = session.run(cypher_query, keyword=keyword)
                    all_results.extend(list(result))
                
                if not all_results:
                    logger.debug("No graph results found")
                    return ""
                
                # Format results as context
                context_lines = ["\n=== Related Knowledge Graph Information ==="]
                for record in all_results[:15]:  # Limit to 15 relationships
                    source = record.get("source", "Unknown")
                    rel = record.get("relationship", "RELATED_TO")
                    target = record.get("target", "Unknown")
                    context_lines.append(f"- {source} {rel} {target}")
                
                context_lines.append("="*45)
                return "\n".join(context_lines)
                
        except Exception as e:
            logger.warning(f"Graph query error: {e}")
            return ""

    def _get_reranker(self):
        """Lazy-load the reranker model (singleton pattern)."""
        if self._reranker is None:
            logger.info(f"Loading reranker model: {settings.reranker_model}")
            try:
                self._reranker = CrossEncoder(settings.reranker_model, max_length=512)
                logger.info("Reranker model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise
        return self._reranker

    def retrieve(self, query_text: str, top_k: int = None, use_reranker: bool = True) -> Tuple[List[NodeWithScore], str]:
        """
        Hybrid retrieval with configurable reranking.
        
        Args:
            query_text: Query string
            top_k: Number of final results to return (defaults to settings.top_k_final)
            use_reranker: If True, use two-stage retrieval (broad -> rerank -> top-k)
                         If False, use direct vector search with top_k results
            
        Returns:
            Tuple of (reranked_vector_nodes, graph_context)
        """
        if not self._index:
            raise RuntimeError("RAG Engine not initialized.")
        
        # Use config defaults if not specified
        if top_k is None:
            top_k = settings.top_k_final
        
        if use_reranker:
            # STAGE 1: Broad retrieval - fetch more candidates for reranking
            candidates_count = top_k * 3  # Fetch 3x for better reranking pool
            logger.info(f"Two-stage retrieval: Fetching {candidates_count} candidates for reranking to top {top_k}")
            retriever = self._index.as_retriever(similarity_top_k=candidates_count)
            candidate_nodes = retriever.retrieve(query_text)
            
            # STAGE 2: Rerank using CrossEncoder
            if len(candidate_nodes) > top_k:
                logger.info(f"Reranking {len(candidate_nodes)} candidates to top {top_k}")
                try:
                    reranker = self._get_reranker()
                    
                    # Create (query, document) pairs for reranking
                    pairs = [(query_text, node.get_content()) for node in candidate_nodes]
                    
                    # Get reranking scores
                    rerank_scores = reranker.predict(pairs)
                    
                    # Sort nodes by reranking score (descending)
                    scored_nodes = list(zip(candidate_nodes, rerank_scores))
                    scored_nodes.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top-K after reranking
                    vector_nodes = [node for node, score in scored_nodes[:top_k]]
                    
                    logger.info(f"Reranking complete: selected top {len(vector_nodes)} documents")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}. Using original retrieval.")
                    vector_nodes = candidate_nodes[:top_k]
            else:
                # Not enough candidates to rerank, use all retrieved
                vector_nodes = candidate_nodes
                logger.info(f"Retrieved {len(vector_nodes)} candidates (< top_k, skipping rerank)")
        else:
            # Direct vector search without reranking
            logger.info(f"Direct retrieval: Fetching top {top_k} results (reranker disabled)")
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            vector_nodes = retriever.retrieve(query_text)
        
        # Graph retrieval (with fallback)
        try:
            graph_context = self.query_graph_store(query_text)
        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")
            graph_context = ""
        
        return vector_nodes, graph_context
