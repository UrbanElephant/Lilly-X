"""
Core RAG Engine logic.
Encapsulates LlamaIndex setup and querying with support for multiple retrieval strategies.
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
    from llama_index.core.postprocessor import MetadataReplacementPostProcessor
    from llama_index.core.retrievers import AutoMergingRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
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
    """Singleton-style class to handle RAG operations with multiple retrieval strategies."""
    
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
        self._storage_context = None  # Needed for hierarchical retrieval
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
            self._storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self._index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=self._storage_context,
            )
            
            # Configure query engine based on retrieval strategy
            self._query_engine = self._create_query_engine()
            
            logger.info(f"RAG Engine initialized successfully with '{settings.retrieval_strategy}' strategy.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant/Index: {e}")
            raise

    def _create_query_engine(self):
        """
        Create a query engine based on the configured retrieval strategy.
        
        Returns:
            Configured query engine instance
        """
        strategy = settings.retrieval_strategy
        logger.info(f"Creating query engine with strategy: {strategy}")
        
        # Build node postprocessors list
        node_postprocessors = []
        
        if strategy == "semantic":
            # Standard semantic search - no special postprocessors needed
            logger.info("Using standard semantic retrieval")
            return self._index.as_query_engine(
                similarity_top_k=settings.top_k,
                node_postprocessors=node_postprocessors
            )
        
        elif strategy == "sentence_window":
            # Sentence Window: Replace sentence nodes with their surrounding context window
            logger.info(f"Using sentence window retrieval (window_size={settings.sentence_window_size})")
            
            # Add metadata replacement postprocessor BEFORE any other processing
            # This expands the sentence to its full window context
            try:
                window_postprocessor = MetadataReplacementPostProcessor(
                    target_metadata_key="window"
                )
                node_postprocessors.append(window_postprocessor)
                logger.info("MetadataReplacementPostProcessor configured for sentence window")
            except Exception as e:
                logger.error(f"Failed to initialize MetadataReplacementPostProcessor: {e}")
                logger.warning("Falling back to standard retrieval without window replacement")
            
            return self._index.as_query_engine(
                similarity_top_k=settings.top_k,
                node_postprocessors=node_postprocessors
            )
        
        elif strategy == "hierarchical":
            # Hierarchical: Use AutoMergingRetriever to merge child chunks into parents
            logger.info(f"Using hierarchical retrieval (parent={settings.parent_chunk_size}, child={settings.child_chunk_size})")
            
            try:
                # Create base retriever
                base_retriever = self._index.as_retriever(
                    similarity_top_k=settings.top_k
                )
                
                # Wrap with AutoMergingRetriever
                retriever = AutoMergingRetriever(
                    base_retriever,
                    self._storage_context,
                    verbose=True
                )
                
                # Create query engine from the auto-merging retriever
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    node_postprocessors=node_postprocessors
                )
                
                logger.info("AutoMergingRetriever configured successfully")
                return query_engine
                
            except Exception as e:
                logger.error(f"Failed to initialize AutoMergingRetriever: {e}")
                logger.warning("Falling back to standard semantic retrieval")
                return self._index.as_query_engine(similarity_top_k=settings.top_k)
        
        else:
            logger.warning(f"Unknown retrieval strategy '{strategy}', using semantic fallback")
            return self._index.as_query_engine(similarity_top_k=settings.top_k)

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
            
        logger.info(f"Querying with '{settings.retrieval_strategy}' strategy: {query_text}")
        
        # For hierarchical and sentence_window strategies, the query engine handles the special logic
        # For reranking integration, we use the retrieve() method instead
        if use_reranker and settings.retrieval_strategy == "semantic":
            # Use custom retrieve + rerank flow for semantic strategy
            vector_nodes, graph_context = self.retrieve(query_text, top_k=top_k, use_reranker=True)
            
            # Generate response using LLM with retrieved context
            context_str = "\n\n".join([node.get_content() for node in vector_nodes])
            if graph_context:
                context_str = f"{graph_context}\n\n{context_str}"
            
            prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {query_text}

Answer:"""
            
            response_text = str(Settings.llm.complete(prompt))
            
            return RAGResponse(
                response=response_text,
                source_nodes=vector_nodes[:top_k] if top_k else vector_nodes
            )
        else:
            # Use the query engine directly (handles sentence_window and hierarchical internally)
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
            # Import GraphOperations for entity context retrieval
            from src.graph_ops import GraphOperations
            
            graph_ops = GraphOperations(driver=self._neo4j_driver)
            
            # Get entity context using heuristic extraction
            facts = graph_ops.get_entity_context(query_text, limit=15)
            
            if not facts:
                logger.debug("No graph facts found for query")
                return ""
            
            # Format results as context
            context_lines = ["\n=== Related Knowledge Graph Information ==="]
            for fact in facts:
                context_lines.append(f"- {fact}")
            
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
        Hybrid retrieval with configurable reranking and strategy-aware processing.
        
        Note: For sentence_window and hierarchical strategies, use the query() method instead
        as it handles the special postprocessing automatically.
        
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
        
        # Strategy-specific retrieval
        strategy = settings.retrieval_strategy
        
        if strategy == "hierarchical":
            logger.warning("retrieve() called with hierarchical strategy. Use query() instead for proper auto-merging.")
            # Fall back to base retrieval for manual use
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            vector_nodes = retriever.retrieve(query_text)
            
        elif strategy == "sentence_window":
            logger.info(f"Retrieving with sentence_window strategy")
            # Retrieve candidate nodes
            retriever = self._index.as_retriever(similarity_top_k=top_k * 3 if use_reranker else top_k)
            candidate_nodes = retriever.retrieve(query_text)
            
            # Apply window replacement
            try:
                postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
                vector_nodes = postprocessor.postprocess_nodes(candidate_nodes)
                logger.info(f"Applied MetadataReplacementPostProcessor to {len(vector_nodes)} nodes")
            except Exception as e:
                logger.warning(f"Window replacement failed: {e}. Using original nodes.")
                vector_nodes = candidate_nodes
            
            # Apply reranking if enabled
            if use_reranker and len(vector_nodes) > top_k:
                vector_nodes = self._apply_reranking(query_text, vector_nodes, top_k)
            else:
                vector_nodes = vector_nodes[:top_k]
                
        else:  # semantic or unknown
            if use_reranker:
                # STAGE 1: Broad retrieval - fetch more candidates for reranking
                candidates_count = top_k * 3  # Fetch 3x for better reranking pool
                logger.info(f"Two-stage retrieval: Fetching {candidates_count} candidates for reranking to top {top_k}")
                retriever = self._index.as_retriever(similarity_top_k=candidates_count)
                candidate_nodes = retriever.retrieve(query_text)
                
                # STAGE 2: Rerank using CrossEncoder
                if len(candidate_nodes) > top_k:
                    vector_nodes = self._apply_reranking(query_text, candidate_nodes, top_k)
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

    def _apply_reranking(self, query_text: str, candidate_nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        """
        Apply reranking to candidate nodes and return top-k.
        
        Args:
            query_text: Query string
            candidate_nodes: List of candidate nodes to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            List of top-k reranked nodes
        """
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
            reranked_nodes = [node for node, score in scored_nodes[:top_k]]
            
            logger.info(f"Reranking complete: selected top {len(reranked_nodes)} documents")
            return reranked_nodes
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original retrieval order.")
            return candidate_nodes[:top_k]
