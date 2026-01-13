"""
Core RAG Engine logic.
Encapsulates LlamaIndex setup and querying with support for multiple retrieval strategies.
"""

import logging
import json  # For exception handling (json.JSONDecodeError)
import json_repair  # More robust JSON parsing for LLM outputs
from typing import Tuple, List, Any
from dataclasses import dataclass

# Core LlamaIndex imports (REQUIRED)
from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# LLM and Embeddings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Vector Store
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Reranker (optional, loaded lazily)
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sentence_transformers not available - reranking will be disabled")

from src.config import settings
from src.database import get_qdrant_client
from src.graph_database import get_neo4j_driver
from src.schemas import QueryPlan, SubQuery, QueryIntent

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    response: str
    source_nodes: List[NodeWithScore]
    query_plan: 'QueryPlan' = None  # Expose query decomposition for UI visualization


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
        Execute a query with intelligent routing based on QueryIntent.
        
        ROUTING LOGIC:
        - If QueryIntent is GLOBAL_DISCOVERY: Execute global_search() for community summaries
        - Otherwise: Execute standard vector/graph search
        
        Args:
            query_text: Query string
            top_k: Number of final results to return (defaults to settings.top_k_final)
            use_reranker: Whether to use reranker for two-stage retrieval
            
        Returns:
            RAGResponse with answer, source nodes, and query plan
        """
        if not self._query_engine:
            raise RuntimeError("RAG Engine not initialized.")
            
        logger.info(f"Querying with '{settings.retrieval_strategy}' strategy: {query_text}")
        
        # Step 1: Plan query first to determine intent and expose decomposition
        query_plan = self.plan_query(query_text)
        
        # Step 2: Check if this is a GLOBAL_DISCOVERY query
        # Heuristic: If ANY sub-query has GLOBAL_DISCOVERY intent, use global search
        has_global_intent = any(
            sub_query.intent == QueryIntent.GLOBAL_DISCOVERY
            for sub_query in query_plan.sub_queries
        )
        
        if has_global_intent:
            logger.info("🌐 Detected GLOBAL_DISCOVERY intent - routing to global search")
            
            # Execute global search (community-based retrieval)
            response_text = self.global_search(query_text)
            
            # Return response without source nodes (global search doesn't use vector nodes)
            return RAGResponse(
                response=response_text,
                source_nodes=[],  # No vector nodes for global search
                query_plan=query_plan
            )
        
        # Step 3: Standard retrieval for specific queries
        logger.info("📊 Standard query intent - using vector/graph search")
        
        # For hierarchical and sentence_window strategies, the query engine handles the special logic
        # For reranking integration, we use the retrieve() method instead
        if use_reranker and settings.retrieval_strategy == "semantic":
            # Use custom retrieve + rerank flow for semantic strategy
            # Note: retrieve() now handles query planning internally
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
                source_nodes=vector_nodes[:top_k] if top_k else vector_nodes,
                query_plan=query_plan
            )
        else:
            # Use the query engine directly (handles sentence_window and hierarchical internally)
            response_obj = self._query_engine.query(query_text)
            
            return RAGResponse(
                response=str(response_obj),
                source_nodes=response_obj.source_nodes[:top_k] if top_k else response_obj.source_nodes,
                query_plan=query_plan
            )

    def global_search(self, query: str) -> str:
        """
        Execute global search using community summaries instead of vector search.
        
        This method is designed for abstract, high-level queries (GLOBAL_DISCOVERY intent)
        where community-level context is more appropriate than entity-level details.
        
        Workflow:
        1. Extract keywords from user query using LLM
        2. Retrieve relevant Community Summaries via graph_ops.get_community_context()
        3. Synthesize answer based ONLY on community summaries (no vector search)
        
        Args:
            query: User query string (typically abstract/high-level)
            
        Returns:
            Synthesized answer based on community summaries
        """
        logger.info(f"🌐 Executing GLOBAL SEARCH for: {query}")
        
        # Step 1: Extract keywords from query
        keywords = self._extract_keywords_for_global_search(query)
        logger.info(f"📌 Extracted keywords: {keywords}")
        
        if not keywords:
            logger.warning("No keywords extracted from query. Falling back to standard search.")
            # Fallback to standard query if keyword extraction fails
            response = self.query(query, use_reranker=False)
            return response.response
        
        # Step 2: Retrieve Community Summaries
        try:
            from src.graph_ops import GraphOperations
            graph_ops = GraphOperations(driver=self._neo4j_driver)
            community_summaries = graph_ops.get_community_context(
                keywords=keywords,
                top_k=5  # Retrieve top 5 most relevant communities
            )
            
            if not community_summaries:
                logger.warning("No community summaries found. Graph may not have communities yet.")
                return (
                    "I cannot provide a high-level overview because community detection "
                    "has not been run yet. Please run the community summarization pipeline first, "
                    "or ask a more specific question."
                )
            
            logger.info(f"✅ Retrieved {len(community_summaries)} community summaries")
            
        except Exception as e:
            logger.error(f"Failed to retrieve community summaries: {e}")
            return f"Error retrieving global context: {str(e)}"
        
        # Step 3: Synthesize answer using ONLY community summaries
        context = "\n\n".join([
            f"Community {idx + 1}:\n{summary}"
            for idx, summary in enumerate(community_summaries)
        ])
        
        prompt = f"""You are providing a high-level overview based on community summaries from a knowledge graph.

**Community Summaries:**
{context}

**User Question:**
{query}

**Instructions:**
1. Synthesize an answer using ONLY the information from the community summaries above
2. Provide a broad, high-level overview that addresses the user's question
3. If the summaries don't fully answer the question, acknowledge the limitations
4. Be concise but comprehensive
5. Do NOT invent details not present in the summaries

**Answer:**"""
        
        try:
            response = Settings.llm.complete(prompt)
            answer = response.text.strip()
            
            logger.info(f"✅ Global search response generated ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"LLM generation failed during global search: {e}")
            return f"Error generating global search response: {str(e)}"
    
    def _extract_keywords_for_global_search(self, query: str) -> List[str]:
        """
        Extract keywords from a query for community matching.
        
        Uses the LLM to identify key concepts and topics in the user's query
        that can be matched against community keywords.
        
        Args:
            query: User query string
            
        Returns:
            List of extracted keywords (3-7 keywords)
        """
        keyword_extraction_prompt = f"""Extract 3-7 key concepts, topics, or themes from the following question.
These keywords will be used to find relevant communities in a knowledge graph.

Question: {query}

Return ONLY a JSON array of keywords (lowercase, singular form when possible).
Example: ["machine learning", "neural networks", "training", "optimization"]

Keywords:"""
        
        try:
            response = Settings.llm.complete(keyword_extraction_prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            # Clean markdown code blocks
            if "```json" in response_text or "```" in response_text:
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON array
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            # Try to parse
            try:
                keywords = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback to json_repair
                keywords = json_repair.loads(response_text)
            
            # Validate and clean
            if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                # Normalize: lowercase, strip whitespace
                keywords = [k.lower().strip() for k in keywords if k.strip()]
                logger.debug(f"Extracted {len(keywords)} keywords via LLM: {keywords}")
                return keywords
            else:
                logger.warning(f"Invalid keyword format from LLM: {keywords}")
                return self._fallback_keyword_extraction(query)
                
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}. Using fallback.")
            return self._fallback_keyword_extraction(query)
    
    def _fallback_keyword_extraction(self, query: str) -> List[str]:
        """
        Simple fallback keyword extraction using word filtering.
        
        Args:
            query: User query string
            
        Returns:
            List of extracted keywords
        """
        # Remove common stop words
        stop_words = {
            'what', 'are', 'is', 'the', 'how', 'why', 'when', 'where', 'who',
            'tell', 'me', 'about', 'can', 'you', 'please', 'give', 'show',
            'explain', 'describe', 'a', 'an', 'in', 'on', 'for', 'to', 'of',
            'this', 'that', 'these', 'those', 'do', 'does', 'did'
        }
        
        # Extract words longer than 3 characters
        words = query.lower().split()
        keywords = [
            word.strip('.,!?;:')
            for word in words
            if len(word) > 3 and word.lower() not in stop_words
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        logger.debug(f"Fallback extraction: {unique_keywords}")
        return unique_keywords[:7]  # Limit to 7 keywords


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
            try:
                from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
                self._reranker = FlagEmbeddingReranker(
                    model=settings.reranker_model,
                    top_n=settings.top_k_final,
                )
                logger.info(f"Loaded reranker: {settings.reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
        return self._reranker
    
    def plan_query(self, query_text: str) -> QueryPlan:
        """
        Decompose complex queries into sub-queries using LLM.
        
        Uses the QUERY_PLAN_PROMPT to analyze the query and determine if it should
        be split into multiple atomic sub-queries for better retrieval.
        
        Args:
            query_text: Original user query
            
        Returns:
            QueryPlan with decomposed sub-queries (or single sub-query for simple queries)
        """
        try:
            from src.prompts import get_query_plan_prompt
            
            # Build prompt using function (avoids .format() KeyError with JSON braces)
            prompt = get_query_plan_prompt(query_text)
            
            # Call LLM
            logger.info(f"🧠 Planning query decomposition for: {query_text[:100]}...")
            response = Settings.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Clean response: strip markdown code blocks and extra formatting
            # Remove ```json and ``` markers
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON object if there's extra text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            # Parse JSON using json_repair for robustness (handles trailing commas, missing quotes, etc.)
            plan_dict = json_repair.loads(response_text)
            
            # Enforce max sub-queries limit
            if "sub_queries" in plan_dict and len(plan_dict["sub_queries"]) > settings.planner_max_subqueries:
                logger.warning(f"Truncating {len(plan_dict['sub_queries'])} sub-queries to max {settings.planner_max_subqueries}")
                plan_dict["sub_queries"] = plan_dict["sub_queries"][:settings.planner_max_subqueries]
            
            # Create QueryPlan object
            query_plan = QueryPlan(**plan_dict)
            
            logger.info(f"✅ Query plan created with {len(query_plan.sub_queries)} sub-queries")
            for idx, sq in enumerate(query_plan.sub_queries, 1):
                logger.info(f"  Sub-query {idx} [{sq.intent}]: {sq.focused_query}")
            
            return query_plan
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse query plan JSON: {e}. Falling back to single query.")
            # Fallback: treat as single query
            return QueryPlan(
                root_query=query_text,
                sub_queries=[
                    SubQuery(
                        original_text=query_text,
                        focused_query=query_text,
                        intent=QueryIntent.FACTUAL
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Query planning failed: {e}. Falling back to single query.")
            # Fallback: treat as single query
            return QueryPlan(
                root_query=query_text,
                sub_queries=[
                    SubQuery(
                        original_text=query_text,
                        focused_query=query_text,
                        intent=QueryIntent.FACTUAL
                    )
                ]
            )

    def retrieve(self, query_text: str, top_k: int = None, use_reranker: bool = True) -> Tuple[List[NodeWithScore], str]:
        """
        Hybrid retrieval with query planning, multi-hop graph traversal, and configurable reranking.
        
        This method now decomposes complex queries into sub-queries, retrieves from vector and graph stores
        for each sub-query, then aggregates and deduplicates results.
        
        Args:
            query_text: Query string
            top_k: Number of final results to return (defaults to settings.top_k_final)
            use_reranker: If True, use two-stage retrieval (broad -> rerank -> top-k)
                         If False, use direct vector search with top_k results
            
        Returns:
            Tuple of (reranked_vector_nodes, combined_graph_context)
        """
        if not self._index:
            raise RuntimeError("RAG Engine not initialized.")
        
        # Use config defaults if not specified
        if top_k is None:
            top_k = settings.top_k_final
        
        # STEP 1: Query Planning - Decompose complex queries
        query_plan = self.plan_query(query_text)
        
        # STEP 2: Retrieve for each sub-query
        all_vector_nodes = []
        all_graph_facts = []
        seen_node_ids = set()  # For deduplication
        
        logger.info(f"📊 Retrieving for {len(query_plan.sub_queries)} sub-queries...")
        
        for idx, sub_query in enumerate(query_plan.sub_queries, 1):
            logger.info(f"  [{idx}/{len(query_plan.sub_queries)}] Retrieving for: {sub_query.focused_query}")
            
            # STEP 2a: Vector retrieval for this sub-query
            try:
                retriever = self._index.as_retriever(
                    similarity_top_k=top_k * 2  # Get extra candidates per sub-query
                )
                sub_vector_nodes = retriever.retrieve(sub_query.focused_query)
                
                # Deduplicate by node ID
                for node in sub_vector_nodes:
                    if node.id_ not in seen_node_ids:
                        seen_node_ids.add(node.id_)
                        all_vector_nodes.append(node)
                
                logger.info(f"    Vector: Retrieved {len(sub_vector_nodes)} nodes, {len(all_vector_nodes)} total unique")
                
            except Exception as e:
                logger.error(f"    Vector retrieval failed for sub-query: {e}")
            
            # STEP 2b: Graph retrieval for this sub-query
            try:
                sub_graph_context = self.query_graph_store(sub_query.focused_query)
                if sub_graph_context:
                    # Extract individual facts from context
                    facts = [line.strip() for line in sub_graph_context.split('\n') if line.strip().startswith('-')]
                    all_graph_facts.extend(facts)
                    logger.info(f"    Graph: Retrieved {len(facts)} facts, {len(all_graph_facts)} total")
            except Exception as e:
                logger.error(f"    Graph retrieval failed for sub-query: {e}")
        
        # STEP 3: Deduplicate graph facts
        unique_graph_facts = list(dict.fromkeys(all_graph_facts))  # Preserve order, remove duplicates
        logger.info(f"✅ Aggregated {len(all_vector_nodes)} unique vector nodes and {len(unique_graph_facts)} unique graph facts")
        
        # STEP 4: Apply reranking to combined vector nodes
        if use_reranker and len(all_vector_nodes) > top_k:
            logger.info(f"🔄 Reranking {len(all_vector_nodes)} aggregated nodes to top {top_k}")
            vector_nodes = self._apply_reranking(query_text, all_vector_nodes, top_k)
        else:
            vector_nodes = all_vector_nodes[:top_k]
        
        # STEP 5: Format combined graph context
        if unique_graph_facts:
            context_lines = ["\n=== Related Knowledge Graph Information (Multi-Hop) ==="]
            context_lines.extend(unique_graph_facts)
            context_lines.append("=" * 60)
            graph_context = "\n".join(context_lines)
        else:
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
