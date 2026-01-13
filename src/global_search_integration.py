"""
Global Search Implementation for Microsoft-style GraphRAG.

This module provides the global_search method and routing logic to be integrated
into src/rag_engine.py for community-based retrieval.
"""

import logging
from typing import List
from llama_index.core import Settings
from src.graph_ops import GraphOperations
from src.schemas import QueryIntent

logger = logging.getLogger(__name__)


# ============================================================
# Global Search Method (Add to RAGEngine class)
# ============================================================

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
        import json
        import json_repair
        
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


# ============================================================
# Updated Query Routing Logic (Modify RAGEngine.query method)
# ============================================================

def query_with_routing(self, query_text: str, top_k: int = None, use_reranker: bool = True):
    """
    Execute a query with routing based on QueryIntent.
    
    ROUTING LOGIC:
    - If QueryIntent is GLOBAL_DISCOVERY: Execute global_search()
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
    
    # Step 1: Plan query first to determine intent
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
        from src.rag_engine import RAGResponse
        return RAGResponse(
            response=response_text,
            source_nodes=[],  # No vector nodes for global search
            query_plan=query_plan
        )
    
    else:
        logger.info("📊 Standard query intent - routing to vector/graph search")
        
        # Execute standard retrieval (existing logic)
        if use_reranker and settings.retrieval_strategy == "semantic":
            vector_nodes, graph_context = self.retrieve(query_text, top_k=top_k, use_reranker=True)
            
            context_str = "\\n\\n".join([node.get_content() for node in vector_nodes])
            if graph_context:
                context_str = f"{graph_context}\\n\\n{context_str}"
            
            prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {query_text}

Answer:"""
            
            response_text = str(Settings.llm.complete(prompt))
            
            from src.rag_engine import RAGResponse
            return RAGResponse(
                response=response_text,
                source_nodes=vector_nodes[:top_k] if top_k else vector_nodes,
                query_plan=query_plan
            )
        else:
            response_obj = self._query_engine.query(query_text)
            
            from src.rag_engine import RAGResponse
            return RAGResponse(
                response=str(response_obj),
                source_nodes=response_obj.source_nodes[:top_k] if top_k else response_obj.source_nodes,
                query_plan=query_plan
            )


# ============================================================
# Integration Instructions
# ============================================================
"""
TO INTEGRATE INTO src/rag_engine.py:

1. Add these three methods to the RAGEngine class:
   - global_search()
   - _extract_keywords_for_global_search()
   - _fallback_keyword_extraction()

2. REPLACE the existing query() method with query_with_routing()
   (or modify existing query() to include the routing logic from query_with_routing())

3. Add this import at the top:
   from src.schemas import QueryIntent

4. The routing logic will automatically:
   - Detect GLOBAL_DISCOVERY intent via query planning
   - Route to global_search() for abstract queries
   - Use standard retrieval for specific queries

EXAMPLE USAGE:

# In src/app.py or wherever you call the RAG engine:
rag_engine = RAGEngine()

# This will automatically route based on intent
response = rag_engine.query(user_query)

# Global query example (will use community summaries):
"What are the main themes covered in this knowledge base?"

# Specific query example (will use standard retrieval):
"How does the authentication module work in Flask?"
"""
