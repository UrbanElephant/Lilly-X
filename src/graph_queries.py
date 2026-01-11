"""Advanced Cypher queries for Multi-Hop GraphRAG retrieval.

This module defines Cypher query templates for advanced graph traversal and entity
search operations, supporting the Reasoning-GraphRAG upgrade.
"""

# ============================================================
# Multi-Hop Context Retrieval
# ============================================================

CYPHER_MULTI_HOP_CONTEXT = """
// Multi-Hop Context Retrieval Query
// Finds a starting node and traverses 1-2 hops to find related context
// Returns the complete path for reasoning chain construction

MATCH (start:Entity)
WHERE start.name = $entity_name
  OR $entity_name IN start.aliases

// Variable-length path traversal (1-2 hops)
MATCH path = (start)-[r*1..2]-(related:Entity)

// Filter out irrelevant node types if specified
WHERE NOT related:Document 
  OR $include_documents = true

// Return the path with metadata
WITH path, start, related, relationships(path) as rels

RETURN 
  start.name AS source_entity,
  start.entity_type AS source_type,
  start.description AS source_description,
  related.name AS related_entity,
  related.entity_type AS related_type,
  related.description AS related_description,
  [rel in rels | type(rel)] AS relationship_chain,
  length(path) AS hop_distance,
  path

ORDER BY hop_distance ASC, related.confidence_score DESC
LIMIT $max_results
"""


# ============================================================
# Entity Search with Fuzzy Matching
# ============================================================

CYPHER_ENTITY_SEARCH_FUZZY = """
// Entity Search with Fuzzy/Fulltext Matching
// Prepares for removal of Regex heuristic by using native Neo4j text search
// Supports both exact and fuzzy matching for entity discovery

// Option 1: If fulltext index exists (recommended for production)
// CALL db.index.fulltext.queryNodes("entity_fulltext_index", $search_term)
// YIELD node, score
// WHERE node:Entity
// RETURN node.name AS entity_name,
//        node.entity_type AS entity_type,
//        node.description AS description,
//        node.aliases AS aliases,
//        score
// ORDER BY score DESC
// LIMIT $max_results

// Option 2: Fallback using pattern matching (for systems without fulltext index)
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($search_term)
  OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($search_term))
  OR toLower(e.description) CONTAINS toLower($search_term)

// Calculate a simple relevance score
WITH e,
  CASE 
    WHEN toLower(e.name) = toLower($search_term) THEN 100
    WHEN toLower(e.name) STARTS WITH toLower($search_term) THEN 90
    WHEN any(alias IN e.aliases WHERE toLower(alias) = toLower($search_term)) THEN 85
    WHEN toLower(e.name) CONTAINS toLower($search_term) THEN 70
    WHEN any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($search_term)) THEN 60
    WHEN toLower(e.description) CONTAINS toLower($search_term) THEN 50
    ELSE 0
  END AS relevance_score

WHERE relevance_score > 0

RETURN 
  e.name AS entity_name,
  e.entity_type AS entity_type,
  e.description AS description,
  e.aliases AS aliases,
  e.canonical_name AS canonical_name,
  e.confidence_score AS confidence,
  relevance_score

ORDER BY relevance_score DESC, e.confidence_score DESC
LIMIT $max_results
"""


# ============================================================
# Query Parameter Helpers
# ============================================================

def get_multi_hop_params(
    entity_name: str,
    max_results: int = 10,
    include_documents: bool = False
) -> dict:
    """
    Get parameter dictionary for CYPHER_MULTI_HOP_CONTEXT query.
    
    Args:
        entity_name: Name of the starting entity to traverse from
        max_results: Maximum number of paths to return
        include_documents: Whether to include Document nodes in traversal
        
    Returns:
        Dictionary of query parameters
    """
    return {
        "entity_name": entity_name,
        "max_results": max_results,
        "include_documents": include_documents
    }


def get_fuzzy_search_params(
    search_term: str,
    max_results: int = 10
) -> dict:
    """
    Get parameter dictionary for CYPHER_ENTITY_SEARCH_FUZZY query.
    
    Args:
        search_term: Search term to find entities
        max_results: Maximum number of entities to return
        
    Returns:
        Dictionary of query parameters
    """
    return {
        "search_term": search_term,
        "max_results": max_results
    }


# ============================================================
# Additional Query Templates (for future implementation)
# ============================================================

CYPHER_SHORTEST_PATH = """
// Find shortest path between two entities (for reasoning chains)
MATCH (start:Entity {name: $start_entity}),
      (end:Entity {name: $end_entity}),
      path = shortestPath((start)-[*..5]-(end))
RETURN path, length(path) AS distance
"""


CYPHER_SUBGRAPH_EXTRACTION = """
// Extract a subgraph around an entity (for context window assembly)
MATCH (center:Entity)
WHERE center.name = $entity_name
MATCH (center)-[r*1..$depth]-(neighbor:Entity)
WITH center, collect(DISTINCT neighbor) AS neighbors, collect(DISTINCT r) AS relationships
RETURN center, neighbors, relationships
"""
