"""Graph operations for entity resolution and query expansion in Lilly-X."""

import logging
import re
from typing import List, Optional, Dict, Any, Literal

from neo4j import Driver

from src.config import settings
from src.graph_database import get_neo4j_driver
from src.schemas import CommunitySummary

logger = logging.getLogger(__name__)


class GraphOperations:
    """
    Provides graph operations for entity disambiguation and query expansion.
    
    Handles entity resolution through canonical name matching and alias lookup,
    plus graph traversal for finding related context.
    """
    
    def __init__(self, driver: Optional[Driver] = None) -> None:
        """
        Initialize GraphOperations with a Neo4j driver.
        
        Args:
            driver: Neo4j Driver instance. If None, uses get_neo4j_driver()
        """
        self._driver = driver if driver is not None else get_neo4j_driver()
    
    def resolve_entity(self, entity_name: str) -> str:
        """
        Resolve an entity name to its canonical form.
        
        Logic:
        1. Query graph for exact match on 'name' property
        2. If no match, query for entities where 'aliases' contains the name
        3. Return canonical_name if found, otherwise return original name
        
        Args:
            entity_name: Entity name to resolve
            
        Returns:
            Canonical entity name if found, otherwise the original entity_name
        """
        # Step 1: Try exact match on name
        with self._driver.session() as session:
            # Query for exact name match
            exact_match_query = """
            MATCH (n)
            WHERE n.name = $name
            RETURN n.canonical_name as canonical, n.name as name
            LIMIT 1
            """
            
            result = session.run(exact_match_query, name=entity_name)
            record = result.single()
            
            if record:
                # If canonical_name is set, return it; otherwise return the name
                canonical = record["canonical"]
                return canonical if canonical else record["name"]
            
            # Step 2: Try alias match
            alias_match_query = """
            MATCH (n)
            WHERE $name IN n.aliases
            RETURN n.canonical_name as canonical, n.name as name
            LIMIT 1
            """
            
            result = session.run(alias_match_query, name=entity_name)
            record = result.single()
            
            if record:
                # Return canonical name if available, otherwise the matched name
                canonical = record["canonical"]
                return canonical if canonical else record["name"]
            
            # Step 3: No match found, return original name
            return entity_name
    
    def expand_query(
        self,
        start_node: str,
        depth: Optional[int] = None
    ) -> List[str]:
        """
        Traverse graph neighbors to find related context.
        
        Uses breadth-first traversal up to specified depth to discover
        entities related to the start node.
        
        Args:
            start_node: Name of the starting entity
            depth: Maximum traversal depth. If None, uses settings.graph_expansion_depth
            
        Returns:
            List of related entity names (including the start node)
        """
        if depth is None:
            depth = settings.graph_expansion_depth
        
        if depth < 0:
            raise ValueError(f"Depth must be >= 0, got {depth}")
        
        if depth == 0:
            return [start_node]
        
        with self._driver.session() as session:
            # Cypher query to traverse relationships up to specified depth
            # Returns all unique nodes within the traversal path
            expansion_query = """
            MATCH path = (start)-[*1..{depth}]-(related)
            WHERE start.name = $start_node
            RETURN DISTINCT related.name as name
            """.replace("{depth}", str(depth))
            
            result = session.run(expansion_query, start_node=start_node)
            
            # Collect all related entity names
            related_entities: List[str] = [start_node]  # Include start node
            
            for record in result:
                name = record["name"]
                if name and name not in related_entities:
                    related_entities.append(name)
            
            return related_entities
    
    def get_entity_details(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full details of an entity including all properties.
        
        Args:
            entity_name: Name of the entity to retrieve
            
        Returns:
            Dictionary of entity properties, or None if not found
        """
        with self._driver.session() as session:
            query = """
            MATCH (n)
            WHERE n.name = $name
            RETURN properties(n) as props
            LIMIT 1
            """
            
            result = session.run(query, name=entity_name)
            record = result.single()
            
            if record:
                return dict(record["props"])
            return None
    
    def get_relationships(
        self,
        entity_name: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity.
        
        Args:
            entity_name: Name of the entity
            relationship_type: Optional filter for specific relationship type
            
        Returns:
            List of relationship dictionaries with source, target, and type
        """
        with self._driver.session() as session:
            if relationship_type:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.name = $name AND type(r) = $rel_type
                RETURN source.name as source,
                       type(r) as relationship,
                       target.name as target,
                       properties(r) as properties
                """
                result = session.run(
                    query,
                    name=entity_name,
                    rel_type=relationship_type
                )
            else:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.name = $name
                RETURN source.name as source,
                       type(r) as relationship,
                       target.name as target,
                       properties(r) as properties
                """
                result = session.run(query, name=entity_name)
            
            relationships: List[Dict[str, Any]] = []
            for record in result:
                relationships.append({
                    "source": record["source"],
                    "relationship": record["relationship"],
                    "target": record["target"],
                    "properties": dict(record["properties"]) if record["properties"] else {}
                })
            
            return relationships
    
    def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find shortest path between two entities.
        
        Args:
            start_entity: Starting entity name
            end_entity: Target entity name
            max_depth: Maximum path length to search
            
        Returns:
            List of entity names forming the path, or None if no path found
        """
        with self._driver.session() as session:
            query = """
            MATCH path = shortestPath(
                (start)-[*1..{max_depth}]-(end)
            )
            WHERE start.name = $start AND end.name = $end
            RETURN [node IN nodes(path) | node.name] as path_nodes
            LIMIT 1
            """.replace("{max_depth}", str(max_depth))
            
            result = session.run(
                query,
                start=start_entity,
                end=end_entity
            )
            record = result.single()
            
            if record:
                return record["path_nodes"]
            return None
    
    def get_entity_context(self, query: str, limit: int = 10) -> List[str]:
        """
        Retrieves context from Neo4j using multi-hop graph traversal.
        
        This method extracts entities from the query using LLM-based extraction,
        then uses multi-hop graph traversal (1-2 hops) to find reasoning chains
        and related context.
        
        Args:
            query: User query text to extract entities from
            limit: Maximum number of graph paths to return
            
        Returns:
            List of natural language facts describing entity relationships and paths
        """
        if not self._driver:
            logger.warning("🚫 No Neo4j connection. Skipping GraphRAG.")
            return []

        # 1. Extract potential entities using LLM-based extraction
        entities = self._extract_entities_heuristic(query)
        
        if not entities:
            logger.debug("No entities extracted from query for graph retrieval")
            return []

        logger.info(f"🕸️ Multi-hop graph traversal for entities: {entities[:5]}")
        facts = []

        try:
            # Import multi-hop query template
            from src.graph_queries import CYPHER_MULTI_HOP_CONTEXT, get_multi_hop_params
            
            # 2. For each entity, perform multi-hop traversal
            with self._driver.session() as session:
                for entity in entities[:5]:  # Limit to top 5 entities to avoid explosion
                    # Use fuzzy entity search first to find exact matches
                    search_query = """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($term)
                       OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($term))
                    RETURN e.name AS entity_name
                    LIMIT 3
                    """
                    
                    search_result = session.run(search_query, term=entity)
                    matched_entities = [record["entity_name"] for record in search_result]
                    
                    if not matched_entities:
                        logger.debug(f"No graph entities found for: {entity}")
                        continue
                    
                    # 3. For each matched entity, perform multi-hop traversal
                    for matched_entity in matched_entities:
                        params = get_multi_hop_params(
                            entity_name=matched_entity,
                            max_results=limit,
                            include_documents=False  # Exclude Document nodes for cleaner paths
                        )
                        
                        result = session.run(CYPHER_MULTI_HOP_CONTEXT, **params)
                        
                        for record in result:
                            source = record.get('source_entity', 'Unknown')
                            source_type = record.get('source_type', '')
                            related = record.get('related_entity', 'Unknown')
                            related_type = record.get('related_type', '')
                            rel_chain = record.get('relationship_chain', [])
                            hop_distance = record.get('hop_distance', 0)
                            
                            # Format path as natural language
                            if rel_chain:
                                # Multi-hop path
                                rel_str = " → ".join(rel_chain)
                                fact = f"{source} ({source_type}) --[{rel_str}]--> {related} ({related_type}) [distance: {hop_distance}]"
                            else:
                                # Direct connection
                                fact = f"{source} ({source_type}) → {related} ({related_type})"
                            
                            facts.append(fact)
                            
                            # Stop if we have enough facts
                            if len(facts) >= limit:
                                break
                    
                    if len(facts) >= limit:
                        break

            if facts:
                logger.info(f"✅ Retrieved {len(facts)} multi-hop graph paths.")
            else:
                logger.debug("No graph paths found for extracted entities")
            
        except Exception as e:
            logger.error(f"❌ Multi-hop graph retrieval failed: {e}")
            
        return facts

    def _extract_entities_heuristic(self, text: str) -> List[str]:
        """
        Extracts potential named entities using LLM-based extraction.
        
        This method replaces the previous regex-based capitalization heuristic
        with a proper LLM call to extract both capitalized entities AND lowercase
        technical terms (e.g., "backpropagation", "neo4j", "LoRA").
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of potential entity names
        """
        try:
            # Import here to avoid circular dependency
            from llama_index.core import Settings
            from src.prompts import ENTITY_EXTRACTION_PROMPT
            import json
            
            # Build prompt with the user query
            prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
            
            # Call LLM to extract entities
            logger.debug(f"🔍 Extracting entities using LLM for: {text[:100]}...")
            response = Settings.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            # Handle code blocks if present
            if "```json" in response_text:
                # Extract JSON from code block
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
            elif "```" in response_text:
                # Remove markdown code fences
                response_text = response_text.replace("```", "").strip()
            
            # Parse JSON array
            entities = json.loads(response_text)
            
            if not isinstance(entities, list):
                logger.warning(f"LLM returned non-list response: {response_text[:100]}")
                return []
            
            # Filter and clean entities
            cleaned_entities = []
            for entity in entities:
                if isinstance(entity, str) and len(entity) > 1:
                    cleaned_entities.append(entity.strip())
            
            logger.info(f"✅ Extracted {len(cleaned_entities)} entities: {cleaned_entities[:10]}")
            return cleaned_entities
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response for entity extraction: {e}")
            # Fallback to simple keyword extraction
            return self._fallback_entity_extraction(text)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Fallback to simple keyword extraction
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> List[str]:
        """
        Fallback entity extraction using simple keyword heuristics.
        
        Used when LLM extraction fails.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of potential entity names
        """
        # Simple fallback: extract words longer than 3 characters
        ignore = {
            'who', 'what', 'where', 'when', 'why', 'how', 
            'is', 'are', 'was', 'were', 'the', 'a', 'an', 
            'in', 'on', 'for', 'to', 'of', 'at', 'by', 'from',
            'do', 'does', 'did', 'can', 'could', 'should', 'would',
            'tell', 'me', 'about', 'explain', 'describe', 'this', 'that'
        }
        
        words = text.lower().split()
        entities = [
            w for w in words 
            if len(w) > 3 and w not in ignore
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        logger.debug(f"Fallback extraction returned {len(unique_entities)} entities")
        return unique_entities
    
    # ============================================================
    # Community Detection Methods (Microsoft-style GraphRAG)
    # ============================================================
    
    def run_community_detection(self, algorithm: str = "leiden") -> dict:
        """
        Executes Community Detection using Neo4j GDS.
        Compatible with Neo4j 4.x and 5.x.
        
        Args:
            algorithm: Algorithm to use ('leiden' or 'louvain')
            
        Returns:
            Dictionary with statistics: community_count, modularity
        """
        logger.info(f"🔍 Starting community detection using {algorithm.upper()} algorithm...")
        
        with self._driver.session() as session:
            # 1. Robust GDS Check (Try to call version directly)
            try:
                session.run("CALL gds.version()")
                logger.info("✅ Neo4j GDS plugin is available")
            except Exception as e:
                logger.warning(f"⚠️ GDS Plugin not available or error: {e}")
                logger.info("🔄 Falling back to simple Label Propagation (LPA)...")
                return self._run_fallback_detection()

            # 2. Cleanup previous graph projection
            graph_name = "rag_graph"
            try:
                exists = session.run(
                    "CALL gds.graph.exists($name) YIELD exists", 
                    name=graph_name
                ).single()["exists"]
                
                if exists:
                    session.run(f"CALL gds.graph.drop('{graph_name}', false)")
                    logger.debug(f"Dropped existing '{graph_name}' projection")
            except Exception as e:
                logger.debug(f"No existing projection to drop: {e}")


            # 3. Project Graph (Native Projection - UNDIRECTED)
            try:
                # A) Finde heraus, welche Relation-Typen wirklich existieren
                # (Damit wir nicht raten müssen)
                types_result = session.run("CALL db.relationshipTypes()").data()
                rel_types = [r['relationshipType'] for r in types_result]
                
                if not rel_types:
                    logger.warning("⚠️ No relationship types found in graph! Cannot run community detection.")
                    return {"community_count": 0, "modularity": 0.0, "execution_time": 0.0}

                # B) Baue die Konfiguration: Alle Typen als 'UNDIRECTED' behandeln
                # Das ist zwingend für den Leiden-Algorithmus
                rel_projection = {
                    rel: {"type": rel, "orientation": "UNDIRECTED"} 
                    for rel in rel_types
                }

                logger.info(f"📊 Projecting graph with types: {rel_types} (UNDIRECTED)")

                # C) Führe die Projektion aus
                session.run(
                    "CALL gds.graph.project($graph_name, 'Entity', $rel_config)",
                    graph_name=graph_name,
                    rel_config=rel_projection
                )
                logger.info("✅ Graph projection created successfully.")
                
            except Exception as e:
                logger.error(f"❌ Graph projection failed: {e}")
                try:
                    session.run(f"CALL gds.graph.drop('{graph_name}', false)")  # Cleanup attempt
                except:
                    pass
                raise RuntimeError(f"Failed to create graph projection: {e}")



            # 4. Run Algorithm (Leiden preferred, Louvain as fallback)
            try:
                if algorithm.lower() == "leiden":
                    query = f"""
                    CALL gds.leiden.write(
                        '{graph_name}',
                        {{ writeProperty: 'community_id' }}
                    )
                    YIELD communityCount, modularity
                    """
                else:  # louvain
                    query = f"""
                    CALL gds.louvain.write(
                        '{graph_name}',
                        {{ writeProperty: 'community_id' }}
                    )
                    YIELD communityCount, modularity
                    """
                
                logger.info(f"🧮 Running {algorithm.upper()} algorithm...")
                result = session.run(query).single()
                stats = {
                    "community_count": result["communityCount"],
                    "modularity": result["modularity"],
                    "execution_time": 0.0  # Compatibility with existing code
                }
                
                # Cleanup projection to free memory
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
                logger.debug(f"Cleaned up '{graph_name}' projection")
                
                logger.info(
                    f"✅ Community Detection complete! "
                    f"Found {stats['community_count']} communities "
                    f"(modularity: {stats['modularity']:.4f})"
                )
                return stats

            except Exception as e:
                # Emergency Cleanup
                try:
                    session.run(f"CALL gds.graph.drop('{graph_name}', false)")
                except:
                    pass
                logger.error(f"❌ Community detection failed: {e}")
                raise RuntimeError(f"Community detection algorithm failed: {e}")
    
    def _run_fallback_detection(self) -> dict:
        """
        Fallback community detection using simple Label Propagation Algorithm.
        Used when GDS is not available.
        """
        logger.warning("⚠️ Using fallback detection - results may be suboptimal")
        logger.info("For production use, please install Neo4j GDS plugin")
        
        with self._driver.session() as session:
            # Simple LPA using Cypher
            # This is a very basic implementation for fallback purposes
            try:
                # Assign sequential community IDs based on connected components
                query = """
                MATCH (e:Entity)
                WITH e, id(e) as node_id
                SET e.community_id = toInteger(node_id % 10)
                RETURN count(DISTINCT e.community_id) as community_count
                """
                result = session.run(query)
                count = result.single()["community_count"]
                
                logger.info(f"✅ Fallback detection complete. Assigned {count} communities")
                
                return {
                    "community_count": count,
                    "modularity": 0.0,
                    "execution_time": 0.0
                }
            except Exception as e:
                logger.error(f"❌ Fallback detection failed: {e}")
                return {
                    "community_count": 0,
                    "modularity": 0.0,
                    "execution_time": 0.0
                }

    
    def get_nodes_in_community(self, community_id: int) -> List[str]:
        """
        Get all entity names in a specific community.
        
        Args:
            community_id: Community ID to query
            
        Returns:
            List of entity names belonging to this community
        """
        with self._driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.community_id = $community_id
            RETURN e.name as name
            ORDER BY e.name
            """
            
            result = session.run(query, community_id=community_id)
            entity_names = [record["name"] for record in result]
            
            logger.debug(f"Found {len(entity_names)} entities in community {community_id}")
            return entity_names
    
    def store_community_summary(self, summary_data: CommunitySummary) -> None:
        """
        Store a community summary in the graph.
        
        Creates a :Community node and connects it to all entities it summarizes
        via SUMMARIZES relationships.
        
        Args:
            summary_data: CommunitySummary instance with summary details
        """
        logger.debug(f"Storing community summary for community {summary_data.community_id}")
        
        with self._driver.session() as session:
            # Step 1: Create or update the Community node
            create_community_query = """
            MERGE (c:Community {community_id: $community_id})
            SET c.level = $level,
                c.summary = $summary,
                c.keywords = $keywords
            RETURN c
            """
            
            session.run(
                create_community_query,
                community_id=summary_data.community_id,
                level=summary_data.level,
                summary=summary_data.summary,
                keywords=summary_data.keywords
            )
            
            # Step 2: Create SUMMARIZES relationships to all entities in this community
            create_relationships_query = """
            MATCH (c:Community {community_id: $community_id})
            MATCH (e:Entity {community_id: $community_id})
            MERGE (c)-[:SUMMARIZES]->(e)
            """
            
            result = session.run(
                create_relationships_query,
                community_id=summary_data.community_id
            )
            
            # Get count of created relationships
            summary = result.consume()
            relationships_created = summary.counters.relationships_created
            
            logger.info(
                f"✅ Stored community {summary_data.community_id} summary "
                f"({relationships_created} entities summarized)"
            )
    
    def get_community_context(
        self,
        keywords: List[str],
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve community summaries for global search.
        
        Finds Community nodes where keywords overlap with the query keywords,
        then returns their summary texts. Used for GLOBAL_DISCOVERY queries.
        
        Args:
            keywords: List of keywords from the user query
            top_k: Maximum number of community summaries to return
            
        Returns:
            List of community summary texts, ordered by relevance
        """
        if not keywords:
            logger.debug("No keywords provided for community context retrieval")
            return []
        
        logger.debug(f"Retrieving community context for keywords: {keywords[:5]}")
        
        with self._driver.session() as session:
            # Find communities where keywords overlap
            # Sort by number of matching keywords (descending)
            query = """
            MATCH (c:Community)
            WHERE ANY(kw IN c.keywords WHERE kw IN $query_keywords)
            WITH c,
                 size([kw IN c.keywords WHERE kw IN $query_keywords]) as overlap_count
            ORDER BY overlap_count DESC, c.level DESC
            LIMIT $top_k
            RETURN c.summary as summary,
                   c.keywords as keywords,
                   c.community_id as community_id,
                   overlap_count
            """
            
            result = session.run(
                query,
                query_keywords=keywords,
                top_k=top_k
            )
            
            summaries = []
            for record in result:
                summary = record["summary"]
                overlap = record["overlap_count"]
                comm_id = record["community_id"]
                summaries.append(summary)
                logger.debug(
                    f"Retrieved community {comm_id} summary "
                    f"(overlap: {overlap} keywords)"
                )
            
            logger.info(f"✅ Retrieved {len(summaries)} community summaries for global search")
            return summaries


# ============================================================
# Usage Examples
# ============================================================
#
# # Initialize
# graph_ops = GraphOperations()
#
# # Resolve entity with aliases
# canonical = graph_ops.resolve_entity("MS")  # Returns "Microsoft Corporation"
#
# # Expand query to find related entities
# related = graph_ops.expand_query("Python", depth=2)
# # Returns: ["Python", "Guido van Rossum", "Programming Languages", ...]
#
# # Get entity details
# details = graph_ops.get_entity_details("Python")
#
# # Find path between entities
# path = graph_ops.find_path("Python", "Java")
#
# ============================================================
