"""Graph operations for entity resolution and query expansion in Lilly-X."""

from typing import List, Optional, Dict, Any

from neo4j import Driver

from src.config import settings
from src.graph_database import get_neo4j_driver


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
