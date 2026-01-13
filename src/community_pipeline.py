"""Community Summarization Pipeline for Global GraphRAG Indexing.

This module orchestrates the end-to-end process of detecting communities in the
knowledge graph and generating LLM-based summaries for each community.
"""

import logging
import json
from typing import Optional, List, Dict, Any

from tqdm import tqdm

from src.graph_ops import GraphOperations
from src.schemas import CommunitySummary
from src.graph_database import get_neo4j_driver

logger = logging.getLogger(__name__)


# ============================================================
# Prompt Templates
# ============================================================

COMMUNITY_SUMMARY_PROMPT = """You are analyzing a community of related entities from a knowledge graph.

**Entities in this community:**
{entities}

**Task:**
1. Analyze the common themes, topics, and relationships among these entities
2. Provide a concise, high-level summary (2-3 sentences) of what this community represents
3. Extract 3-5 key keywords that characterize this community

**Output Format:**
Return your response as a JSON object with this exact structure:
{{
    "summary": "A concise summary of the community theme",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}}

**Important:** Return ONLY the JSON object, no additional text or markdown.
"""


# ============================================================
# Community Summarization Pipeline
# ============================================================

class CommunitySummarizationPipeline:
    """
    Orchestrates the Global GraphRAG indexing process.
    
    This pipeline:
    1. Runs community detection on the knowledge graph
    2. Extracts communities and their entities
    3. Generates LLM-based summaries for each community
    4. Stores summaries back to the graph
    
    Attributes:
        graph_ops: GraphOperations instance for graph interactions
        llm: Language model for generating summaries
    """
    
    def __init__(
        self,
        graph_ops: Optional[GraphOperations] = None,
        llm: Optional[Any] = None
    ):
        """
        Initialize the community summarization pipeline.
        
        Args:
            graph_ops: GraphOperations instance. If None, creates a new instance.
            llm: Language model instance. If None, uses Settings.llm from LlamaIndex.
        """
        self.graph_ops = graph_ops if graph_ops is not None else GraphOperations()
        
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        else:
            # Import here to avoid circular dependencies
            try:
                from llama_index.core import Settings
                self.llm = Settings.llm
                logger.info("Using LlamaIndex Settings.llm for community summarization")
            except ImportError:
                logger.error("Failed to import LlamaIndex Settings. LLM not available.")
                self.llm = None
    
    def run_pipeline(
        self,
        algorithm: str = "leiden",
        level: int = 0,
        max_communities: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete community summarization pipeline.
        
        Args:
            algorithm: Community detection algorithm ('leiden' or 'louvain')
            level: Hierarchical level for community summaries (default: 0)
            max_communities: Optional limit on number of communities to process
            
        Returns:
            Dictionary with pipeline statistics and results
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Community Summarization Pipeline")
        logger.info("=" * 60)
        
        if self.llm is None:
            raise RuntimeError(
                "LLM is not available. Cannot run community summarization pipeline."
            )
        
        results = {
            "detection_stats": {},
            "communities_processed": 0,
            "communities_failed": 0,
            "summaries_stored": 0
        }
        
        try:
            # Step 1: Run community detection
            logger.info("\n📊 Step 1: Running community detection...")
            detection_stats = self.graph_ops.run_community_detection(algorithm=algorithm)
            results["detection_stats"] = detection_stats
            
            logger.info(
                f"✅ Detected {detection_stats['community_count']} communities "
                f"(modularity: {detection_stats['modularity']:.4f})"
            )
            
            # Step 2: Query for all distinct community IDs
            logger.info("\n🔍 Step 2: Querying for community IDs...")
            community_ids = self._get_all_community_ids()
            
            if not community_ids:
                logger.warning("⚠️ No communities found in graph. Exiting pipeline.")
                return results
            
            logger.info(f"✅ Found {len(community_ids)} distinct communities")
            
            # Apply max_communities limit if specified
            if max_communities is not None and len(community_ids) > max_communities:
                logger.info(f"⚠️ Limiting to first {max_communities} communities")
                community_ids = community_ids[:max_communities]
            
            # Step 3: Process each community
            logger.info(f"\n🤖 Step 3: Generating summaries for {len(community_ids)} communities...")
            
            with tqdm(total=len(community_ids), desc="Processing communities") as pbar:
                for comm_id in community_ids:
                    try:
                        # Generate and store summary
                        success = self._process_community(comm_id, level)
                        
                        if success:
                            results["communities_processed"] += 1
                            results["summaries_stored"] += 1
                        else:
                            results["communities_failed"] += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process community {comm_id}: {e}")
                        results["communities_failed"] += 1
                    
                    finally:
                        pbar.update(1)
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("✅ Community Summarization Pipeline Complete!")
            logger.info("=" * 60)
            logger.info(f"Total communities detected: {detection_stats['community_count']}")
            logger.info(f"Communities processed: {results['communities_processed']}")
            logger.info(f"Communities failed: {results['communities_failed']}")
            logger.info(f"Summaries stored: {results['summaries_stored']}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {e}")
            raise
    
    def _get_all_community_ids(self) -> List[int]:
        """
        Query Neo4j for all distinct community IDs.
        
        Returns:
            List of community IDs sorted by size (descending)
        """
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.community_id IS NOT NULL
            WITH e.community_id as comm_id, count(e) as size
            RETURN comm_id, size
            ORDER BY size DESC
            """
            
            result = session.run(query)
            community_ids = [record["comm_id"] for record in result]
            
            logger.debug(f"Retrieved {len(community_ids)} community IDs from graph")
            return community_ids
    
    def _process_community(self, community_id: int, level: int) -> bool:
        """
        Process a single community: fetch entities, generate summary, store.
        
        Args:
            community_id: ID of the community to process
            level: Hierarchical level for this community
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch entity names in this community
            entity_names = self.graph_ops.get_nodes_in_community(community_id)
            
            if not entity_names:
                logger.warning(f"⚠️ Community {community_id} has no entities. Skipping.")
                return False
            
            logger.debug(
                f"Processing community {community_id} with {len(entity_names)} entities"
            )
            
            # Generate summary using LLM
            summary_data = self._generate_community_summary(
                community_id=community_id,
                entity_names=entity_names,
                level=level
            )
            
            if summary_data is None:
                logger.error(f"Failed to generate summary for community {community_id}")
                return False
            
            # Store summary in graph
            self.graph_ops.store_community_summary(summary_data)
            
            logger.info(
                f"✅ Community {community_id}: {len(entity_names)} entities → "
                f"'{summary_data.summary[:60]}...'"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing community {community_id}: {e}")
            return False
    
    def _generate_community_summary(
        self,
        community_id: int,
        entity_names: List[str],
        level: int
    ) -> Optional[CommunitySummary]:
        """
        Generate a summary for a community using the LLM.
        
        Args:
            community_id: ID of the community
            entity_names: List of entity names in the community
            level: Hierarchical level
            
        Returns:
            CommunitySummary instance or None if generation fails
        """
        try:
            # Format entities for prompt
            entities_text = "\n".join([f"- {name}" for name in entity_names[:50]])
            
            # Add count if there are more entities
            if len(entity_names) > 50:
                entities_text += f"\n... and {len(entity_names) - 50} more entities"
            
            # Build prompt
            prompt = COMMUNITY_SUMMARY_PROMPT.format(entities=entities_text)
            
            # Call LLM
            logger.debug(f"Calling LLM for community {community_id} summary...")
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            summary_dict = self._parse_llm_response(response_text)
            
            if summary_dict is None:
                logger.error(f"Failed to parse LLM response for community {community_id}")
                return None
            
            # Create CommunitySummary object
            community_summary = CommunitySummary(
                community_id=community_id,
                level=level,
                summary=summary_dict.get("summary", "No summary available"),
                keywords=summary_dict.get("keywords", [])
            )
            
            return community_summary
            
        except Exception as e:
            logger.error(f"Error generating summary for community {community_id}: {e}")
            return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM JSON response with robust error handling.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
            elif "```" in response_text:
                response_text = response_text.replace("```", "").strip()
            
            # Try standard JSON parsing
            try:
                data = json.loads(response_text)
                return data
            except json.JSONDecodeError:
                # Try json_repair as fallback
                try:
                    import json_repair
                    data = json_repair.loads(response_text)
                    logger.debug("Used json_repair to parse LLM response")
                    return data
                except Exception:
                    logger.warning("json_repair failed, falling back to default response")
                    return None
                    
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None


# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    """
    Example usage of the Community Summarization Pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pipeline
    pipeline = CommunitySummarizationPipeline()
    
    # Run the pipeline
    results = pipeline.run_pipeline(
        algorithm="leiden",  # or "louvain"
        level=0,
        max_communities=None  # Process all communities
    )
    
    # Print results
    print("\n📊 Pipeline Results:")
    print(f"  - Communities detected: {results['detection_stats']['community_count']}")
    print(f"  - Communities processed: {results['communities_processed']}")
    print(f"  - Communities failed: {results['communities_failed']}")
    print(f"  - Summaries stored: {results['summaries_stored']}")
