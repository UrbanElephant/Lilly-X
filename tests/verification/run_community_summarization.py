#!/usr/bin/env python3
"""
Script to run the Community Summarization Pipeline.

This script executes the full Global GraphRAG indexing process:
1. Community Detection
2. LLM-based Summary Generation
3. Storage of Community Metadata
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.community_pipeline import CommunitySummarizationPipeline
from src.graph_ops import GraphOperations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# --- FIX START: FORCE OLLAMA SETTINGS ---
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config import settings as app_settings

print("🔧 Configuring Global Settings for Ollama...")
Settings.llm = Ollama(
    model=app_settings.llm_model,
    base_url=app_settings.ollama_base_url,
    request_timeout=360.0,
    context_window=8192
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=app_settings.embedding_model,
    cache_folder="./models"
)
print(f"✅ Settings configured: LLM={app_settings.llm_model}, Embed={app_settings.embedding_model}")
# --- FIX END ---


def main():
    """Execute the community summarization pipeline."""
    
    print("\n" + "="*70)
    print("  🌐 Global GraphRAG - Community Summarization Pipeline")
    print("="*70)
    
    try:
        # Initialize components
        logger.info("Initializing pipeline components...")
        graph_ops = GraphOperations()
        pipeline = CommunitySummarizationPipeline(graph_ops=graph_ops)
        
        # Check if graph has entities
        from src.graph_database import get_neo4j_driver
        driver = get_neo4j_driver()
        
        with driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()["count"]
            
            if entity_count == 0:
                logger.error("❌ No Entity nodes found in graph!")
                logger.error("   Please run data ingestion first:")
                logger.error("   python3 -m src.ingest")
                sys.exit(1)
            
            logger.info(f"✅ Graph contains {entity_count} Entity nodes")
        
        # Run the pipeline
        results = pipeline.run_pipeline(
            algorithm="leiden",  # Use Leiden algorithm (preferred)
            level=0,             # Base level communities
            max_communities=None # Process all communities (remove limit for production)
        )
        
        # Display results
        print("\n" + "="*70)
        print("  📊 PIPELINE RESULTS")
        print("="*70)
        print(f"  Community Detection:")
        print(f"    - Algorithm: Leiden")
        print(f"    - Total communities: {results['detection_stats']['community_count']}")
        print(f"    - Modularity score: {results['detection_stats']['modularity']:.4f}")
        print(f"    - Execution time: {results['detection_stats']['execution_time']:.2f}s")
        print(f"\n  Summary Generation:")
        print(f"    - Communities processed: {results['communities_processed']}")
        print(f"    - Communities failed: {results['communities_failed']}")
        print(f"    - Summaries stored: {results['summaries_stored']}")
        print("="*70)
        
        # Success rate
        if results['communities_processed'] > 0:
            success_rate = (results['summaries_stored'] / results['communities_processed']) * 100
            print(f"\n  ✅ Success Rate: {success_rate:.1f}%")
        
        print("\n  To verify results in Neo4j Browser:")
        print("    MATCH (c:Community)-[:SUMMARIZES]->(e:Entity) RETURN c, e LIMIT 25")
        print()
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
