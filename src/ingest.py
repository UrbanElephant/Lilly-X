import os
import logging
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings, setup_environment
from src.database import get_qdrant_client
from llama_index.core.node_parser import SemanticSplitterNodeParser

logger = logging.getLogger(__name__)

def ingest_documents(docs_dir: Path = None):
    if not docs_dir: docs_dir = settings.docs_dir
    setup_environment()
    
    client = get_qdrant_client()
    try: client.get_collection(settings.qdrant_collection)
    except: client.create_collection(settings.qdrant_collection, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
    
    logger.info("📂 Loading documents...")
    documents = SimpleDirectoryReader(str(docs_dir), recursive=True).load_data()
    
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    
    # [FIX] Configurable Workers & Crash Fallback
    num_workers = int(os.getenv("INGEST_WORKERS", "4"))
    
    # Parser
    parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model)
    
    pipeline = IngestionPipeline(
        transformations=[parser, Settings.embed_model],
        vector_store=vector_store,
    )

    logger.info(f"🚀 Running Ingestion (Workers: {num_workers})...")
    try:
        pipeline.run(documents=documents, show_progress=True, num_workers=num_workers)
    except OSError as e:
        if e.errno == 24:
            logger.warning("⚠️ File limit hit. Falling back to serial ingestion.")
            pipeline.run(documents=documents, show_progress=True, num_workers=1)
        else: raise e
        
    return VectorStoreIndex.from_vector_store(vector_store)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_documents()
