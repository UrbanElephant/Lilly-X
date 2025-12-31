"""Document ingestion pipeline for LLIX RAG system."""

print("Starting ingestion script (initializing imports)...", flush=True)

import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings
from src.database import get_qdrant_client, close_qdrant_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_llama_index() -> None:
    """Configure LlamaIndex global settings."""
    logger.info("Setting up LlamaIndex configuration...")
    
    # Configure embedding model
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        cache_folder="./models",
    )
    
    # Configure chunk settings
    text_splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    # Set global settings
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap
    
    logger.info("✓ LlamaIndex configuration complete")


def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    """
    Create Qdrant collection if it doesn't exist.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
    """
    try:
        # Check if collection exists
        client.get_collection(collection_name)
        logger.info(f"✓ Collection '{collection_name}' already exists")
    except Exception:
        # Collection doesn't exist, create it
        logger.info(f"Creating collection '{collection_name}'...")
        
        # Get embedding dimension from the model
        # BAAI/bge-large-en-v1.5 produces 1024-dimensional vectors
        vector_size = 1024
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"✓ Collection '{collection_name}' created successfully")


def ingest_documents(
    docs_dir: Optional[Path] = None,
    file_extensions: Optional[List[str]] = None,
) -> VectorStoreIndex:
    """
    Ingest documents from a directory into Qdrant.
    
    Args:
        docs_dir: Directory containing documents to ingest
        file_extensions: List of file extensions to process (e.g., ['.txt', '.pdf'])
    
    Returns:
        VectorStoreIndex: The created index
    """
    # Use default docs directory if not provided
    if docs_dir is None:
        docs_dir = settings.docs_dir
    
    if file_extensions is None:
        file_extensions = [".txt", ".pdf", ".md"]
    
    logger.info("=" * 70)
    logger.info("Lilly-X Document Ingestion Pipeline")
    logger.info("=" * 70)
    logger.info(f"Documents directory: {docs_dir}")
    logger.info(f"File extensions: {file_extensions}")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.qdrant_collection}")
    logger.info("")
    
    # Setup LlamaIndex
    setup_llama_index()
    
    # Get Qdrant client
    logger.info("Connecting to Qdrant...")
    client = get_qdrant_client()
    
    # Create collection if needed
    create_collection_if_not_exists(client, settings.qdrant_collection)
    
    # Load documents
    logger.info(f"Loading documents from {docs_dir}...")
    documents = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=file_extensions,
        recursive=True,
    ).load_data()
    
    logger.info(f"✓ Loaded {len(documents)} document(s)")
    
    if len(documents) == 0:
        logger.warning(f"No documents found in {docs_dir}")
        logger.info("Please add documents to the directory and try again.")
        return None
    
    # Display document info
    for i, doc in enumerate(documents, 1):
        logger.info(f"  {i}. {Path(doc.metadata.get('file_name', 'unknown')).name}")
    logger.info("")
    
    # Create vector store
    logger.info("Creating Qdrant vector store...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index and ingest documents
    logger.info("Processing documents and generating embeddings...")
    logger.info("(This may take a few minutes depending on document size)")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Collection: {settings.qdrant_collection}")
    logger.info(f"Vector store: {settings.qdrant_url}")
    logger.info("")
    
    # Get collection info
    collection_info = client.get_collection(settings.qdrant_collection)
    logger.info(f"Total vectors in collection: {collection_info.points_count}")
    logger.info("")
    
    return index


if __name__ == "__main__":
    """Run document ingestion."""
    try:
        # Check if docs directory exists
        docs_path = Path("./data/docs")
        if not docs_path.exists():
            logger.error(f"Documents directory not found: {docs_path}")
            logger.info("Please create the directory and add documents.")
            exit(1)
        
        # Run ingestion
        index = ingest_documents(docs_dir=docs_path)
        
        if index is not None:
            logger.info("Ingestion successful! You can now query the RAG system.")
        
    except KeyboardInterrupt:
        logger.info("\nIngestion cancelled by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        exit(1)
    finally:
        close_qdrant_client()
