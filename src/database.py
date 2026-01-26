import logging
from qdrant_client import QdrantClient, AsyncQdrantClient
from src.config import settings

logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    """Get synchronous Qdrant client."""
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            # Add API key here if needed in future
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Qdrant client: {e}")
        raise

def get_async_qdrant_client() -> AsyncQdrantClient:
    """Get asynchronous Qdrant client for Turbo RAG."""
    try:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            # Add API key here if needed in future
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create Async Qdrant client: {e}")
        raise

def close_qdrant_client(client):
    """Close the client connection."""
    if client:
        client.close()
