"""Database client management for Qdrant vector store."""

from typing import Optional

from qdrant_client import QdrantClient

from src.config import settings


# Singleton instance
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Get or create a singleton QdrantClient instance.

    Returns:
        QdrantClient: Configured Qdrant client connected to the vector database.

    Raises:
        ConnectionError: If unable to connect to Qdrant.
    """
    global _qdrant_client

    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                timeout=30.0,
                prefer_grpc=False,  # Use REST API (port 6333)
            )
            # Verify connection
            _qdrant_client.get_collections()
            print(f"✓ Connected to Qdrant at {settings.qdrant_url}")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Qdrant at {settings.qdrant_url}: {e}"
            ) from e

    return _qdrant_client


def close_qdrant_client() -> None:
    """Close the Qdrant client connection and reset the singleton."""
    global _qdrant_client

    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
            print("✓ Qdrant client connection closed")
        except Exception as e:
            print(f"⚠ Error closing Qdrant client: {e}")
        finally:
            _qdrant_client = None


if __name__ == "__main__":
    """Test Qdrant connection."""
    print("=" * 50)
    print("Lilly-X - Qdrant Connection Test")
    print("=" * 50)
    print()
    
    try:
        # Test connection
        print(f"📡 Connecting to Qdrant at {settings.qdrant_url}...")
        client = get_qdrant_client()
        print()
        
        # Get collections
        print("📋 Fetching collections...")
        collections = client.get_collections()
        print(f"✓ Found {len(collections.collections)} collection(s):")
        for collection in collections.collections:
            print(f"   - {collection.name}")
        print()
        
        # Test collection creation (if not exists)
        collection_name = settings.qdrant_collection
        print(f"🔍 Checking collection '{collection_name}'...")
        try:
            client.get_collection(collection_name)
            print(f"✓ Collection '{collection_name}' already exists")
        except Exception:
            print(f"ℹ Collection '{collection_name}' does not exist yet (will be created during ingestion)")
        print()
        
        print("=" * 50)
        print("✅ Connection test SUCCESSFUL!")
        print("=" * 50)
        print()
        print("Configuration:")
        print(f"  - Qdrant URL: {settings.qdrant_url}")
        print(f"  - Collection: {settings.qdrant_collection}")
        print(f"  - Embedding Model: {settings.embedding_model}")
        print(f"  - LLM Model: {settings.llm_model}")
        print()
        
    except Exception as e:
        print()
        print("=" * 50)
        print("❌ Connection test FAILED!")
        print("=" * 50)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure Qdrant container is running: podman ps")
        print("  2. Verify Qdrant is accessible: curl http://127.0.0.1:6333")
        print("  3. Check firewall settings")
        exit(1)
    finally:
        close_qdrant_client()

