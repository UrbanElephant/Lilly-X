#!/usr/bin/env python
"""Quick test script for Qdrant connection."""

import sys
sys.path.insert(0, '/home/gerrit/Antigravity/Lilly-X')

from src.config import settings
from qdrant_client import QdrantClient

print("=" * 60)
print("Lilly-X - Quick Qdrant Connection Test")
print("=" * 60)
print()

print(f"Qdrant URL from config: {settings.qdrant_url}")
print(f"Collection name: {settings.qdrant_collection}")
print()

try:
    print("Connecting...")
    client = QdrantClient(url=settings.qdrant_url, timeout=10.0)
    print("✓ Connected successfully!")
    print()
    
    print("Fetching collections...")
    collections = client.get_collections()
    print(f"✓ Found {len(collections.collections)} collection(s)")
    for coll in collections.collections:
        print(f"   - {coll.name}")
    print()
    
    print("=" * 60)
    print("✅ CONNECTION TEST PASSED!")
    print("=" * 60)
    print()
    print("Ready for data ingestion!")
    
except Exception as e:
    print()
    print("=" * 60)
    print("❌ CONNECTION TEST FAILED!")
    print("=" * 60)
    print(f"Error: {e}")
    print()
    print("Please check:")
    print("  1. Qdrant container is running: podman ps")
    print("  2. Accessible at: curl http://127.0.0.1:6333")
    sys.exit(1)
