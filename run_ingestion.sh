#!/bin/bash
# LLIX - Run Document Ingestion

set -e

echo "==================================="
echo "Lilly-X Document Ingestion"
echo "==================================="
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate

# Check if Qdrant is running
echo "Checking Qdrant connection..."
if ! curl -s http://127.0.0.1:6333/healthz > /dev/null 2>&1; then
    echo "❌ Qdrant is not running!"
    echo "Please start Qdrant: podman start qdrant"
    exit 1
fi
echo "✓ Qdrant is running"
echo ""

# Check for documents
if [ ! -d "data/docs" ] || [ -z "$(ls -A data/docs 2>/dev/null)" ]; then
    echo "⚠ No documents found in data/docs/"
    echo "Please add documents (.txt, .pdf, .md) to data/docs/ directory"
    exit 1
fi

echo "Documents to ingest:"
ls -lh data/docs/
echo ""

# Run ingestion
echo "Starting ingestion..."
echo "(This may take several minutes on first run to download embedding model)"
echo ""

python -m src.ingest

echo ""
echo "==================================="
echo "Ingestion Complete!"
echo "==================================="
