# Lilly-X Ingestion Guide

## Overview

The Lilly-X ingestion pipeline processes documents and stores them as vector embeddings in Qdrant for retrieval-augmented generation (RAG).

## Quick Start

```bash
# 1. Add documents to the data/docs directory
cp your_document.pdf data/docs/

# 2. Run ingestion
./run_ingestion.sh

# Or run directly:
source venv/bin/activate
python -m src.ingest
```

## Supported File Types

- `.txt` - Plain text files
- `.pdf` - PDF documents
- `.md` - Markdown files

## How It Works

1. **Document Loading**: Files are loaded from `data/docs/` using LlamaIndex's SimpleDirectoryReader
2. **Text Chunking**: Documents are split into chunks using SentenceSplitter
   - Chunk size: 1024 tokens (configurable in `.env`)
   - Overlap: 200 tokens (to maintain context across chunks)
3. **Embedding Generation**: Each chunk is converted to a 1024-dimensional vector using BAAI/bge-large-en-v1.5
4. **Vector Storage**: Embeddings are uploaded to Qdrant collection `tech_books`

## Configuration

Edit `.env` to customize:

```bash
# Data directory
DOCS_DIR=./data/docs

# Chunking parameters
CHUNK_SIZE=1024
CHUNK_OVERLAP=200

# Embedding model
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Qdrant settings
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=tech_books
```

## First Run

On first run, the embedding model (~1.3GB) will be downloaded:
- Model: BAAI/bge-large-en-v1.5
- Cache location: `./models/`
- This is a one-time download

## Example Output

```
======================================================================
Lilly-X Document Ingestion Pipeline
======================================================================
Documents directory: data/docs
File extensions: ['.txt', '.pdf', '.md']
Qdrant URL: http://127.0.0.1:6333
Collection: tech_books

Loading embedding model: BAAI/bge-large-en-v1.5
✓ LlamaIndex configuration complete
Connecting to Qdrant...
✓ Connected to Qdrant at http://127.0.0.1:6333
Creating collection 'tech_books'...
✓ Collection 'tech_books' created successfully
Loading documents from data/docs...
✓ Loaded 1 document(s)
  1. hello_rag.txt

Processing documents and generating embeddings...
100%|████████████████████████████████████| 1/1 [00:15<00:00, 15.23s/it]

======================================================================
✅ Ingestion Complete!
======================================================================
Documents processed: 1
Collection: tech_books
Vector store: http://127.0.0.1:6333

Total vectors in collection: 12
```

## Monitoring

Check Qdrant collection:
```bash
curl http://127.0.0.1:6333/collections/tech_books
```

View ingestion logs:
```bash
tail -f ingestion_output.log
```

## Troubleshooting

### Qdrant Not Running
```bash
podman start qdrant
curl http://127.0.0.1:6333/healthz
```

### Out of Memory
Reduce batch size in `.env`:
```bash
BATCH_SIZE=16
```

### Model Download Fails
Check internet connection and disk space. Model cache is in `./models/`

### Re-ingesting Documents
To re-ingest, either:
1. Delete the collection via Qdrant API
2. Use a different collection name in `.env`

## Next Steps

After successful ingestion:
1. Verify vectors: `curl http://127.0.0.1:6333/collections/tech_books`
2. Ready for querying: Build query interface (Module 3)
