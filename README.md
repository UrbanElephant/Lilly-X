# Lilly-X - Local RAG System

A high-performance Local Retrieval-Augmented Generation (RAG) system optimized for 128GB RAM environments.

## Architecture

- **Vector Database**: Qdrant (containerized via Podman)
- **LLM Engine**: Ollama (native host installation)
- **Embedding Model**: BAAI/bge-large-en-v1.5
- **LLM Model**: llama3:70b
- **Framework**: LlamaIndex

## Project Structure

```
Lilly-X/
├── compose.yaml          # Podman Compose configuration for Qdrant
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── src/
│   ├── __init__.py
│   ├── config.py         # Central settings management
│   └── database.py       # Qdrant client singleton
└── data/
    └── books/            # Documents for ingestion
```

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 2. Start Infrastructure

```bash
# Start Qdrant with Podman
podman compose up -d

# Verify Qdrant is running
curl http://localhost:6333/healthz
```

### 3. Configure Environment

Edit `.env` to customize settings:
- Qdrant connection details
- Ollama model selection
- Document directory path
- Performance parameters

### 4. Verify Ollama

Ensure Ollama is running natively on the host:

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull required models if needed
ollama pull llama3:70b
```

## Configuration

The system uses `pydantic-settings` for configuration management. All settings can be configured via:

1. `.env` file (recommended)
2. Environment variables
3. Default values in `src/config.py`

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST API endpoint |
| `QDRANT_COLLECTION` | `tech_books` | Collection name for embeddings |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LLM_MODEL` | `llama3:70b` | LLM for text generation |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `DOCS_DIR` | `./data/books` | Document directory |

## Qdrant Optimization

The `compose.yaml` is optimized for high-RAM environments:

- **mmap_threshold = 0**: Keeps entire vector index in RAM
- **max_vectors_size = 32GB**: Allows large collections
- **HNSW indexing**: Fast approximate nearest neighbor search
- **8 segments**: Optimized for parallel processing

## Next Steps

1. Implement document ingestion pipeline
2. Create RAG query interface
3. Add monitoring and logging
4. Develop UI/CLI interface

## Requirements

- Python 3.10+
- Podman (rootless mode supported)
- 128GB RAM (minimum 16GB)
- Ollama installed on host
- ~100GB storage for models and vectors

## License

Proprietary - Lilly-X Project
