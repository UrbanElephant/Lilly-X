# Lilly-X - Local RAG System

A high-performance Local Retrieval-Augmented Generation (RAG) system optimized for 128GB RAM environments with AMD iGPU acceleration.

## Architecture

- **Vector Database**: Qdrant (containerized via Podman)
- **LLM Engine**: Ollama (native host installation)
- **Embedding Model**: BAAI/bge-m3 (1024 dimensions)
- **LLM Model**: mistral-nemo:12b
- **Framework**: LlamaIndex
- **Hardware Acceleration**: AMD Radeon 8060S iGPU (32GB VRAM) with ROCm

### Hardware Optimization

Optimized for **AMD Ryzen AI MAX-395** workstations:
- **CPU**: Ryzen AI MAX-395
- **RAM**: 128GB DDR5
- **iGPU**: AMD Radeon 8060S with 32GB dedicated VRAM
- **ROCm**: Configured with `HSA_OVERRIDE_GFX_VERSION=11.0.2`
- **Context Window**: 8192 tokens (via `num_ctx=8192`)
- **Chunk Strategy**: 1024-token chunks with 200-token overlap

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
ollama pull mistral-nemo:12b
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
| `LLM_MODEL` | `mistral-nemo:12b` | LLM for text generation |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model (1024-dim) |
| `CHUNK_SIZE` | `1024` | Text chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DOCS_DIR` | `./data/books` | Document directory |

## Qdrant Optimization

The `compose.yaml` is optimized for high-RAM environments:

- **mmap_threshold = 0**: Keeps entire vector index in RAM
- **max_vectors_size = 32GB**: Allows large collections
- **HNSW indexing**: Fast approximate nearest neighbor search
- **8 segments**: Optimized for parallel processing

## Performance Features

### Context Window Optimization
- **8192-token context window** via `num_ctx=8192` in Ollama configuration
- Handles multiple 1024-token chunks without truncation
- Optimized for comprehensive retrieval-augmented generation

### iGPU Acceleration
- ROCm-accelerated inference on AMD Radeon 8060S
- 32GB dedicated VRAM for model and context
- ~2-3x faster inference compared to larger models

### Memory Optimization
- Qdrant configured to keep vectors in RAM (mmap_threshold=0)
- Embedding cache in `./models` directory
- Efficient batch processing with configurable batch sizes

## Quick Start

```bash
# Start everything
cd /home/gerrit/Antigravity/LLIX
bash start_all.sh

# Or start components separately:
# 1. Start Qdrant
podman run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage:z qdrant/qdrant:latest

# 2. Start Streamlit UI
bash start.sh
```

Access the UI at: **http://localhost:8501**

## Requirements

### Software
- Python 3.10+
- Podman (rootless mode supported)
- Ollama installed on host
- Streamlit for UI

### Hardware (Recommended)
- **CPU**: AMD Ryzen AI MAX-395 or similar
- **RAM**: 128GB DDR5 (minimum 16GB)
- **GPU**: AMD Radeon 8060S iGPU with 32GB VRAM (or equivalent)
- **Storage**: ~100GB for models, vectors, and documents

### ROCm Configuration (for AMD iGPU)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.2
```

## License

Proprietary - Lilly-X Project
