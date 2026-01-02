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
- **Python 3.12** (recommended) - Python 3.10/3.11 compatible, **3.14+ not supported**
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

## Troubleshooting & Compatibility

### Python Version Requirements

**✅ Supported:** Python 3.12 (Recommended)  
**🔧 Compatible:** Python 3.10, 3.11  
**❌ Unsupported:** Python 3.14+

#### Python 3.14+ Incompatibility

Python 3.14 and newer versions have **dependency conflicts** with current LlamaIndex and related packages. Symptoms include:
- Import errors with `llama-index-core`
- Missing `asyncio` library errors
- Weak reference errors with NoneType objects
- Streamlit compatibility issues

**Solution:** Use Python 3.12 for stable operation.

#### Switching to Python 3.12

If you're experiencing environment issues:

```bash
# Remove existing environment
rm -rf venv venv_314_broken

# Create fresh venv with Python 3.12
python3.12 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
bash verify_setup.sh
```

### Environment Persistence Issues

#### Problem: Settings Not Applied

If the system doesn't use `mistral-nemo:12b` or correct chunk sizes after restart:

**Cause:** The `.env` file is not being read or is missing.

**Solution:**

```bash
# 1. Create .env from template
cp .env.template .env

# 2. Verify settings
cat .env | grep -E "LLM_MODEL|CHUNK_SIZE"

# 3. Expected output:
# LLM_MODEL=mistral-nemo:12b
# CHUNK_SIZE=1024
```

#### Problem: Old Model Still Loading

If `ibm/granite4:32b-a9b-h` is still being used:

```bash
# Check default in config.py
grep "default=" src/config.py | grep llm_model

# Should show: default="mistral-nemo:12b"
```

If not, the code update didn't apply. Re-pull the latest changes or manually update `src/config.py`.

### Maintenance Commands

#### Reset Environment (Clean Slate)

```bash
# Full environment reset
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Verify System Health

```bash
# Run verification script
bash verify_setup.sh

# Manual verification
source venv/bin/activate
python -c "import llama_index; print(f'LlamaIndex: {llama_index.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

#### Check Configuration

```bash
# Verify model settings
grep -E "mistral-nemo|BAAI/bge-m3|chunk_size.*1024" src/config.py

# Verify environment template
cat .env.template
```

#### Clear Embedding Cache

If you need to regenerate embeddings:

```bash
rm -rf models/
# Models will re-download on next ingestion
```

#### Reset Qdrant Database

```bash
# Stop and remove Qdrant container
podman stop qdrant
podman rm qdrant

# Remove stored vectors
podman volume rm qdrant_storage

# Restart Qdrant
podman run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage:z qdrant/qdrant:latest

# Re-ingest documents
source venv/bin/activate
python -m src.ingest
```

### Common Issues

#### Issue: "Module not found" errors

**Solution:** Reinstall dependencies with Python 3.12
```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Issue: Streamlit won't start

**Check:**
```bash
source venv/bin/activate
which streamlit
streamlit --version
```

**Fix:**
```bash
pip install --upgrade streamlit
```

#### Issue: Qdrant connection errors

**Verify Qdrant is running:**
```bash
curl http://localhost:6333/healthz
```

**Restart if needed:**
```bash
podman restart qdrant
```

#### Issue: ROCm/GPU not detected

**Verify ROCm environment:**
```bash
echo $HSA_OVERRIDE_GFX_VERSION
# Should output: 11.0.2
```

**Set permanently:**
```bash
# Add to ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.2' >> ~/.bashrc
source ~/.bashrc
```

#### Issue: Out of Memory (OOM) errors

The `num_ctx=8192` setting in `src/rag_engine.py` prevents most OOM issues. If you still experience them:

**Check context window:**
```bash
grep "num_ctx" src/rag_engine.py
# Should show: additional_kwargs={"num_ctx": 8192}
```

**Reduce batch size if needed:**
```bash
# Edit .env
BATCH_SIZE=16  # Reduce from 32
```

### Getting Help

If issues persist after following troubleshooting steps:

1. **Verify Python version:** `python --version` (should be 3.12.x)
2. **Check git status:** `git status` (should be clean)
3. **Review logs:** Check `streamlit_launch.log` or console output
4. **Run verification:** `bash verify_setup.sh`

## License

Proprietary - Lilly-X Project
