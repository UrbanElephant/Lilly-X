# Model Update Summary - LLIX System

## Date: 2026-01-02

## Changes Made

Successfully updated the LLIX RAG system from **ibm/granite4:32b-a9b-h** to **mistral-nemo:12b** for better stability and performance on the high-RAM (128GB) iGPU/ROCm environment.

---

## Files Modified

### 1. **src/config.py**
- ✅ Changed default `llm_model` from `ibm/granite4:32b-a9b-h` to `mistral-nemo:12b`
- Updated line 36

### 2. **.env.template**
- ✅ Updated `LLM_MODEL=mistral-nemo:12b`
- ✅ Confirmed `EMBEDDING_MODEL=BAAI/bge-m3` (consistent with config.py)
- Updated lines 7-8

### 3. **start.sh**
- ✅ Updated display message to show new model name
- Updated line 17

### 4. **START_INSTRUCTIONS.md**
- ✅ Updated all references from old model to new model
- ✅ Updated `ollama list` grep command to search for `mistral-nemo`
- ✅ Updated example output and troubleshooting sections

---

## Configuration Verification

### ✅ RAG Engine Settings (src/rag_engine.py)
Already optimized with proper context window settings:
- **context_window**: 8192
- **additional_kwargs**: {"num_ctx": 8192}
- **request_timeout**: 360.0 seconds

### ✅ Ingestion Settings (src/ingest.py)
Properly configured to use global settings:
- **chunk_size**: 1024 (from `settings.chunk_size`)
- **chunk_overlap**: 200 (from `settings.chunk_overlap`)
- Settings are pulled from config.py automatically

---

## Optimization Details

### Model Specifications
- **Model**: mistral-nemo:12b
- **Context Window**: 8192 tokens
- **Chunk Size**: 1024 tokens
- **Chunk Overlap**: 200 tokens
- **Embedding Model**: BAAI/bge-m3
- **Vector Dimension**: 1024
- **Top-K Retrieval**: 5 results

### Performance Benefits
1. **Smaller model size** - Faster inference on iGPU
2. **Better stability** - Less memory pressure on ROCm
3. **Optimized context** - 8192 tokens sufficient for 1024-token chunks
4. **Improved throughput** - Faster response times for user queries

---

## How to Apply Changes

### Option 1: Using Default Configuration
The changes are already in `src/config.py`, so no action needed if not using `.env` file.

### Option 2: Using .env File (Recommended)
Create or update your `.env` file:

```bash
cd /home/gerrit/Antigravity/LLIX
cp .env.template .env
# Edit .env if needed, or it already has the correct model
```

The `.env` file will override the defaults from config.py.

---

## Next Steps

### 1. Verify Ollama Has the Model
```bash
ollama list | grep mistral-nemo
```

If not found, pull it:
```bash
ollama pull mistral-nemo:12b
```

### 2. Start the System
```bash
cd /home/gerrit/Antigravity/LLIX

# Start Qdrant (if not running)
podman run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant:latest

# Start Streamlit UI
bash start.sh
```

### 3. Re-ingest Documents (if needed)
If you want to optimize embeddings with the new configuration:

```bash
source venv/bin/activate
python -m src.ingest
```

---

## Backward Compatibility

To switch back to the old model (if needed):
1. Edit `.env` file: `LLM_MODEL=ibm/granite4:32b-a9b-h`
2. Restart the application

Or set environment variable:
```bash
export LLM_MODEL=ibm/granite4:32b-a9b-h
bash start.sh
```

---

## Technical Notes

- The system uses **pydantic-settings** for configuration management
- Settings priority: `.env` file > environment variables > defaults in config.py
- The RAG engine is initialized once (singleton pattern) at startup
- Context window of 8192 allows for ~8 chunks (1024 tokens each) in memory
- The iGPU/ROCm benefits from the smaller model due to reduced VRAM requirements

---

## Status: ✅ COMPLETE

All files have been updated successfully. The system is now configured to use **mistral-nemo:12b** as the default LLM model.
