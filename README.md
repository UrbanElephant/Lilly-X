# LLIX: High-Performance RAG System (Max+ 395 Edition)

A stable, hardware-optimized RAG (Retrieval-Augmented Generation) system built for the **AMD Ryzen AI MAX+ 395** workstation running **Fedora 42**.

## 🚀 Hardware & Environment Specs

The following configuration is **mandatory** for stability and performance on this specific hardware stack.

| Component | Specification | Notes |
|-----------|---------------|-------|
| **Device** | AMD Ryzen AI MAX+ 395 | 16 Cores / 32 Threads, 3.0-5.1 GHz |
| **RAM** | 128GB LPDDR5x | Shared Memory Architecture (32GB reserved for iGPU) |
| **OS** | Fedora 42 (Rawhide/Bleeding Edge) | Requires strict Python version management |
| **Python** | **3.12.x** (Strict) | **CRITICAL**: Do NOT use system Python 3.14 due to GCC 15 build failures with Torch/Pandas. |

---

## 🛠️ Optimizations & Architecture

### 1. CPU-Forced Inference (Stability First)
Due to current ROCm instability on the bleeding-edge Fedora 42 kernel with the 8060S iGPU, this release forces **CPU-only inference** for `ollama`.

*   **Logic**: `OLLAMA_LLM_LIBRARY=cpu` + `ROCR_VISIBLE_DEVICES=""`
*   **Performance**: Leverages the **16 physical cores** of the Max+ 395 (`OLLAMA_NUM_THREAD=16`).
*   **Concurrency**: Single parallel request (`OLLAMA_NUM_PARALLEL=1`) guarantees ingestion stability for large PDFs.

### 2. System Limits
*   **File Descriptors**: Increased to `65536` (`ulimit -n`) to prevent `OSError: [Errno 24]` during high-throughput ingestion.

### 3. Modular Dependency Stack
*   Using `llama-index-core` (modular) instead of the metapackage to avoid bloat.
*   **Local-First**: Configured for local **Ollama** (Mistral-Nemo) and **HuggingFace** (BGE-M3) embeddings. No cloud dependencies.

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have **Python 3.12** installed:
```bash
sudo dnf install python3.12
```

### 2. Setup Environment
```bash
# Clone repository
git clone https://github.com/your-username/LLIX.git
cd LLIX

# Create Virtual Environment (MUST be Python 3.12)
/usr/bin/python3.12 -m venv venv

# Install Dependencies
./venv/bin/pip install --upgrade pip setuptools wheel
./venv/bin/pip install -r requirements.txt
```

### 3. Run the System (Golden Script)
Use the included wrapper script to ensure all hardware optimizations are applied. **Do not run python/streamlit directly.**

**To Ingest Documents:**
```bash
./run_llix.sh ingest
```

**To Launch UI:**
```bash
./run_llix.sh ui
```

---

## 📦 Project Structure

*   `src/ingest.py`: Serialized ingestion pipeline (GraphRAG + Vector).
*   `src/app.py`: Streamlit UI with Graph reasoning visualization.
*   `run_llix.sh`: **Check this file** for the critical env exports.
*   `requirements.txt`: Curated, modular dependency list.

---

## ⚠️ Known Issues / Troubleshooting
*   **Pip Hangs**: If pip gets stuck resolving dependencies, ensure you are using the provided `requirements.txt` and a fresh venv.
*   **Ollama 500 Errors**: If you see this, ensure `OLLAMA_NUM_PARALLEL=1` is set (handled by `run_llix.sh`).
