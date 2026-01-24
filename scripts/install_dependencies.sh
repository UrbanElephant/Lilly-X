#!/usr/bin/env bash
#
# Dependency Installation Script for Lilly-X Advanced RAG
# Platform: Fedora 43 | AMD Ryzen AI MAX-395
# Python: 3.12
#

set -e  # Exit on error

echo "================================================================================================"
echo "  Lilly-X Advanced RAG - Dependency Installation"
echo "  Platform: Fedora 43 (Ryzen AI MAX-395)"
echo "================================================================================================"

# Check Python version
PYTHON_VERSION=$(/usr/bin/python3.12 --version 2>&1 | awk '{print $2}')
echo ""
echo "🐍 Python Version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.12 ]]; then
    echo "❌ Error: Python 3.12 is required"
    exit 1
fi

# ============================================================================
# Core Dependencies
# ============================================================================

echo ""
echo "📦 Installing Core LlamaIndex Dependencies..."
/usr/bin/python3.12 -m pip install --upgrade pip

/usr/bin/python3.12 -m pip install \
    llama-index-core \
    llama-index-readers-file \
    llama-index-llms-ollama \
    llama-index-embeddings-huggingface \
    llama-index-vector-stores-qdrant \
    llama-index-graph-stores-neo4j

# ============================================================================
# Advanced RAG Dependencies
# ============================================================================

echo ""
echo "🚀 Installing Advanced RAG Components..."

# BM25 Retriever (keyword search)
/usr/bin/python3.12 -m pip install \
    llama-index-retrievers-bm25

# Reranker (cross-encoder)
/usr/bin/python3.12 -m pip install \
    llama-index-postprocessor-flag-embedding-reranker

# Sentence Transformers (for embeddings and reranking)
/usr/bin/python3.12 -m pip install \
    sentence-transformers

# ============================================================================
# Database Clients
# ============================================================================

echo ""
echo "💾 Installing Database Clients..."

/usr/bin/python3.12 -m pip install \
    qdrant-client \
    neo4j

# ============================================================================
# Utility Libraries
# ============================================================================

echo ""
echo "🔧 Installing Utility Libraries..."

/usr/bin/python3.12 -m pip install \
    json-repair \
    pydantic \
    pydantic-settings \
    python-dotenv \
    streamlit

# ============================================================================
# Hardware-Specific (AMD Ryzen)
# ============================================================================

echo ""
echo "⚡ Installing Hardware Monitoring..."

/usr/bin/python3.12 -m pip install \
    psutil

# ============================================================================
# Development Tools (Optional)
# ============================================================================

echo ""
echo "🛠️  Installing Development Tools (optional)..."

/usr/bin/python3.12 -m pip install \
    ragas \
    datasets \
    watchdog

# ============================================================================
# Verification
# ============================================================================

echo ""
echo "================================================================================================"
echo "  ✅ Dependency Installation Complete"
echo "================================================================================================"

echo ""
echo "🔍 Verifying installations..."

# Test imports
/usr/bin/python3.12 << 'EOF'
import sys

packages = [
    ("llama_index.core", "LlamaIndex Core"),
    ("llama_index.retrievers.bm25", "BM25 Retriever"),
    ("llama_index.postprocessor.flag_embedding_reranker", "Flag Reranker"),
    ("sentence_transformers", "Sentence Transformers"),
    ("qdrant_client", "Qdrant Client"),
    ("neo4j", "Neo4j Driver"),
    ("json_repair", "JSON Repair"),
    ("streamlit", "Streamlit"),
]

print("\n📋 Package Verification:")
print("-" * 60)

all_ok = True
for module_name, display_name in packages:
    try:
        __import__(module_name)
        print(f"✅ {display_name:<30} Installed")
    except ImportError:
        print(f"❌ {display_name:<30} MISSING")
        all_ok = False

print("-" * 60)

if all_ok:
    print("\n🎉 All packages installed successfully!")
    sys.exit(0)
else:
    print("\n⚠️  Some packages are missing. Review errors above.")
    sys.exit(1)
EOF

VERIFY_STATUS=$?

echo ""
if [ $VERIFY_STATUS -eq 0 ]; then
    echo "================================================================================================"
    echo "  🚀 Ready to run Lilly-X Advanced RAG!"
    echo "================================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Ensure services are running:"
    echo "     - Qdrant:  docker ps | grep qdrant"
    echo "     - Ollama:  systemctl status ollama"
    echo "     - Neo4j:   docker ps | grep neo4j"
    echo ""
    echo "  2. Run ingestion (if needed):"
    echo "     ./run_ingestion.sh"
    echo ""
    echo "  3. Start the app:"
    echo "     streamlit run src/app.py"
    echo ""
else
    echo "================================================================================================"
    echo "  ⚠️  Installation completed with warnings"
    echo "================================================================================================"
    echo ""
    echo "Some optional packages may be missing. Core functionality should work."
    echo ""
fi
