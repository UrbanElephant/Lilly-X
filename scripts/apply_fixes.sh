#!/bin/bash
set -e

echo "🚑 LILLY-X CONFIGURATION REPAIR..."
echo "=========================================="

# 1. Restore the Venv Builder (Critical for Py3.14 fix)
echo "🔧 Restoring scripts/fix_runtime_312.sh..."
mkdir -p scripts
cat << 'EOF' > scripts/fix_runtime_312.sh
#!/bin/bash
set -e
PYTHON_BIN="/usr/bin/python3.12"
echo "🔧 Starting Environment Repair..."
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ CRITICAL: Python 3.12 not found ($PYTHON_BIN)."
    echo "   Run: sudo dnf install python3.12 python3.12-devel"
    exit 1
fi
if [ -d ".venv" ]; then rm -rf .venv; fi
echo "🔨 Creating .venv with Python 3.12..."
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi
echo "✅ Repair Complete."
EOF
chmod +x scripts/fix_runtime_312.sh

# 2. Fix the Container Connection (The "Missing Models" Fix)
echo "🔌 Creating scripts/connect_garden.sh..."
cat << 'EOF' > scripts/connect_garden.sh
#!/bin/bash
set -e
CONTAINER="garden-production"

echo "🛑 Stopping disconnected container..."
podman stop $CONTAINER 2>/dev/null || true
podman rm $CONTAINER 2>/dev/null || true

echo "🔍 Detecting Model Storage..."
# Prefer host directory if populated, otherwise fallback to named volume
if [ -d "$HOME/.ollama/models" ]; then
    echo "   ✅ Found models in $HOME/.ollama"
    VOLUME_MAP="-v $HOME/.ollama:/root/.ollama:Z"
else
    echo "   ⚠️  Models not found in Home. Using 'ollama_data' volume."
    VOLUME_MAP="-v ollama_data:/root/.ollama"
fi

echo "🚀 Starting Garden with TURBO settings..."
podman run -d \
  --name $CONTAINER \
  --restart always \
  --network host \
  --security-opt label=disable \
  --device /dev/kfd --device /dev/dri \
  $VOLUME_MAP \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e HSA_ENABLE_SDMA=0 \
  -e OLLAMA_NUM_PARALLEL=4 \
  -e OLLAMA_MAX_LOADED_MODELS=2 \
  ollama/ollama:latest

echo "✅ Container Connected!"
echo "👉 Verify models with: curl http://localhost:11434/api/tags"
EOF
chmod +x scripts/connect_garden.sh

# 3. Fix Configuration (Enforce garden-thinker)
echo "⚙️  Updating src/config.py..."
mkdir -p src
cat << 'EOF' > src/config.py
import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "tech_books"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "garden-thinker"  # Enforced
    embedding_model: str = "BAAI/bge-m3"
    docs_dir: Path = Path("./data/docs")
    chunk_size: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_strategy: Literal["semantic", "sentence_window", "hierarchical"] = "semantic"
    sentence_window_size: int = 3
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()

def setup_environment():
    from llama_index.core import Settings as LlamaSettings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    # Configure LLM (Garden 70B+)
    LlamaSettings.llm = Ollama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        request_timeout=600.0,
        context_window=8192,
        additional_kwargs={"num_ctx": 8192}
    )
    # Configure Embeddings (CPU)
    LlamaSettings.embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        cache_folder="./models",
        device="cpu"
    )
EOF

# 4. Fix RAG Engine (Imports)
echo "🧠 Patching src/rag_engine.py..."
cat << 'EOF' > src/rag_engine.py
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

# [FIX] Absolute Imports
from src.advanced_rag.pipeline import AdvancedRAGPipeline
from src.advanced_rag.retrieval import SimpleGraphRetriever
from src.config import settings
from src.database import get_qdrant_client
from src.graph_database import get_neo4j_driver

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    response: str
    source_nodes: List[NodeWithScore]
    metadata: dict = None

class RAGEngine:
    def __init__(self, vector_index=None, enable_decomposition=True, enable_hyde=False, enable_rewriting=False, verbose=True):
        self.verbose = verbose
        self._index = vector_index
        self._neo4j_driver = None
        self._pipeline = None
        self._initialize(enable_decomposition, enable_hyde, enable_rewriting)
    
    def _initialize(self, enable_decomposition, enable_hyde, enable_rewriting):
        # 1. LLM & Embeddings (Loaded from Global Settings)
        if not Settings.llm:
            from src.config import setup_environment
            setup_environment()

        # 2. Graph
        try:
            self._neo4j_driver = get_neo4j_driver()
            graph_retriever = SimpleGraphRetriever(self._neo4j_driver, Settings.llm) if self._neo4j_driver else None
        except:
            graph_retriever = None

        # 3. Vector Store
        if self._index is None:
            client = get_qdrant_client()
            vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
            self._index = VectorStoreIndex.from_vector_store(vector_store)

        # 4. Pipeline
        self._pipeline = AdvancedRAGPipeline(
            vector_index=self._index,
            graph_retriever=graph_retriever,
            enable_decomposition=enable_decomposition,
            enable_hyde=enable_hyde,
            enable_rewriting=enable_rewriting,
            reranker_model=settings.reranker_model,
            reranker_device="cpu"
        )

    async def aquery(self, user_query: str, top_k=25, top_n=5) -> RAGResponse:
        try:
            ranked_nodes = await self._pipeline.run(query=user_query, top_k=top_k, top_n=top_n)
            if not ranked_nodes: return RAGResponse("No info found.", [])
            
            context = "\n".join([n.node.get_content() for n in ranked_nodes])
            prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
            response = Settings.llm.complete(prompt)
            return RAGResponse(response.text.strip(), ranked_nodes, {"top_k": top_k})
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            retriever = self._index.as_retriever(similarity_top_k=top_n)
            nodes = retriever.retrieve(user_query)
            context = "\n".join([n.get_content() for n in nodes])
            response = Settings.llm.complete(f"Context:\n{context}\n\nQ: {user_query}\n\nA:")
            return RAGResponse(response.text.strip(), nodes, {"error": str(e)})

    def query(self, *args, **kwargs):
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(self.aquery(*args, **kwargs))
EOF

# 5. Fix Ingestion (Resilience)
echo "📝 Patching src/ingest.py..."
cat << 'EOF' > src/ingest.py
import os
import logging
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings, setup_environment
from src.database import get_qdrant_client
from llama_index.core.node_parser import SemanticSplitterNodeParser

logger = logging.getLogger(__name__)

def ingest_documents(docs_dir: Path = None):
    if not docs_dir: docs_dir = settings.docs_dir
    setup_environment()
    
    client = get_qdrant_client()
    try: client.get_collection(settings.qdrant_collection)
    except: client.create_collection(settings.qdrant_collection, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
    
    logger.info("📂 Loading documents...")
    documents = SimpleDirectoryReader(str(docs_dir), recursive=True).load_data()
    
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    
    # [FIX] Configurable Workers & Crash Fallback
    num_workers = int(os.getenv("INGEST_WORKERS", "4"))
    
    # Parser
    parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model)
    
    pipeline = IngestionPipeline(
        transformations=[parser, Settings.embed_model],
        vector_store=vector_store,
    )

    logger.info(f"🚀 Running Ingestion (Workers: {num_workers})...")
    try:
        pipeline.run(documents=documents, show_progress=True, num_workers=num_workers)
    except OSError as e:
        if e.errno == 24:
            logger.warning("⚠️ File limit hit. Falling back to serial ingestion.")
            pipeline.run(documents=documents, show_progress=True, num_workers=1)
        else: raise e
        
    return VectorStoreIndex.from_vector_store(vector_store)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_documents()
EOF

# 6. Fix Ingestion Launcher
echo "🚀 Patching run_ingestion.sh..."
cat << 'EOF' > run_ingestion.sh
#!/bin/bash
set -e
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ .venv missing. Run scripts/fix_runtime_312.sh"
    exit 1
fi
source .venv/bin/activate
# [FIX] Increase Limits
ulimit -n 4096 2>/dev/null || echo "⚠️ ulimit unchanged"
export INGEST_WORKERS=4
export QDRANT__STORAGE__PERFORMANCE__MMAP_THRESHOLD_KB=0 
python src/ingest.py
EOF
chmod +x run_ingestion.sh

echo "=========================================="
echo "✅ REPAIR COMPLETE."
echo "👉 STEP 1: Run 'bash scripts/fix_runtime_312.sh' (Mandatory!)"
echo "👉 STEP 2: Run 'bash scripts/connect_garden.sh' (Reconnects Models)"
echo "👉 STEP 3: Run './run_ingestion.sh'"
echo "=========================================="
