#!/bin/bash
set -e

echo "🔧 PATCHING CONFIGURATION (RESTORING NEO4J)..."

cat << 'EOF' > src/config.py
import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # 1. Vector Database (Qdrant)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "tech_books"
    
    # 2. Graph Database (Neo4j) - [RESTORED]
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # 3. Model Infrastructure (Garden)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "garden-thinker"
    embedding_model: str = "BAAI/bge-m3"
    
    # 4. Data Processing
    docs_dir: Path = Path("./data/docs")
    chunk_size: int = 1024
    chunk_overlap: int = 200
    
    # 5. Advanced RAG
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    top_k_final: int = 5
    retrieval_strategy: Literal["semantic", "sentence_window", "hierarchical"] = "semantic"
    sentence_window_size: int = 3
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256
    
    # 6. GraphRAG Strategy
    graph_expansion_depth: int = 2
    planner_max_subqueries: int = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()

def setup_environment():
    """Initialize LlamaIndex global settings."""
    import logging
    from llama_index.core import Settings as LlamaSettings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    logger = logging.getLogger(__name__)
    
    # Use environment overrides if present, otherwise defaults
    llm_model = os.getenv("LLM_MODEL", settings.llm_model)
    embed_model_name = os.getenv("EMBED_MODEL", settings.embedding_model)
    
    try:
        # Configure LLM (Garden 70B+)
        llm = Ollama(
            model=llm_model,
            base_url=settings.ollama_base_url,
            request_timeout=600.0,
            context_window=8192,
            temperature=0.1,
            additional_kwargs={"num_ctx": 8192}
        )
        
        # Configure Embeddings (CPU Optimized)
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            cache_folder="./models",
            device="cpu"
        )
        
        LlamaSettings.llm = llm
        LlamaSettings.embed_model = embed_model
        logger.info(f"✅ Environment Configured: {llm_model} / {embed_model_name}")
        
    except Exception as e:
        logger.error(f"❌ Setup Failed: {e}")
        raise

EOF

chmod +x src/config.py
echo "✅ CONFIGURATION PATCHED."
echo "👉 Now verify with: python -c 'from src.graph_database import get_neo4j_driver; print(get_neo4j_driver())'"
