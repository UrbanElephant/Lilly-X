"""Central configuration management using pydantic-settings."""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

# Storage directory for persisted vector index
# Can be overridden via environment variable STORAGE_DIR
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://127.0.0.1:6333",
        description="Qdrant vector database URL",
    )
    qdrant_collection: str = Field(
        default="tech_books",
        description="Qdrant collection name for storing embeddings",
    )

    # Neo4j Configuration
    neo4j_url: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j graph database URL (Bolt protocol)",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j database username",
    )
    neo4j_password: str = Field(
        default="password",
        description="Neo4j database password",
    )

    # Ollama Configuration (running natively on host)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    llm_model: str = Field(
        default="mistral-nemo:12b",
        description="LLM model identifier for text generation",
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace embedding model identifier",
    )

    # Data Configuration
    docs_dir: Path = Field(
        default=Path("./data/docs"),
        description="Directory containing documents to ingest",
    )

    # Optional: Performance Settings
    chunk_size: int = Field(
        default=1024,
        description="Text chunk size for document splitting",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between consecutive chunks",
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for embedding generation",
    )
    top_k: int = Field(
        default=3,
        description="Number of top results to retrieve from vector store",
    )

    # Two-Stage Retrieval Configuration
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace re-ranker model identifier for two-stage retrieval",
    )
    top_k_retrieval: int = Field(
        default=25,
        description="Number of results to retrieve in first stage (broad net)",
    )
    top_k_final: int = Field(
        default=5,
        description="Number of top results after re-ranking to send to LLM",
    )

    # Multi-Turn Conversation & Disambiguation Configuration
    memory_window_size: int = Field(
        default=10,
        description="Number of conversation turns to keep in memory for context",
    )
    graph_expansion_depth: int = Field(
        default=2,
        description="Depth for query expansion in knowledge graph traversal",
    )

    # Query Planning Configuration (Reasoning-GraphRAG)
    planner_max_subqueries: int = Field(
        default=3,
        description="Maximum number of sub-queries to generate from a complex query (prevents explosion)",
    )
    graph_hop_depth: int = Field(
        default=2,
        description="Maximum depth for multi-hop graph traversal in context retrieval",
    )

    # Advanced Retrieval Strategy Configuration
    retrieval_strategy: Literal["semantic", "sentence_window", "hierarchical"] = Field(
        default="semantic",
        description="Retrieval strategy for chunking: semantic, sentence_window, or hierarchical",
    )
    sentence_window_size: int = Field(
        default=3,
        description="Window size for sentence window retrieval (number of sentences before/after)",
    )
    parent_chunk_size: int = Field(
        default=1024,
        description="Parent chunk size for hierarchical chunking (in tokens)",
    )
    child_chunk_size: int = Field(
        default=256,
        description="Child chunk size for hierarchical chunking (in tokens)",
    )

    def __init__(self, **kwargs) -> None:
        """Initialize settings and ensure docs_dir exists."""
        super().__init__(**kwargs)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings: Settings = Settings()


def setup_environment():
    """
    Initialize LlamaIndex global Settings with configured models.
    
    This function should be called once at app startup to configure:
    - LLM (Ollama with configurable model)
    - Embedding model (HuggingFace with configurable model)
    - Other global settings
    
    Optimized for stability on AMD Ryzen AI MAX-395.
    Models can be configured via environment variables:
    - LLM_MODEL (default: mistral-nemo)
    - EMBED_MODEL (default: BAAI/bge-m3)
    
    Returns:
        Tuple of (llm, embed_model) for reference
    """
    import logging
    from llama_index.core import Settings as LlamaSettings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    logger = logging.getLogger(__name__)
    
    # Get model names from environment or use defaults
    llm_model = os.getenv("LLM_MODEL", settings.llm_model)
    embed_model_name = os.getenv("EMBED_MODEL", settings.embedding_model)
    
    logger.info(f"Setting up environment with LLM: {llm_model}, Embeddings: {embed_model_name}")
    
    # Setup LLM with safer settings to prevent OOM/runner crashes
    try:
        llm = Ollama(
            model=llm_model,
            base_url=settings.ollama_base_url,
            request_timeout=600.0,  # Increased timeout for complex reasoning
            context_window=4096,    # Safer context window to prevent Ollama crashes
            temperature=0.1,        # Low temperature for factual RAG responses
            additional_kwargs={"num_ctx": 4096}  # Match context window
        )
        logger.info(f"✅ LLM configured: {llm_model}")
    except Exception as e:
        logger.error(f"❌ Failed to configure LLM '{llm_model}': {e}")
        raise
    
    # Setup Embedding Model
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            cache_folder="./models",
            device="cpu"  # Explicitly use CPU, saving iGPU for future optimization
        )
        logger.info(f"✅ Embeddings configured: {embed_model_name}")
    except Exception as e:
        logger.error(f"❌ Failed to configure embeddings '{embed_model_name}': {e}")
        raise
    
    # Configure global Settings
    LlamaSettings.llm = llm
    LlamaSettings.embed_model = embed_model
    LlamaSettings.chunk_size = settings.chunk_size
    
    logger.info("✅ Environment setup complete")
    
    return llm, embed_model
