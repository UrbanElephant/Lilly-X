"""Central configuration management using pydantic-settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        default="",
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
