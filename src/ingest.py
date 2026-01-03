"""Document ingestion pipeline for LLIX RAG system."""

print("Starting ingestion script (initializing imports)...", flush=True)

import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    BaseExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.schema import BaseNode
from llama_index.core.bridge.pydantic import Field
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings
from src.database import get_qdrant_client, close_qdrant_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLMEntityExtractor(BaseExtractor):
    """
    Extracts entities using the configured LLM instead of an external library.
    Robust replacement for the fragile SpanMarker EntityExtractor.
    """
    llm: object = Field(description="The LLM to use for extraction")
    
    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, **kwargs)
        
    async def aextract(self, nodes: list[BaseNode]) -> list[dict]:
        metadata_list = []
        for node in nodes:
            # Simple prompt to get entities
            content = node.get_content(metadata_mode="all")
            prompt = (
                f"Analyze the following text and extract key entities (People, Organizations, Technologies). "
                f"Return ONLY a comma-separated list of entities. If none, return 'None'.\n\n"
                f"Text: {content[:1000]}\nEntities:"
            )
            try:
                # Use the LLM directly
                response = await self.llm.acomplete(prompt)
                entities = str(response).strip()
                metadata_list.append({"entities": entities})
            except Exception as e:
                # Fallback on error
                metadata_list.append({"entities": "Error extracting entities"})
        return metadata_list


def setup_llama_index():
    """Configure LlamaIndex global settings with LLM and embedding model."""
    logger.info("Setting up LlamaIndex configuration...")
    
    # Configure LLM (Mistral-Nemo for metadata extraction)
    logger.info(f"Loading LLM: {settings.llm_model}")
    llm = Ollama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        request_timeout=1200.0,  # Correct parameter name for Ollama client timeout
        context_window=8192,
    )
    
    # Configure embedding model
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        cache_folder="./models",
    )
    
    # Configure chunk settings
    text_splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap
    
    logger.info("✓ LlamaIndex configuration complete")
    
    return llm, embed_model



def get_advanced_pipeline(llm, embed_model, vector_store=None):
    """
    Creates an advanced IngestionPipeline with Semantic Splitting and Metadata Enrichment.
    
    Args:
        llm: Language model for metadata extraction
        embed_model: Embedding model for semantic splitting and embeddings
        vector_store: Optional vector store for incremental persistence (streaming upsert)
    """
    return IngestionPipeline(
        transformations=[
            # 1. Semantic Splitting (Dynamic boundaries based on embedding distance)
            SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95, 
                embed_model=embed_model
            ),
            
            # 2. Metadata Enrichment (The 'Golden Source' Logic)
            # Extracts a summary of the prev/current/next chunk context
            SummaryExtractor(llm=llm, summaries=["prev", "self", "next"]),
            # Generates 3 questions that this chunk can answer (improves retrieval match)
            QuestionsAnsweredExtractor(llm=llm, questions=3),
            # Extracts entities (People, Orgs, Tech) using LLM
            LLMEntityExtractor(llm=llm),
            
            # 3. Embedding Generation
            embed_model,
        ],
        vector_store=vector_store,  # Enable streaming upsert if provided
    )

def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    """
    Create Qdrant collection if it doesn't exist.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
    """
    try:
        # Check if collection exists
        client.get_collection(collection_name)
        logger.info(f"✓ Collection '{collection_name}' already exists")
    except Exception:
        # Collection doesn't exist, create it
        logger.info(f"Creating collection '{collection_name}'...")
        
        # Get embedding dimension from the model
        # BAAI/bge-large-en-v1.5 produces 1024-dimensional vectors
        vector_size = 1024
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"✓ Collection '{collection_name}' created successfully")


def ingest_documents(
    docs_dir: Optional[Path] = None,
    file_extensions: Optional[List[str]] = None,
) -> VectorStoreIndex:
    """
    Ingest documents from a directory into Qdrant.
    
    Args:
        docs_dir: Directory containing documents to ingest
        file_extensions: List of file extensions to process (e.g., ['.txt', '.pdf'])
    
    Returns:
        VectorStoreIndex: The created index
    """
    # Use default docs directory if not provided
    if docs_dir is None:
        docs_dir = settings.docs_dir
    
    if file_extensions is None:
        file_extensions = [".txt", ".pdf", ".md"]
    
    logger.info("=" * 70)
    logger.info("Lilly-X Document Ingestion Pipeline")
    logger.info("=" * 70)
    logger.info(f"Documents directory: {docs_dir}")
    logger.info(f"File extensions: {file_extensions}")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.qdrant_collection}")
    logger.info("")
    
    # Setup LlamaIndex
    llm, embed_model = setup_llama_index()
    
    # Get Qdrant client
    logger.info("Connecting to Qdrant...")
    client = get_qdrant_client()
    
    # Create collection if needed
    create_collection_if_not_exists(client, settings.qdrant_collection)
    
    # Load documents
    logger.info(f"Loading documents from {docs_dir}...")
    documents = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=file_extensions,
        recursive=True,
    ).load_data()
    
    logger.info(f"✓ Loaded {len(documents)} document(s)")
    
    if len(documents) == 0:
        logger.warning(f"No documents found in {docs_dir}")
        logger.info("Please add documents to the directory and try again.")
        return None
    
    # Display document info
    for i, doc in enumerate(documents, 1):
        logger.info(f"  {i}. {Path(doc.metadata.get('file_name', 'unknown')).name}")
    logger.info("")
    
    # Create vector store
    logger.info("Creating Qdrant vector store...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # --- ADVANCED PIPELINE LOGIC WITH STREAMING UPSERT ---
    # 1. Initialize the Advanced Pipeline with vector store for incremental persistence
    logger.info("Initializing Advanced Ingestion Pipeline with streaming upsert...")
    pipeline = get_advanced_pipeline(llm, embed_model, vector_store=vector_store)

    # 2. Run the Pipeline (Transformation & Embedding + Automatic Persistence)
    logger.info("🚀 Running Pipeline: Semantic Splitting & Metadata Enrichment...")
    logger.info("(This will take longer than before due to LLM processing)")
    logger.info("(Data is being saved incrementally - safe to interrupt)")
    
    # Passing documents through the pipeline generates and PERSISTS 'nodes'
    nodes = pipeline.run(documents=documents, show_progress=True, num_workers=1)
    logger.info(f"✓ Pipeline finished. Generated {len(nodes)} enriched nodes.")

    # 3. Create Index from Vector Store (data already persisted)
    # Since the pipeline saved nodes incrementally, we just create an index reference
    logger.info("Creating index from persisted vector store...")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Collection: {settings.qdrant_collection}")
    logger.info(f"Vector store: {settings.qdrant_url}")
    logger.info("")
    
    # Get collection info
    collection_info = client.get_collection(settings.qdrant_collection)
    logger.info(f"Total vectors in collection: {collection_info.points_count}")
    logger.info("")
    
    return index


if __name__ == "__main__":
    """Run document ingestion."""
    try:
        # Check if docs directory exists
        docs_path = Path("./data/docs")
        if not docs_path.exists():
            logger.error(f"Documents directory not found: {docs_path}")
            logger.info("Please create the directory and add documents.")
            exit(1)
        
        # Run ingestion
        index = ingest_documents(docs_dir=docs_path)
        
        if index is not None:
            logger.info("Ingestion successful! You can now query the RAG system.")
        
    except KeyboardInterrupt:
        logger.info("\nIngestion cancelled by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        exit(1)
    finally:
        close_qdrant_client()
