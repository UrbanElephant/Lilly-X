"""Document ingestion pipeline for LLIX RAG system."""
import json
import logging
import hashlib
import os
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field as PydanticField, field_validator
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    BaseExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.schema import BaseNode
from llama_index.core.bridge.pydantic import Field as LlamaField
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings
from src.database import get_qdrant_client, close_qdrant_client

# --- GRAPH IMPORTS ---
from src.graph_schema import KnowledgeGraphUpdate
from src.graph_database import get_neo4j_driver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# Pydantic Models & State
# =====================================================

class DocumentMetadata(BaseModel):
    document_type: str = PydanticField(default="Unknown")
    authors: Union[str, List[str]] = PydanticField(default="Unknown")
    key_dates: Union[str, List[str]] = PydanticField(default="Unknown")

    @field_validator('authors', 'key_dates', mode='before')
    @classmethod
    def flatten_list(cls, v: Union[str, List[str]]) -> str:
        if isinstance(v, list):
            return ", ".join([str(item) for item in v])
        return v

class IngestionState:
    def __init__(self, state_file: Path = Path("./ingestion_state.json")):
        self.state_file = state_file
        self.state: dict[str, str] = {}
        
    def load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}
    
    def save_state(self) -> None:
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    @staticmethod
    def compute_hash(file_path: Path) -> str:
        md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception:
            return ""
    
    def has_changed(self, file_path: Path) -> bool:
        return self.state.get(str(file_path)) != self.compute_hash(file_path)
    
    def update_file(self, file_path: Path) -> None:
        self.state[str(file_path)] = self.compute_hash(file_path)

# =====================================================
# Extractors
# =====================================================

class StructuredMetadataExtractor(BaseExtractor):
    llm: object = LlamaField(description="LLM")
    
    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, **kwargs)
        
    async def aextract(self, nodes: list[BaseNode]) -> list[dict]:
        metadata_list = []
        for node in nodes:
            content = node.get_content(metadata_mode="all")[:2000]
            prompt = f"Extract metadata (document_type, authors, key_dates) as JSON from:\n{content}"
            try:
                response = await self.llm.acomplete(prompt)
                metadata_list.append(DocumentMetadata().model_dump())
            except Exception:
                metadata_list.append(DocumentMetadata().model_dump())
        return metadata_list

class GraphExtractor(BaseExtractor):
    """
    Extracts Knowledge Graph entities/relationships and writes them DIRECTLY to Neo4j.
    """
    llm: object = LlamaField(description="The LLM to use for extraction")

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, **kwargs)

    async def aextract(self, nodes: list[BaseNode]) -> list[dict]:
        print(f"DEBUG: GraphExtractor called for {len(nodes)} nodes", flush=True)
        
        graph_prompt = PromptTemplate(
            "Extract knowledge graph nodes (Entities) and relationships from the text below.\n"
            "Use strict JSON format matching the schema.\n"
            "Text:\n{text}"
        )

        metadata_list = []
        for node in nodes:
            try:
                content = node.get_content(metadata_mode="all")[:3000]
                
                kg_data = await self.llm.astructured_predict(
                    KnowledgeGraphUpdate,
                    prompt=graph_prompt,
                    text=content
                )
                
                # Write to DB
                if kg_data and (kg_data.entities or kg_data.relationships):
                    self._write_to_neo4j(kg_data)
                
                metadata_list.append({"graph_extracted": True})
            except Exception as e:
                print(f"ERROR in GraphExtractor: {e}", flush=True)
                metadata_list.append({"graph_error": str(e)})
        return metadata_list

    def _write_to_neo4j(self, data: KnowledgeGraphUpdate):
        driver = get_neo4j_driver()
        if not driver:
            print("ERROR: No Neo4j driver available!", flush=True)
            return

        print(f"DEBUG: Writing {len(data.entities)} entities and {len(data.relationships)} rels to Neo4j", flush=True)
        
        try:
            with driver.session() as session:
                # 1. Write Entities
                for entity in data.entities:
                    # Using 'entity_type' instead of 'label' as per schema
                    # Sanitizing label to avoid Cypher injection (simple alphanumeric check recommended in prod)
                    label = entity.entity_type if entity.entity_type.isalnum() else "Entity"
                    
                    cypher = f"MERGE (n:{label} {{name: $name}})"
                    session.run(cypher, name=entity.name)
                
                # 2. Write Relationships
                for rel in data.relationships:
                    # Schema has flattened fields: source_type, source_entity, etc.
                    s_label = rel.source_type if rel.source_type.isalnum() else "Entity"
                    t_label = rel.target_type if rel.target_type.isalnum() else "Entity"
                    r_type = rel.relationship_type.upper() # Ensure uppercase for relationship types
                    
                    cypher = f"""
                        MATCH (source:{s_label} {{name: $source_name}})
                        MATCH (target:{t_label} {{name: $target_name}})
                        MERGE (source)-[r:{r_type}]->(target)
                    """
                    session.run(cypher, 
                                source_name=rel.source_entity, 
                                target_name=rel.target_entity)
                                
        except Exception as e:
            print(f"ERROR writing to Neo4j DB: {e}", flush=True)

# =====================================================
# Pipeline Setup
# =====================================================

def setup_llama_index():
    llm = Ollama(model=settings.llm_model, base_url=settings.ollama_base_url, request_timeout=1200.0, context_window=8192)
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model, cache_folder="./models")
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model

def get_advanced_pipeline(llm, embed_model, vector_store=None):
    return IngestionPipeline(
        transformations=[
            SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model),
            StructuredMetadataExtractor(llm=llm),
            GraphExtractor(llm=llm),
            embed_model,
        ],
        vector_store=vector_store,
    )

def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

def ingest_documents(docs_dir: Optional[Path] = None) -> VectorStoreIndex:
    if docs_dir is None: docs_dir = settings.docs_dir
    llm, embed_model = setup_llama_index()
    client = get_qdrant_client()
    create_collection_if_not_exists(client, settings.qdrant_collection)
    
    ingestion_state = IngestionState()
    ingestion_state.load_state()
    
    all_files = list(docs_dir.glob("**/*.pdf")) + list(docs_dir.glob("**/*.txt")) + list(docs_dir.glob("**/*.md"))
    files_to_process = [f for f in all_files if ingestion_state.has_changed(f)]
    
    if not files_to_process:
        logger.info("No new files.")
        return VectorStoreIndex.from_vector_store(vector_store=QdrantVectorStore(client=client, collection_name=settings.qdrant_collection))
    
    logger.info(f"Processing {len(files_to_process)} files...")
    documents = SimpleDirectoryReader(input_files=[str(f) for f in files_to_process]).load_data()
    
    vector_store = QdrantVectorStore(client=client, collection_name=settings.qdrant_collection)
    pipeline = get_advanced_pipeline(llm, embed_model, vector_store=vector_store)
    
    logger.info("🚀 Running GraphRAG Pipeline...")
    pipeline.run(documents=documents, show_progress=True)
    
    for f in files_to_process:
        ingestion_state.update_file(f)
    ingestion_state.save_state()
    
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

if __name__ == "__main__":
    ingest_documents(Path("./data/docs"))
