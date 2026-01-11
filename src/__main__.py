"""Entry point for running ingestion as a module."""
from pathlib import Path
from src.ingest import ingest_documents

if __name__ == "__main__":
    ingest_documents(Path("./data/docs"))
