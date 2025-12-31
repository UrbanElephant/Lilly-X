"""
RAG Query Script (CLI Wrapper)
Uses src.rag_engine to retrieve context and answer queries.
"""

import sys
import argparse
import logging
from src.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def query_rag(query_text: str) -> None:
    """
    Query the RAG system using the RAGEngine.
    """
    print(f"\nQuery: {query_text}")
    print("-" * 50)
    
    try:
        # Initialize Engine
        engine = RAGEngine()
        
        # 1. Retrieve & Display Context (for debugging)
        print("\nRetrieving context...")
        nodes = engine.retrieve(query_text)
        
        if not nodes:
            print("No relevant context found.")
        else:
            for i, node in enumerate(nodes, 1):
                content_preview = node.node.get_content().replace('\n', ' ')[:200]
                print(f"[{i}] {content_preview}...")
                print(f"    (Score: {node.score:.4f}, Source: {node.node.metadata.get('file_name', 'Unknown')})")

        # 2. Generate Answer
        print("\nGenerating answer (via Ollama)...")
        response_obj = engine.query(query_text)
        
        print("\n" + "=" * 50)
        print("Answer:")
        print("=" * 50)
        print(response_obj.response)
        print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Lilly-X RAG System")
    parser.add_argument("query", nargs="?", help="The question to ask")
    args = parser.parse_args()

    if args.query:
        query_text = args.query
    else:
        try:
            query_text = input("Enter your query: ")
        except EOFError:
            query_text = ""
    
    if query_text:
        query_rag(query_text)
    else:
        print("No query provided.")
