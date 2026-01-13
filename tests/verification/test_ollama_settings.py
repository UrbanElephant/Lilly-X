#!/usr/bin/env python3
"""Quick test to verify Ollama settings initialization."""

print("Testing Ollama Settings Configuration...")

try:
    # Test imports
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from src.config import settings as app_settings
    
    print("✓ All imports successful")
    
    # Test configuration
    print(f"\nConfiguring with:")
    print(f"  LLM Model: {app_settings.llm_model}")
    print(f"  Ollama URL: {app_settings.ollama_base_url}")
    print(f"  Embed Model: {app_settings.embedding_model}")
    
    Settings.llm = Ollama(
        model=app_settings.llm_model,
        base_url=app_settings.ollama_base_url,
        request_timeout=360.0,
        context_window=8192
    )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=app_settings.embedding_model,
        cache_folder="./models"
    )
    
    print("\n✅ Settings configuration successful!")
    print(f"  Settings.llm = {type(Settings.llm).__name__}")
    print(f"  Settings.embed_model = {type(Settings.embed_model).__name__}")
    
    # Test that we can actually use the LLM
    print("\nTesting LLM with simple query...")
    response = Settings.llm.complete("Say 'Hello GraphRAG' in one short sentence.")
    print(f"  LLM Response: {response.text[:100]}")
    
    print("\n🎉 All tests passed! Ollama is ready for community summarization.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
