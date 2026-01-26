import sys
import os
import time
import asyncio
import streamlit as st
from pathlib import Path
from requests.exceptions import ConnectionError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings, setup_environment
from src.rag_engine import RAGEngine, RAGResponse

st.set_page_config(page_title="Lilly-X | Garden Edition", page_icon="🌿", layout="wide")

@st.cache_resource
def initialize_system():
    try:
        setup_environment()
        engine = RAGEngine(enable_decomposition=True, verbose=True)
        return engine, "success"
    except Exception as e:
        return None, str(e)

engine, status = initialize_system()

if not engine:
    st.error(f"System Failed: {status}")
    st.stop()

st.title("🌿 Lilly-X @ The Garden")
st.caption(f"Brain: {settings.llm_model} | Graph: Neo4j | Vector: Qdrant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask the Garden..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking (Async GraphRAG)..."):
            # SIMPLIFIED CALL: Trust the engine's safe loop handling
            response_obj = engine.query(prompt)
            
            response_text = response_obj.response
            st.markdown(response_text)
            
            # Metadata
            sources = len(response_obj.source_nodes)
            st.caption(f"📚 Sources: {sources} | ⚡ Strategy: Hybrid")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text
            })
