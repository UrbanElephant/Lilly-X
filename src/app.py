"""
Lilly-X Chat Interface
Professional RAG system powered by Advanced Retrieval Pipeline

Hardware: AMD Ryzen AI MAX-395 | 128GB RAM | Radeon 8060S iGPU
Platform: Fedora 43 with local Ollama (Mistral-Nemo)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import streamlit as st
import time
from typing import Optional

# Import local modules
from src.config import settings, setup_environment
from src.rag_engine import RAGEngine, RAGResponse


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Lilly-X | Advanced RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def initialize_system():
    """
    Initialize the RAG system components.
    
    Returns:
        Tuple of (RAGEngine, initialization_status)
    """
    try:
        # Setup LlamaIndex global Settings
        setup_environment()
        
        # Initialize RAG Engine with Advanced Pipeline
        engine = RAGEngine(
            enable_decomposition=True,
            enable_hyde=False,
            enable_rewriting=False,
            verbose=True,
        )
        
        return engine, "success"
        
    except FileNotFoundError as e:
        return None, f"storage_missing: {e}"
    except Exception as e:
        return None, f"error: {e}"


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize system
engine, init_status = initialize_system()

# Check initialization status
if engine is None:
    st.error("🚨 **System Initialization Failed**")
    
    if "storage_missing" in init_status:
        st.warning("""
        **Vector store not found.**
        
        It looks like the knowledge base hasn't been ingested yet.
        
        **To fix this:**
        1. Ensure Qdrant is running: `docker ps | grep qdrant`
        2. Run ingestion: `./run_ingestion.sh`
        3. Restart this app
        """)
    else:
        st.error(f"**Error:** {init_status}")
        st.info("""
        **System Requirements:**
        - ✅ Qdrant running on http://localhost:6333
        - ✅ Ollama running on http://localhost:11434
        - ✅ Neo4j running on bolt://localhost:7687 (optional)
        - ✅ Knowledge base ingested (check ./data/docs/)
        """)
    
    st.stop()


# ============================================================================
# SESSION STATE
# ============================================================================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize advanced mode state
if "advanced_mode" not in st.session_state:
    st.session_state.advanced_mode = True  # Default to Advanced RAG


# ============================================================================
# SIDEBAR - SYSTEM CONTROL
# ============================================================================

with st.sidebar:
    st.title("🧠 Lilly-X Control")
    st.markdown("---")
    
    # === Advanced Mode Toggle ===
    st.markdown("### ⚙️ RAG Mode")
    
    try:
        # Use toggle for modern Streamlit
        advanced_mode = st.toggle(
            "Advanced RAG Pipeline",
            value=st.session_state.advanced_mode,
            help="Enable query decomposition, hybrid retrieval, and reranking"
        )
    except AttributeError:
        # Fallback for older Streamlit versions
        advanced_mode = st.checkbox(
            "Advanced RAG Pipeline",
            value=st.session_state.advanced_mode,
            help="Enable query decomposition, hybrid retrieval, and reranking"
        )
    
    # Update engine if mode changed
    if advanced_mode != st.session_state.advanced_mode:
        st.session_state.advanced_mode = advanced_mode
        engine.use_advanced = advanced_mode
        
        if advanced_mode:
            st.success("✅ Switched to **Advanced RAG**")
        else:
            st.info("ℹ️ Switched to **Simple RAG**")
        
        st.rerun()
    
    # Display current mode
    if st.session_state.advanced_mode:
        st.success("🚀 **Mode:** Advanced RAG")
        st.caption("Query decomposition + Hybrid retrieval + Reranking")
    else:
        st.info("⚡ **Mode:** Simple RAG")
        st.caption("Direct vector search")
    
    st.markdown("---")
    
    # === System Status ===
    st.markdown("### 📊 System Status")
    
    # Hardware info
    st.markdown("**Hardware:**")
    st.caption("🖥️ AMD Ryzen AI MAX-395")
    st.caption("💾 128GB RAM")
    st.caption("🎮 Radeon 8060S iGPU (32GB)")
    
    st.markdown("**Platform:**")
    st.caption("🐧 Fedora 43 Linux")
    st.caption(f"🐍 Python {sys.version.split()[0]}")
    
    st.markdown("---")
    
    # === Model Configuration ===
    st.markdown("### 🤖 AI Models")
    st.caption(f"**LLM:** {settings.llm_model}")
    st.caption(f"**Embeddings:** {settings.embedding_model}")
    
    if st.session_state.advanced_mode:
        st.caption(f"**Reranker:** {settings.reranker_model}")
    
    st.markdown("---")
    
    # === Service Status ===
    st.markdown("### 🔌 Services")
    st.write("✅ Qdrant (Vector Store)")
    st.write("✅ Ollama (LLM Server)")
    
    # Check Neo4j availability
    if hasattr(engine, '_neo4j_driver') and engine._neo4j_driver:
        st.write("✅ Neo4j (Knowledge Graph)")
    else:
        st.write("⚠️ Neo4j (Disabled)")
    
    st.markdown("---")
    
    # === Clear Chat ===
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat cleared!")
        st.rerun()
    
    st.markdown("---")
    st.caption("**Lilly-X** | Sovereign AI on Ryzen")
    st.caption(f"Collection: `{settings.qdrant_collection}`")


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.title("🧠 Lilly-X Chat")

# Subtitle with mode indicator
mode_badge = "🚀 Advanced" if st.session_state.advanced_mode else "⚡ Simple"
st.markdown(f"**Mode:** {mode_badge} RAG  |  **Strategy:** `{settings.retrieval_strategy}`")

st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            if meta.get("retrieval_time"):
                st.caption(f"⏱️ Retrieved in {meta['retrieval_time']:.2f}s")


# ============================================================================
# CHAT INPUT & RESPONSE GENERATION
# ============================================================================

if prompt := st.chat_input("Ask me anything about your knowledge base..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Show thinking indicator
            with st.spinner("🧠 Processing query..."):
                start_time = time.time()
                
                # Call RAG engine (handle async properly)
                try:
                    # Try async first
                    response = asyncio.run(engine.aquery(
                        user_query=prompt,
                        top_n=5
                    ))
                except RuntimeError:
                    # If async fails (nested event loop), use sync
                    response = engine.query(
                        user_query=prompt,
                        top_n=5
                    )
                
                retrieval_time = time.time() - start_time
            
            # Extract response text
            if isinstance(response, RAGResponse):
                response_text = response.response
                num_sources = len(response.source_nodes) if response.source_nodes else 0
            elif isinstance(response, str):
                response_text = response
                num_sources = 0
            else:
                response_text = str(response)
                num_sources = 0
            
            # Display response
            message_placeholder.markdown(response_text)
            
            # Show metadata
            metadata_parts = [f"⏱️ {retrieval_time:.2f}s"]
            if num_sources > 0:
                metadata_parts.append(f"📄 {num_sources} sources")
            
            st.caption(" | ".join(metadata_parts))
            
            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "num_sources": num_sources,
                    "mode": "advanced" if st.session_state.advanced_mode else "simple"
                }
            })
        
        except Exception as e:
            error_message = f"❌ **Error generating response:** {str(e)}"
            message_placeholder.error(error_message)
            
            # Show detailed error in expander
            with st.expander("🔧 Error Details"):
                st.code(f"{type(e).__name__}: {e}")
                
                # Suggest fixes
                st.markdown("**Possible fixes:**")
                if "connection" in str(e).lower():
                    st.info("🔌 Check if Ollama is running: `systemctl status ollama`")
                elif "timeout" in str(e).lower():
                    st.info("⏱️ LLM timeout - model may be loading. Try again in a moment.")
                else:
                    st.info("🔄 Try clearing chat and asking again.")
            
            # Save error to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "metadata": {"error": str(e)}
            })


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

# Info footer
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("💡 **Tip:** Ask complex questions to see Advanced RAG in action!")

with col2:
    st.caption(f"📚 Collection: `{settings.qdrant_collection}`")

with col3:
    msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.caption(f"💬 {msg_count} queries this session")
