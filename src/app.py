import streamlit as st
import time
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4, UUID
from src.rag_engine import RAGEngine
from src.config import settings
from src.memory import MemoryManager
from src.graph_ops import GraphOperations
from src import prompts


def format_metadata(metadata: dict) -> str:
    """Formats rich metadata for display including Golden Source fields."""
    extras = []
    
    # --- Golden Source Header ---
    header_parts = []
    if "document_type" in metadata and metadata['document_type'] != "Unknown":
        header_parts.append(f"📄 {metadata['document_type']}")
    if "authors" in metadata and metadata['authors'] not in ["None", "Unknown"]:
        header_parts.append(f"✍️ {metadata['authors']}")
    if "key_dates" in metadata and metadata['key_dates'] != "Unknown":
        header_parts.append(f"📅 {metadata['key_dates']}")
        
    if header_parts:
        extras.append(f"**{' | '.join(header_parts)}**")
    
    # Add page label if present
    if "page_label" in metadata:
        extras.append(f"**Page:** {metadata['page_label']}")
    
    # Add filename
    if "file_name" in metadata:
        extras.append(f"**File:** {metadata['file_name']}")
        
    return "\n".join(extras) if extras else ""


def log_feedback(query: str, response: str, feedback_type: str):
    """Log user feedback to feedback.json file."""
    feedback_file = Path("feedback.json")
    
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response[:200],  # Truncate for storage
        "feedback": feedback_type,
        "session_id": str(st.session_state.session_id)
    }
    
    # Load existing feedback
    if feedback_file.exists():
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except:
            feedback_data = []
    else:
        feedback_data = []
    
    # Append new feedback
    feedback_data.append(feedback_entry)
    
    # Save
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)


# Page Config
st.set_page_config(
    page_title="Lilly-X - Local Knowledge Base",
    page_icon="📚",
    layout="wide"
)

# Initialize RAG Engine (Cached Resource)
@st.cache_resource
def get_engine():
    return RAGEngine()

# Initialize Memory Manager (Cached Resource with Persistence)
@st.cache_resource
def get_memory_manager():
    return MemoryManager(use_persistence=True)

# Initialize Graph Operations (Cached Resource)
@st.cache_resource
def get_graph_ops():
    return GraphOperations()

try:
    engine = get_engine()
    memory_mgr = get_memory_manager()
    graph_ops = get_graph_ops()
except Exception as e:
    st.error(f"Failed to initialize components: {e}")
    st.stop()

# Initialize Session ID in Streamlit Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4()

# ========================================
# SIDEBAR - THE COCKPIT
# ========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    # === Pipeline Settings Section ===
    st.markdown("### 🔧 Pipeline Settings")
    
    # Top-K Slider
    retrieval_count = st.slider(
        "Retrieval Count (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of documents to retrieve and show"
    )
    
    # Reranker Toggle (using st.toggle for modern Streamlit)
    try:
        activate_reranker = st.toggle(
            "Activate Reranker (Slow but precise)",
            value=True,
            help="Two-stage retrieval: broad search → CrossEncoder reranking → top results"
        )
    except AttributeError:
        # Fallback for older Streamlit versions
        activate_reranker = st.checkbox(
            "Activate Reranker (Slow but precise)",
            value=True,
            help="Two-stage retrieval: broad search → CrossEncoder reranking → top results"
        )
    
    # Visual feedback
    if activate_reranker:
        st.caption(f"✨ Mode: Two-stage ({retrieval_count * 3} → rerank → {retrieval_count})")
    else:
        st.caption(f"⚡ Mode: Direct vector search ({retrieval_count} results)")
    
    st.markdown("---")
    # === END Pipeline Settings ===
    
    # Model Info
    st.markdown("### 🤖 Models")
    st.success(f"**LLM:** `{settings.llm_model}`")
    st.info(f"**Embeddings:** `{settings.embedding_model}`")
    if activate_reranker:
        st.info(f"**Reranker:** `{settings.reranker_model}`")
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.write("✅ RAG Engine Ready")
    st.write("✅ Qdrant (Vector Store)")
    st.write("✅ Memory Manager")
    
    # Check if Neo4j is available
    if hasattr(engine, '_neo4j_driver') and engine._neo4j_driver:
        st.write("✅ Neo4j (Knowledge Graph)")
    else:
        st.write("⚠️ Neo4j (Disabled)")
    
    st.markdown("---")
    st.caption(f"Session: `{str(st.session_state.session_id)[:8]}...`")

# ========================================
# MAIN INTERFACE
# ========================================
st.title("Lilly-X - Local Knowledge Base 🧠")
st.markdown("Ask questions about your ingested documents.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If assistant message, show sources and feedback
        if message["role"] == "assistant":
            # Sources
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources & Evidence"):
                    mode_used = "Two-Stage Reranker" if message.get("used_reranker") else "Direct Vector Search"
                    st.caption(f"**Retrieval Mode:** {mode_used}")
                    st.caption(f"**Documents Retrieved:** {len(message['sources'])}")
                    st.markdown("---")
                    
                    for src_idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {src_idx}: {src['source']}**")
                        
                        # Score Display
                        score = src.get('score', 0.0)
                        score_label = "🎯 Rerank Score" if message.get("used_reranker") else "🔍 Vector Similarity"
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(label=score_label, value=f"{score:.4f}")
                        with col2:
                            # Metadata
                            meta_text = format_metadata(src.get('metadata', {}))
                            if meta_text:
                                st.markdown(meta_text)
                        
                        # Content Snippet (first 200 chars)
                        st.caption("**Content Snippet:**")
                        snippet = src['content'][:200]
                        st.text(snippet + ("..." if len(src['content']) > 200 else ""))
                        
                        st.markdown("---")
            
            # Feedback buttons
            if "feedback" not in message:
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"up_{idx}"):
                        log_feedback(
                            st.session_state.messages[idx-1]["content"],
                            message["content"],
                            "positive"
                        )
                        st.session_state.messages[idx]["feedback"] = "positive"
                        st.success("Thanks!")
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{idx}"):
                        log_feedback(
                            st.session_state.messages[idx-1]["content"],
                            message["content"],
                            "negative"
                        )
                        st.session_state.messages[idx]["feedback"] = "negative"
                        st.warning("Noted!")
                        st.rerun()
            else:
                st.caption(f"✓ Feedback: {message['feedback']}")

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("🔍 Processing query..."):
                # === Memory & Graph Context ===
                session = memory_mgr.get_history(st.session_state.session_id)
                conversation_history = session.to_string(max_turns=settings.memory_window_size)
                
                # === Entity Resolution ===
                words = prompt.split()
                potential_entities = [w.strip('.,!?') for w in words if w and w[0].isupper() and len(w) > 1]
                resolved_entities = []
                for entity in potential_entities[:3]:
                    canonical = graph_ops.resolve_entity(entity)
                    resolved_entities.append(canonical)
                
                # === Hybrid Retrieval with User Config ===
                # CRITICAL: Pass sidebar values to engine
                vector_nodes, graph_context = engine.retrieve(
                    query_text=prompt,
                    top_k=retrieval_count,          # From slider
                    use_reranker=activate_reranker  # From toggle
                )
                
                # Build context string with source identifiers
                context_parts = []
                if graph_context:
                    context_parts.append(graph_context)
                
                for idx, node in enumerate(vector_nodes, 1):
                    meta = node.node.metadata
                    filename = meta.get('file_name', 'Unknown')
                    content = node.node.get_content()
                    # Format with [Source: filename] for citation enforcement
                    context_parts.append(f"[Source: {filename}]\n{content}\n")
                
                context_str = "\n\n".join(context_parts)
                
                # === Generate Response with Strict Citation Prompt ===
                final_prompt = prompts.build_qa_prompt(
                    user_query=prompt,
                    context_str=context_str,
                    conversation_history=conversation_history
                )
                
                # Call LLM directly
                from llama_index.core import Settings as LlamaSettings
                response = LlamaSettings.llm.complete(final_prompt)
                full_response = str(response)
                
                # Display
                message_placeholder.markdown(full_response)
                
                # === Debug Info ===
                with st.expander("🧠 Retrieval Details"):
                    st.write(f"**Mode:** {'Two-Stage Reranker' if activate_reranker else 'Direct Vector'}")
                    st.write(f"**Retrieved:** {len(vector_nodes)} documents")
                    st.write(f"**Top-K Setting:** {retrieval_count}")
                    if conversation_history:
                        st.write(f"**Memory:** {len(session.messages)} turns loaded")
                    if resolved_entities:
                        st.write(f"**Entities:** {', '.join(resolved_entities[:3])}")
                
                # === Prepare sources for history ===
                source_data = []
                with st.expander("📚 Sources & Evidence"):
                    mode_used = "Two-Stage Reranker" if activate_reranker else "Direct Vector Search"
                    st.caption(f"**Retrieval Mode:** {mode_used}")
                    st.caption(f"**Documents Retrieved:** {len(vector_nodes)}")
                    st.markdown("---")
                    
                    for src_idx, node in enumerate(vector_nodes, 1):
                        meta = node.node.metadata
                        
                        src_info = {
                            "source": meta.get('file_name', 'Unknown'),
                            "content": node.node.get_content()[:200],
                            "score": node.score,
                            "metadata": meta
                        }
                        source_data.append(src_info)
                        
                        st.markdown(f"**Source {src_idx}: {src_info['source']}**")
                        
                        # Score Display
                        score_label = "🎯 Rerank Score" if activate_reranker else "🔍 Vector Similarity"
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(label=score_label, value=f"{src_info['score']:.4f}")
                        with col2:
                            meta_text = format_metadata(meta)
                            if meta_text:
                                st.markdown(meta_text)
                        
                        # Content Snippet
                        st.caption("**Content Snippet:**")
                        st.text(src_info['content'] + "...")
                        
                        st.markdown("---")
            
            # === Save to Memory ===
            memory_mgr.add_message(st.session_state.session_id, "user", prompt)
            memory_mgr.add_message(st.session_state.session_id, "assistant", full_response)
            
            # === Add to UI History ===
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data,
                "used_reranker": activate_reranker
            })
            
            # === Feedback Buttons ===
            st.markdown("---")
            st.caption("**Was this helpful?**")
            col1, col2, col3 = st.columns([1, 1, 10])
            
            with col1:
                if st.button("👍", key="thumbs_up_current"):
                    log_feedback(prompt, full_response, "positive")
                    st.success("Thanks!")
            
            with col2:
                if st.button("👎", key="thumbs_down_current"):
                    log_feedback(prompt, full_response, "negative")
                    st.warning("Noted!")
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
