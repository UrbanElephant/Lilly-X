"""
Lilly-X Streamlit Chat Application - Architect Edition
Advanced RAG interface with Query Decomposition visualization.
"""

import sys
import os

# --- PATH FIX ---
# Add the project root directory to sys.path so 'from src.xxx' works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------------

import streamlit as st
import time
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4

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


# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Lilly-X - Architect Edition",
    page_icon="🧠",
    layout="wide"
)

# ========================================
# CACHED RESOURCES
# ========================================
@st.cache_resource
def get_engine():
    """Initialize RAG Engine (singleton pattern)."""
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}")
        st.stop()

@st.cache_resource
def get_memory_manager():
    """Initialize Memory Manager with persistence."""
    return MemoryManager(use_persistence=True)

@st.cache_resource
def get_graph_ops():
    """Initialize Graph Operations."""
    return GraphOperations()

# Initialize components
try:
    engine = get_engine()
    memory_mgr = get_memory_manager()
    graph_ops = get_graph_ops()
except Exception as e:
    st.error(f"❌ Failed to initialize components: {e}")
    st.info("Please ensure Ollama, Qdrant, and Neo4j are running.")
    st.stop()

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query_plan" not in st.session_state:
    st.session_state.last_query_plan = None

# ========================================
# SIDEBAR - CONFIGURATION COCKPIT
# ========================================
with st.sidebar:
    st.title("⚙️ Configuration Cockpit")
    st.markdown("---")
    
    # === Retrieval Strategy Status ===
    st.markdown("### 🎯 Retrieval Strategy")
    strategy = settings.retrieval_strategy
    strategy_emoji = {
        "semantic": "🔍",
        "sentence_window": "🪟",
        "hierarchical": "🏗️"
    }
    st.success(f"{strategy_emoji.get(strategy, '🔍')} **Active:** `{strategy.upper()}`")
    
    if strategy == "sentence_window":
        st.info(f"📏 Window Size: {settings.sentence_window_size} sentences")
    elif strategy == "hierarchical":
        st.info(f"📦 Parent: {settings.parent_chunk_size} | Child: {settings.child_chunk_size}")
    
    st.markdown("---")
    
    # === Pipeline Controls ===
    st.markdown("### 🔧 Pipeline Controls")
    
    # Top-K Slider
    retrieval_count = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=15,
        value=settings.top_k_final,
        help="Number of documents to retrieve and display"
    )
    
    # Reranker Toggle
    try:
        activate_reranker = st.toggle(
            "🎯 Activate Reranker",
            value=True,
            help="Two-stage retrieval: broad search → CrossEncoder reranking → top results"
        )
    except AttributeError:
        # Fallback for older Streamlit versions
        activate_reranker = st.checkbox(
            "🎯 Activate Reranker",
            value=True,
            help="Two-stage retrieval: broad search → CrossEncoder reranking → top results"
        )
    
    # Visual feedback
    if activate_reranker:
        st.caption(f"✨ Mode: Two-stage ({retrieval_count * 3} → rerank → {retrieval_count})")
    else:
        st.caption(f"⚡ Mode: Direct vector search ({retrieval_count} results)")
    
    st.markdown("---")
    
    # === Clear Chat Button ===
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_query_plan = None
        memory_mgr.clear_history(st.session_state.session_id)
        st.success("Chat history cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # === 🧠 AGENT REASONING SECTION ===
    st.markdown("### 🧠 Agent Reasoning")
    
    if st.session_state.last_query_plan:
        from src.schemas import QueryPlan
        plan = st.session_state.last_query_plan
        
        # Check if it's a simple or complex query
        if len(plan.sub_queries) == 1:
            st.info("🎯 **Direct Retrieval**")
            st.caption("Simple query - no decomposition needed")
        else:
            st.success(f"🔍 **Decomposed into {len(plan.sub_queries)} Sub-Queries**")
            
            # Display each sub-query with intent
            for idx, sq in enumerate(plan.sub_queries, 1):
                intent_emoji = {
                    "factual": "📊",
                    "workflow": "⚙️",
                    "comparison": "⚖️"
                }
                emoji = intent_emoji.get(sq.intent.value, "❓")
                
                with st.expander(f"{emoji} Sub-Query {idx}: {sq.intent.value.capitalize()}"):
                    st.markdown(f"**🎯 Intent:** `{sq.intent.value.upper()}`")
                    st.markdown(f"**📝 Focused Query:**")
                    st.info(sq.focused_query)
                    if sq.original_text != sq.focused_query:
                        st.markdown(f"**💬 Original:**")
                        st.caption(sq.original_text)
    else:
        st.caption("ℹ️ No query analyzed yet")
        st.caption("Ask a question to see reasoning steps!")
    
    st.markdown("---")
    
    # === Model Information ===
    st.markdown("### 🤖 Model Configuration")
    st.caption(f"**LLM:** `{settings.llm_model}`")
    st.caption(f"**Embeddings:** `{settings.embedding_model}`")
    if activate_reranker:
        st.caption(f"**Reranker:** `{settings.reranker_model}`")
    
    st.markdown("---")
    
    # === System Status ===
    st.markdown("### 📊 System Status")
    st.write("✅ RAG Engine Ready")
    st.write("✅ Qdrant (Vector Store)")
    st.write("✅ Memory Manager")
    
    # Check Neo4j availability
    if hasattr(engine, '_neo4j_driver') and engine._neo4j_driver:
        st.write("✅ Neo4j (Knowledge Graph)")
    else:
        st.write("⚠️ Neo4j (Disabled)")
    
    st.markdown("---")
    st.caption(f"Session: `{str(st.session_state.session_id)[:8]}...`")

# ========================================
# MAIN CHAT INTERFACE
# ========================================
st.title("🧠 Lilly-X - Architect Edition")
st.markdown(f"**Persona:** Lilly-X (Advanced RAG with Query Decomposition) | **Strategy:** `{settings.retrieval_strategy}`")

# Display Chat History
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Assistant message: show sources and reasoning
        if message["role"] == "assistant":
            
            # === 📄 SOURCE NODES EXPANDER ===
            if "sources" in message and message["sources"]:
                with st.expander("📄 Source Nodes"):
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
                        
                        # Content Snippet
                        st.caption("**Content Snippet:**")
                        snippet = src['content'][:300]
                        st.text(snippet + ("..." if len(src['content']) > 300 else ""))
                        
                        st.markdown("---")
            
            # === 🧠 QUERY PLAN EXPANDER (Historical) ===
            if "query_plan" in message and message["query_plan"]:
                with st.expander("🧠 Query Planning"):
                    plan = message["query_plan"]
                    st.markdown("### Query Decomposition")
                    st.caption(f"**Original Query:** _{plan.root_query}_")
                    
                    if len(plan.sub_queries) > 1:
                        st.success(f"Complex query decomposed into {len(plan.sub_queries)} sub-queries")
                        for sq_idx, sq in enumerate(plan.sub_queries, 1):
                            intent_badge = {"factual": "📊", "workflow": "⚙️", "comparison": "⚖️"}
                            st.markdown(f"{sq_idx}. **{intent_badge.get(sq.intent.value, '❓')} [{sq.intent.value.upper()}]** {sq.focused_query}")
                    else:
                        st.info("Direct retrieval - simple query")
            
            # === Feedback Buttons ===
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

# ========================================
# CHAT INPUT & RESPONSE GENERATION
# ========================================
if prompt := st.chat_input("What would you like to know?"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # === 1. THINKING & PLANNING PHASE ===
            with st.status("🧠 Thinking & Planning...", expanded=True) as status:
                t0 = time.time()
                
                # Initialize Progress Bar
                p_bar = st.progress(0)
                
                # Step 1: Query Planning (0% → 15%)
                st.write("🧠 Analyzing query structure...")
                from src.schemas import QueryPlan
                
                # Actually call plan_query to get decomposition
                query_plan = engine.plan_query(prompt)
                
                # Store in session state for sidebar display
                st.session_state.last_query_plan = query_plan
                
                # Show decomposition result
                if len(query_plan.sub_queries) > 1:
                    st.write(f"  → Decomposed into {len(query_plan.sub_queries)} sub-queries")
                    for sq_idx, sq in enumerate(query_plan.sub_queries, 1):
                        st.write(f"     {sq_idx}. [{sq.intent.value}] {sq.focused_query[:60]}...")
                else:
                    st.write("  → Direct retrieval (simple query)")
                    
                p_bar.progress(15)
                
                # Step 2: Entity Analysis (15% → 25%)
                st.write("🔍 Identifying entities...")
                words = prompt.split()
                potential_entities = [w.strip('.,!?') for w in words if w and w[0].isupper() and len(w) > 1]
                resolved_entities = []
                for entity in potential_entities[:3]:
                    canonical = graph_ops.resolve_entity(entity)
                    resolved_entities.append(canonical)
                
                time.sleep(0.05)  # Small visual pause for UX
                p_bar.progress(25)
                
                # Step 3: Memory Context (25% → 35%)
                st.write("📚 Loading conversation history...")
                session = memory_mgr.get_history(st.session_state.session_id)
                conversation_history = session.to_string(max_turns=settings.memory_window_size)
                p_bar.progress(35)
                
                # Step 4: Hybrid Retrieval (35% → 60%)
                st.write(f"🔎 Scanning {settings.retrieval_strategy} index...")
                vector_nodes, graph_context = engine.retrieve(
                    query_text=prompt,
                    top_k=retrieval_count,
                    use_reranker=activate_reranker
                )
                p_bar.progress(60)
                
                # Step 5: Results Feedback (60% → 75%)
                st.write(f"✅ Found {len(vector_nodes)} relevant text chunks")
                
                if graph_context:
                    graph_fact_count = len([line for line in graph_context.split("\n") if line.strip() and not line.startswith("===")])
                    st.write(f"🕸️ Found {graph_fact_count} knowledge graph facts")
                    time.sleep(0.05)
                
                p_bar.progress(75)
                
                # Step 6: Context Assembly (75% → 90%)
                st.write("🧩 Assembling context & building prompt...")
                context_parts = []
                if graph_context:
                    context_parts.append(graph_context)
                
                # Prepare Debug Info
                debug_context = []
                
                for node_idx, node in enumerate(vector_nodes, 1):
                    meta = node.node.metadata
                    filename = meta.get('file_name', 'Unknown')
                    node_content = node.node.get_content()
                    
                    # Format with [Source: filename] for citation enforcement
                    context_parts.append(f"[Source: {filename}]\n{node_content}\n")
                    
                    # Collect debug information
                    debug_info = {
                        'score': node.score,
                        'node_text': node_content,
                        'metadata': meta
                    }
                    
                    # Extract window context if available
                    if 'window' in meta:
                        debug_info['window_context'] = meta['window']
                    
                    debug_context.append(debug_info)
                
                context_str = "\n\n".join(context_parts)
                p_bar.progress(90)
                
                # Step 7: Final Prompt (90% → 100%)
                st.write("✍️ Finalizing system prompt...")
                final_prompt = prompts.build_qa_prompt(
                    user_query=prompt,
                    context_str=context_str,
                    conversation_history=conversation_history
                )
                
                p_bar.progress(100)
                retrieval_time = time.time() - t0
                status.update(
                    label=f"✅ Planning complete! ({retrieval_time:.2f}s)", 
                    state="complete", 
                    expanded=False
                )
            
            # === 2. GENERATION PHASE with LIVE TELEMETRY ===
            from llama_index.core import Settings as LlamaSettings
            
            with st.spinner("✍️ Reading context & formulating answer..."):
                # Start generation timer
                t_gen_start = time.time()
                
                try:
                    # Initialize streaming
                    streaming_response = LlamaSettings.llm.stream_complete(final_prompt)
                    
                    # Create separate placeholders for text and stats
                    text_area = st.empty()
                    stats_area = st.empty()
                    
                    # Live Token Loop with Telemetry
                    token_count = 0
                    
                    for chunk in streaming_response:
                        token_count += 1
                        
                        if hasattr(chunk, 'delta'):
                            full_response += chunk.delta
                        else:
                            full_response += str(chunk)
                        
                        # Calculate live metrics
                        elapsed = time.time() - t_gen_start
                        tps = token_count / elapsed if elapsed > 0 else 0
                        
                        # Update text with cursor
                        text_area.markdown(full_response + "▌")
                        
                        # Update live stats
                        stats_area.caption(f"⚡ Generating... | {token_count} tokens | {tps:.1f} tokens/s")
                    
                    # Finalize - remove cursor
                    text_area.markdown(full_response)
                    
                    # Calculate final metrics
                    total_gen_time = time.time() - t_gen_start
                    avg_tps = token_count / total_gen_time if total_gen_time > 0 else 0
                    
                    # Permanent stats footer with HTML formatting
                    stats_area.markdown(f"""
                    <div style="display: flex; gap: 15px; font-size: 0.85em; color: #666; padding: 5px 0;">
                        <span title="Time spent retrieving and assembling context">⏱️ Retrieval: {retrieval_time:.2f}s</span>
                        <span title="Time spent generating response">✍️ Generation: {total_gen_time:.2f}s</span>
                        <span title="Average tokens generated per second">🚀 Speed: {avg_tps:.1f} tok/s</span>
                        <span title="Total tokens in response">📊 Tokens: {token_count}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except AttributeError:
                    # Fallback if streaming is not available
                    response = LlamaSettings.llm.complete(final_prompt)
                    full_response = str(response)
                    text_area = st.empty()
                    text_area.markdown(full_response)
                    
                    total_gen_time = time.time() - t_gen_start
                    
                    st.caption(f"⏱️ Retrieval: {retrieval_time:.2f}s | Generation: {total_gen_time:.2f}s")
            
            generation_time = time.time() - t_gen_start
            
            # === Prepare Sources for History ===
            source_data = []
            with st.expander("📄 Source Nodes"):
                mode_used = "Two-Stage Reranker" if activate_reranker else "Direct Vector Search"
                st.caption(f"**Retrieval Mode:** {mode_used}")
                st.caption(f"**Documents Retrieved:** {len(vector_nodes)}**")
                st.markdown("---")
                
                for src_idx, node in enumerate(vector_nodes, 1):
                    meta = node.node.metadata
                    
                    src_info = {
                        "source": meta.get('file_name', 'Unknown'),
                        "content": node.node.get_content(),
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
                    snippet = src_info['content'][:300]
                    st.text(snippet + ("..." if len(src_info['content']) > 300 else ""))
                    
                    st.markdown("---")
            
            # === Save to Memory ===
            memory_mgr.add_message(st.session_state.session_id, "user", prompt)
            memory_mgr.add_message(st.session_state.session_id, "assistant", full_response)
            
            # === Add to UI History ===
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data,
                "debug_context": debug_context,
                "graph_context": graph_context,
                "query_plan": query_plan,  # Store query plan for historical review
                "used_reranker": activate_reranker,
                "performance": {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time
                }
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
            st.error(f"❌ Error generating response: {e}")
            import traceback
            with st.expander("🔧 Error Details"):
                st.code(traceback.format_exc())
                st.info("Please check if Ollama service is running and all models are loaded.")
