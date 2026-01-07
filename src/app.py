import streamlit as st
import time
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
    # Display Type, Author, and Date prominently if available
    header_parts = []
    if "document_type" in metadata:
        header_parts.append(f"📄 {metadata['document_type']}")
    if "authors" in metadata and metadata['authors'] != "None":
        header_parts.append(f"✍️ {metadata['authors']}")
    if "key_dates" in metadata and metadata['key_dates'] != "Unknown":
        header_parts.append(f"📅 {metadata['key_dates']}")
        
    if header_parts:
        extras.append(f"**{' | '.join(header_parts)}**")
    # ----------------------------

    # Format Questions
    if "questions_this_excerpt_can_answer" in metadata:
        q_str = metadata["questions_this_excerpt_can_answer"]
        extras.append(f"**❓ Relevante Fragen:**\n{q_str}")
    
    # Format Entities (Handle both potential key names from different extractors)
    if "entities" in metadata:
        e_str = metadata["entities"]
        extras.append(f"**🏢 Entitäten:**\n{e_str}")
    elif "excerpt_keywords" in metadata:
        e_str = metadata["excerpt_keywords"]
        extras.append(f"**🏢 Keywords:**\n{e_str}")
        
    return "\n\n".join(extras)

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

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    st.success(f"**Model:** `{settings.llm_model}`")
    st.info(f"**Embedding:** `{settings.embedding_model}`")
    st.markdown("---")
    st.markdown("### Hybrid GraphRAG Status")
    st.write("✅ RAG Engine Ready")
    st.write("✅ Qdrant (Vector Store)")
    st.write("✅ Memory Manager (Persistent)")
    
    # Check if Neo4j is available
    if hasattr(engine, '_neo4j_driver') and engine._neo4j_driver:
        st.write("✅ Neo4j (Knowledge Graph)")
        st.write("✅ Graph Operations Ready")
    else:
        st.write("⚠️ Neo4j (Disabled)")
    
    st.markdown("---")
    st.caption(f"Session ID: `{str(st.session_state.session_id)[:8]}...`")
    st.caption(f"Memory Window: {settings.memory_window_size} turns")

# Main Interface
st.title("Lilly-X - Local Knowledge Base 🧠")
st.markdown("Ask questions about your ingested documents.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it was an assistant message and had sources, show them
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                st.caption(f"Showing Top 5 results from Re-Ranker (Cross-Encoder)")
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** (Re-Rank Confidence: {src['score']:.2f}) 🎯")
                    st.caption(src['content'])
                    # Display metadata if present
                    meta_text = format_metadata(src.get('metadata', {}))
                    if meta_text:
                        st.info(meta_text)
                    st.markdown("---")

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("🔍 Processing query..."):
                # ===== STEP A: Get Conversation Session =====
                session = memory_mgr.get_history(st.session_state.session_id)
                conversation_history = session.to_string(max_turns=settings.memory_window_size)
                
                # ===== STEP B: Resolve Entities via Graph =====
                # Simple entity extraction (just capitalized words for now)
                # TODO: Use proper NER or LLM-based extraction
                words = prompt.split()
                potential_entities = [w.strip('.,!?') for w in words if w[0].isupper() and len(w) > 1]
                
                resolved_entities = []
                for entity in potential_entities[:3]:  # Limit to top 3 for performance
                    canonical = graph_ops.resolve_entity(entity)
                    if canonical != entity:
                        resolved_entities.append(f"{entity} → {canonical}")
                    else:
                        resolved_entities.append(entity)
                
                # ===== STEP C: Expand Graph Context =====
                graph_context_relationships = []
                if resolved_entities and hasattr(engine, '_neo4j_driver') and engine._neo4j_driver:
                    for entity in resolved_entities[:2]:  # Top 2 entities
                        entity_name = entity.split(' → ')[-1]  # Get canonical name
                        try:
                            related = graph_ops.expand_query(entity_name, depth=1)
                            if len(related) > 1:
                                graph_context_relationships.append(
                                    f"'{entity_name}' is related to: {', '.join(related[1:4])}"
                                )
                        except Exception:
                            pass  # Entity not in graph, continue
                
                # ===== Standard Vector Retrieval =====
                result = engine.query(prompt)
                full_response = result.response
                
                # ===== STEP D: Build Enhanced Prompt (for logging/future use) =====
                # Note: Current engine.query doesn't use this yet, but we log it
                vector_sources = []
                if result.source_nodes:
                    for node in result.source_nodes:
                        vector_sources.append({
                            'content': node.node.get_content()[:500],
                            'metadata': node.node.metadata,
                            'score': node.score
                        })
                
                # Build full prompt for debugging
                full_prompt = prompts.build_prompt(
                    user_query=prompt,
                    conversation_history=conversation_history,
                    graph_context_entities=resolved_entities,
                    graph_context_relationships=graph_context_relationships,
                    vector_sources=vector_sources,
                    use_cot=True
                )
                
                # Log prompt stats for debugging
                prompt_stats = prompts.log_prompt_stats(full_prompt)
                
                # ===== Display Response =====
                message_placeholder.markdown(full_response)
                
                # ===== Show Thinking Process (Debug Info) =====
                with st.expander("🧠 Thinking Process (Debug)"):
                    st.caption("**Context Retrieval Process**")
                    if conversation_history:
                        st.write(f"📝 Loaded {len(session.messages)} messages from history")
                    if resolved_entities:
                        st.write(f"🔍 Resolved entities: {', '.join(resolved_entities)}")
                    if graph_context_relationships:
                        st.write("🕸️ Graph context:")
                        for rel in graph_context_relationships:
                            st.caption(f"  - {rel}")
                    st.write(f"📊 Prompt stats: {prompt_stats['estimated_tokens']} tokens (est.)")
                
                # Prepare source data for history
                source_data = []
                if result.source_nodes:
                    with st.expander("📚 View Sources"):
                        st.caption(f"Showing Top 5 results from Re-Ranker (Cross-Encoder)")
                        for node in result.source_nodes:
                            meta = node.node.metadata
                            meta_output = format_metadata(meta)
                            
                            src_info = {
                                "source": meta.get('file_name', 'Unknown'),
                                "content": node.node.get_content().replace('\n', ' ')[:300] + "...",
                                "score": node.score,
                                "metadata": meta
                            }
                            source_data.append(src_info)
                            st.markdown(f"**{src_info['source']}** (Re-Rank Confidence: {src_info['score']:.2f}) 🎯")
                            st.caption(src_info['content'])
                            if meta_output:
                                st.info(meta_output)
                            st.markdown("---")
            
            # ===== STEP F & G: Save to Memory =====
            memory_mgr.add_message(st.session_state.session_id, "user", prompt)
            memory_mgr.add_message(st.session_state.session_id, "assistant", full_response)
            
            # Add Assistant Message to History (Streamlit UI)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": source_data
            })
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
