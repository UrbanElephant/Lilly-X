import streamlit as st
import time
from src.rag_engine import RAGEngine
from src.config import settings


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

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to initialize RAG Engine: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    st.success(f"**Model:** `{settings.llm_model}`")
    st.info(f"**Embedding:** `{settings.embedding_model}`")
    st.markdown("---")
    st.markdown("### Status")
    st.write("✅ RAG Engine Ready")
    st.write("✅ Qdrant Connected")

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
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** (Score: {src['score']:.2f})")
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
            with st.spinner("Thinking..."):
                result = engine.query(prompt)
                full_response = result.response
                
                # simulate stream (optional, or if streaming supported by engine later)
                # for chunk in result.response.split():
                #     full_response += chunk + " "
                #     time.sleep(0.05)
                #     message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Prepare source data for history
                source_data = []
                if result.source_nodes:
                    with st.expander("📚 View Sources"):
                        for node in result.source_nodes:
                            meta = node.node.metadata
                            meta_output = format_metadata(meta)
                            
                            src_info = {
                                "source": meta.get('file_name', 'Unknown'),
                                "content": node.node.get_content().replace('\n', ' ')[:300] + "...",
                                "score": node.score,
                                "metadata": meta  # Store metadata in history
                            }
                            source_data.append(src_info)
                            st.markdown(f"**{src_info['source']}** (Score: {src_info['score']:.2f})")
                            st.caption(src_info['content'])
                            # Display metadata
                            if meta_output:
                                st.info(meta_output)
                            st.markdown("---")
            
            # Add Assistant Message to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": source_data
            })
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
