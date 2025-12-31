import streamlit as st
import time
from src.rag_engine import RAGEngine
from src.config import settings

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
                            src_info = {
                                "source": node.node.metadata.get('file_name', 'Unknown'),
                                "content": node.node.get_content().replace('\n', ' ')[:300] + "...",
                                "score": node.score
                            }
                            source_data.append(src_info)
                            st.markdown(f"**{src_info['source']}** (Score: {src_info['score']:.2f})")
                            st.caption(src_info['content'])
                            st.markdown("---")
            
            # Add Assistant Message to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": source_data
            })
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
