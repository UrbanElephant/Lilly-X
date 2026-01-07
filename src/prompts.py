"""Prompt templates for Lilly-X AI Knowledge Assistant."""

# ============================================================
# System Prompts
# ============================================================

SYSTEM_HEADER = """You are Lilly-X, an advanced AI Knowledge Assistant with hybrid GraphRAG capabilities.

Your mission is to provide accurate, well-reasoned answers based on the Context provided from the knowledge base (documents + knowledge graph).

Core Principles:
1. **Grounded Responses**: Base your answers strictly on the provided Context
2. **Citation**: Reference specific sources when making claims
3. **Honesty**: If the Context doesn't contain enough information, acknowledge the limitation
4. **Clarity**: Provide clear, structured responses that are easy to understand
"""

INSTRUCTION_COT = """Before answering, think step-by-step to verify facts from the Context:
1. Identify the key entities and concepts mentioned in the question
2. Review the provided Context for relevant information
3. Cross-reference facts from multiple sources when available
4. Formulate a coherent answer grounded in the Context
5. Indicate confidence level based on source quality and agreement
"""

# ============================================================
# Context Sections
# ============================================================

def format_conversation_history(history: str) -> str:
    """Format conversation history for inclusion in prompt."""
    if not history:
        return ""
    
    return f"""
## 📝 Conversation History

{history}

---
"""

def format_graph_context(entities: list, relationships: list) -> str:
    """Format knowledge graph context for inclusion in prompt."""
    if not entities and not relationships:
        return ""
    
    sections = []
    
    if entities:
        entity_list = "\n".join([f"- {entity}" for entity in entities])
        sections.append(f"""### 🔍 Related Entities
{entity_list}""")
    
    if relationships:
        rel_list = "\n".join([f"- {rel}" for rel in relationships])
        sections.append(f"""### 🔗 Knowledge Graph Relationships
{rel_list}""")
    
    if sections:
        return f"""
## 🕸️ Knowledge Graph Context

{chr(10).join(sections)}

---
"""
    return ""

def format_vector_context(sources: list) -> str:
    """Format vector retrieval sources for inclusion in prompt."""
    if not sources:
        return ""
    
    source_sections = []
    for idx, source in enumerate(sources, 1):
        content = source.get('content', 'No content')
        metadata = source.get('metadata', {})
        
        # Include metadata if available
        meta_info = []
        if 'file_name' in metadata:
            meta_info.append(f"Source: {metadata['file_name']}")
        if 'document_type' in metadata:
            meta_info.append(f"Type: {metadata['document_type']}")
        if 'authors' in metadata and metadata['authors'] != "None":
            meta_info.append(f"Author: {metadata['authors']}")
        
        meta_str = " | ".join(meta_info) if meta_info else "Unknown Source"
        
        source_sections.append(f"""### Source {idx}: {meta_str}
{content}
""")
    
    return f"""
## 📚 Retrieved Context (Vector Search)

{chr(10).join(source_sections)}

---
"""

# ============================================================
# Main Prompt Template
# ============================================================

FORMAT_TEMPLATE = """{system_header}

{instruction_cot}

{conversation_history}{graph_context}{vector_context}

## ❓ Current Question

{user_query}

## 💬 Your Response

Please provide a comprehensive answer based on the Context above. If the Context doesn't contain sufficient information to answer the question fully, acknowledge this limitation."""


# ============================================================
# Alternative: Concise Template
# ============================================================

CONCISE_TEMPLATE = """{system_header}

{conversation_history}{vector_context}

User Question: {user_query}

Assistant Response:"""


# ============================================================
# Helper Functions
# ============================================================

def build_prompt(
    user_query: str,
    conversation_history: str = "",
    graph_context_entities: list = None,
    graph_context_relationships: list = None,
    vector_sources: list = None,
    use_cot: bool = True,
    concise: bool = False
) -> str:
    """
    Build a complete prompt with all context sections.
    
    Args:
        user_query: The user's current question
        conversation_history: Formatted conversation history string
        graph_context_entities: List of related entity names from graph
        graph_context_relationships: List of relationship descriptions
        vector_sources: List of source dicts from vector retrieval
        use_cot: Whether to include Chain-of-Thought instructions
        concise: Whether to use concise template (minimal formatting)
        
    Returns:
        Complete formatted prompt string
    """
    if concise:
        return CONCISE_TEMPLATE.format(
            system_header=SYSTEM_HEADER,
            conversation_history=format_conversation_history(conversation_history),
            vector_context=format_vector_context(vector_sources or []),
            user_query=user_query
        )
    
    return FORMAT_TEMPLATE.format(
        system_header=SYSTEM_HEADER,
        instruction_cot=INSTRUCTION_COT if use_cot else "",
        conversation_history=format_conversation_history(conversation_history),
        graph_context=format_graph_context(
            graph_context_entities or [],
            graph_context_relationships or []
        ),
        vector_context=format_vector_context(vector_sources or []),
        user_query=user_query
    )


# ============================================================
# Debugging Helper
# ============================================================

def log_prompt_stats(prompt: str) -> dict:
    """
    Calculate and return statistics about a prompt.
    
    Useful for debugging token limits and context window usage.
    """
    return {
        "total_chars": len(prompt),
        "total_words": len(prompt.split()),
        "estimated_tokens": len(prompt) // 4,  # Rough estimate: 1 token ≈ 4 chars
        "lines": len(prompt.split('\n'))
    }
