"""Prompt templates for Lilly-X AI Knowledge Assistant."""

# ============================================================
# System Prompts
# ============================================================

SYSTEM_HEADER = """You are Lilly-X, an advanced AI Knowledge Assistant with hybrid GraphRAG capabilities.

Your mission is to provide accurate, well-reasoned answers based on the Context provided from the knowledge base (documents + knowledge graph).

Core Principles:
1. **Grounded Responses**: Base your answers strictly on the provided Context
2. **Strict Citation**: You MUST cite your sources using the provided metadata. When making a claim, reference the filename or source date directly (e.g., "According to technical_spec.pdf..." or "As stated in the 2026-01-05 report..."). Do NOT reference the 'context' abstractly.
3. **Honesty**: If the Context doesn't contain enough information, acknowledge the limitation
4. **Clarity**: Provide clear, structured responses that are easy to understand

CRITICAL CITATION RULE:
- Never say "according to the context" or "the document states"
- Always use specific source identifiers: filenames, dates, authors, or page numbers
- Example: "According to Finetuning.pdf (page 12), the recommended learning rate is..."
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


# ============================================================
# QA Prompt (Simpler, Direct)
# ============================================================

QA_SYSTEM_PROMPT = """You are Lilly-X, a precision RAG assistant.

CITATION RULE: You are a strict analyst. Every claim you make MUST be backed by the provided context. When referencing information, cite the filename explicitly (e.g., [Source: report_2024.pdf]). If the context is empty or irrelevant, state 'I cannot answer this based on the available documents'.

RULES:
1. Answer ONLY from the Context provided
2. CITE sources explicitly using [Source: filename] format
3. NEVER say "the context states" - use specific filenames/dates/pages
4. If answer not in Context: say "I cannot answer this based on the available documents"
"""

def build_qa_prompt(
    user_query: str,
    context_str: str,
    conversation_history: str = ""
) -> str:
    """
    Build a simple QA prompt with context string for direct LLM completion.
    
    This is used when calling LLM.complete() directly rather than using
    the query engine's built-in prompting.
    
    Args:
        user_query: User's question
        context_str: Pre-formatted context string with sources
        conversation_history: Optional conversation history
        
    Returns:
        Complete prompt string ready for LLM.complete()
    """
    parts = [QA_SYSTEM_PROMPT]
    
    if conversation_history:
        parts.append(f"\n## Conversation History\n{conversation_history}\n")
    
    parts.append(f"\n## Context\n{context_str}\n")
    parts.append(f"\n## Question\n{user_query}\n")
    parts.append("\n## Answer\n")
    
    return "\n".join(parts)
