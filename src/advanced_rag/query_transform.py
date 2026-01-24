"""Query Transformation Module.

Implements advanced query transformation techniques:
- QueryDecomposer: Breaks complex queries into sub-questions
- HyDEGenerator: Generates hypothetical document embeddings
- QueryRewriter: Expands queries with synonyms and reformulations
"""

from typing import List, Optional

from llama_index.core import Settings
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate


class QueryDecomposer:
    """Decomposes complex queries into simpler sub-questions.
    
    Uses the LLM to break down multi-faceted questions into atomic sub-queries
    that can be answered independently and then synthesized.
    """
    
    DECOMPOSE_PROMPT = PromptTemplate(
        """You are an expert at breaking down complex questions into simpler sub-questions.

Given a complex query, decompose it into 2-4 atomic sub-questions that when answered together 
would provide a comprehensive answer to the original query.

Complex Query: {query}

Return ONLY a JSON array of sub-questions, no other text:
["sub-question 1", "sub-question 2", ...]"""
    )
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize the query decomposer.
        
        Args:
            llm: Language model to use for decomposition. Defaults to Settings.llm
        """
        self.llm = llm or Settings.llm
        
    def decompose(self, query: str, max_subqueries: int = 3) -> List[str]:
        """Decompose a complex query into sub-questions.
        
        Args:
            query: The complex query to decompose
            max_subqueries: Maximum number of sub-queries to generate
            
        Returns:
            List of sub-questions
        """
        prompt = self.DECOMPOSE_PROMPT.format(query=query)
        
        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Try to parse JSON array
            import json
            try:
                from json_repair import repair_json
                repaired = repair_json(response_text)
                sub_queries = json.loads(repaired)
            except ImportError:
                # Fallback without json_repair
                import re
                # Extract JSON array
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    sub_queries = json.loads(match.group(0))
                else:
                    # Fallback: split by newlines
                    sub_queries = [line.strip(' -•"\'') for line in response_text.split('\n') 
                                 if line.strip() and not line.strip().startswith('[')]
            
            # Ensure we return a list
            if not isinstance(sub_queries, list):
                sub_queries = [query]
            
            # Limit to max_subqueries
            sub_queries = sub_queries[:max_subqueries]
            
            # If no sub-queries were generated, return original query
            if not sub_queries:
                sub_queries = [query]
                
            return sub_queries
            
        except Exception as e:
            print(f"[QueryDecomposer] Error during decomposition: {e}")
            # Fallback: return original query
            return [query]


class HyDEGenerator:
    """Generates Hypothetical Document Embeddings (HyDE).
    
    Creates hypothetical answers to improve retrieval by searching for documents
    similar to what the answer might look like, rather than the question.
    """
    
    HYDE_PROMPT = PromptTemplate(
        """You are an expert technical writer. 

Given a question, write a detailed, hypothetical answer that would perfectly address this question.
This hypothetical answer will be used to find similar real documents, so make it technical and specific.

Question: {query}

Write a comprehensive hypothetical answer (2-3 paragraphs):"""
    )
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize the HyDE generator.
        
        Args:
            llm: Language model to use for generation. Defaults to Settings.llm
        """
        self.llm = llm or Settings.llm
        
    def generate(self, query: str) -> str:
        """Generate a hypothetical document for the query.
        
        Args:
            query: The user's query
            
        Returns:
            Hypothetical document text
        """
        prompt = self.HYDE_PROMPT.format(query=query)
        
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[HyDEGenerator] Error during generation: {e}")
            # Fallback: return original query
            return query


class QueryRewriter:
    """Rewrites queries with synonyms and reformulations.
    
    Expands the query vocabulary to improve recall by generating alternative
    phrasings and terminology variations.
    """
    
    REWRITE_PROMPT = PromptTemplate(
        """You are an expert at query expansion and reformulation.

Given a query, generate 2-3 alternative phrasings using:
- Synonyms and related terms
- Technical and casual language variations
- Different question structures

Original Query: {query}

Return ONLY a JSON array of alternative queries:
["alternative 1", "alternative 2", ...]"""
    )
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize the query rewriter.
        
        Args:
            llm: Language model to use for rewriting. Defaults to Settings.llm
        """
        self.llm = llm or Settings.llm
        
    def rewrite(self, query: str, include_original: bool = True) -> List[str]:
        """Rewrite query with variations.
        
        Args:
            query: Original query
            include_original: Whether to include the original query in results
            
        Returns:
            List of query variations
        """
        prompt = self.REWRITE_PROMPT.format(query=query)
        
        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Try to parse JSON array
            import json
            try:
                from json_repair import repair_json
                repaired = repair_json(response_text)
                variations = json.loads(repaired)
            except ImportError:
                # Fallback without json_repair
                import re
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    variations = json.loads(match.group(0))
                else:
                    variations = [line.strip(' -•"\'') for line in response_text.split('\n') 
                                if line.strip() and not line.strip().startswith('[')]
            
            if not isinstance(variations, list):
                variations = []
            
            # Add original query if requested
            if include_original:
                variations.insert(0, query)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_variations = []
            for v in variations:
                if v and v not in seen:
                    seen.add(v)
                    unique_variations.append(v)
            
            return unique_variations if unique_variations else [query]
            
        except Exception as e:
            print(f"[QueryRewriter] Error during rewriting: {e}")
            return [query]


# Self-contained testing
if __name__ == "__main__":
    """Quick testing of query transformation components."""
    
    print("=" * 80)
    print("Query Transformation Module - Self Test")
    print("=" * 80)
    
    # Test queries
    test_queries = [
        "What are the key differences between RAG and fine-tuning for LLMs?",
        "Explain vector databases",
    ]
    
    try:
        # Initialize components (requires Settings.llm to be configured)
        from llama_index.core import Settings
        from llama_index.llms.ollama import Ollama
        
        # Try to use configured LLM or create a mock
        if Settings.llm is None:
            print("\n⚠️  Settings.llm not configured, using Ollama default")
            Settings.llm = Ollama(model="mistral-nemo:12b", base_url="http://localhost:11434", request_timeout=30.0)
        
        decomposer = QueryDecomposer()
        hyde_gen = HyDEGenerator()
        rewriter = QueryRewriter()
        
        for query in test_queries:
            print(f"\n{'─' * 80}")
            print(f"Query: {query}")
            print(f"{'─' * 80}")
            
            # Test decomposition
            print("\n1. Query Decomposition:")
            sub_queries = decomposer.decompose(query, max_subqueries=3)
            for i, sq in enumerate(sub_queries, 1):
                print(f"   {i}. {sq}")
            
            # Test HyDE
            print("\n2. HyDE Generation:")
            hyde_doc = hyde_gen.generate(query)
            print(f"   {hyde_doc[:200]}...")
            
            # Test rewriting
            print("\n3. Query Rewriting:")
            variations = rewriter.rewrite(query, include_original=False)
            for i, var in enumerate(variations, 1):
                print(f"   {i}. {var}")
        
        print(f"\n{'=' * 80}")
        print("✅ All tests completed successfully")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
