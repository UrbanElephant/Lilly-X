"""Pydantic schemas for Query Planning and Reasoning-GraphRAG."""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field



# ============================================================
# Community Detection Models
# ============================================================

class CommunitySummary(BaseModel):
    """Represents a community of entities and its summary.
    
    Communities are detected using graph algorithms (Leiden, Louvain) and
    summarized to support global search queries in Microsoft-style GraphRAG.
    """
    
    community_id: int = Field(
        ...,
        description="Unique identifier for this community"
    )
    
    level: int = Field(
        ...,
        description="Hierarchical level of this community (0 = leaf, higher = more abstract)"
    )
    
    summary: str = Field(
        ...,
        description="Natural language summary of what this community represents"
    )
    
    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms and concepts associated with this community"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "community_id": 1,
                "level": 0,
                "summary": "This community focuses on machine learning frameworks and neural networks",
                "keywords": ["tensorflow", "pytorch", "neural networks", "deep learning"]
            }
        }


# ============================================================
# Query Intent Enumeration
# ============================================================

class QueryIntent(str, Enum):
    """Types of query intent for sub-query classification."""
    
    FACTUAL = "factual"
    WORKFLOW = "workflow"
    COMPARISON = "comparison"
    GLOBAL_DISCOVERY = "global_discovery"  # For abstract, high-level questions


# ============================================================
# Query Planning Models
# ============================================================

class SubQuery(BaseModel):
    """Represents a focused sub-query decomposed from a complex user question.
    
    The QueryPlanner breaks down complex queries into atomic sub-queries
    that can be independently processed and then synthesized.
    """
    
    original_text: str = Field(
        ...,
        description="The original portion of text from the user's query that this sub-query addresses"
    )
    
    focused_query: str = Field(
        ...,
        description="A focused, self-contained query optimized for retrieval (may be rephrased for clarity)"
    )
    
    intent: QueryIntent = Field(
        ...,
        description="Classification of the sub-query intent: factual, workflow, or comparison"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "original_text": "how does authentication work",
                "focused_query": "Explain the authentication workflow and mechanisms",
                "intent": "workflow"
            }
        }


class QueryPlan(BaseModel):
    """Represents a complete query decomposition plan for a complex user query.
    
    The QueryPlan contains the original root query and a list of decomposed
    sub-queries that will be executed independently and then synthesized.
    """
    
    root_query: str = Field(
        ...,
        description="The original complete user query before decomposition"
    )
    
    sub_queries: List[SubQuery] = Field(
        default_factory=list,
        description="List of decomposed sub-queries, ordered by execution priority"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "root_query": "How does authentication work and how does it compare to authorization?",
                "sub_queries": [
                    {
                        "original_text": "how does authentication work",
                        "focused_query": "Explain the authentication workflow and mechanisms",
                        "intent": "workflow"
                    },
                    {
                        "original_text": "how does it compare to authorization",
                        "focused_query": "Compare authentication versus authorization",
                        "intent": "comparison"
                    }
                ]
            }
        }
