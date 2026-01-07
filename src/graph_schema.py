"""Knowledge graph schema definitions for Lilly-X hybrid GraphRAG."""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ============================================================
# Relationship Types (Strict Typing)
# ============================================================

RelationshipType = Literal[
    "WORKS_FOR",       # Person -> Organization
    "MEMBER_OF",       # Person -> Organization
    "AUTHORED",        # Person -> Document
    "MENTIONS",        # Document -> Entity (any)
    "RELATES_TO",      # Concept -> Concept
    "OCCURRED_AT",     # Event -> Location/Organization
    "PARTICIPATED_IN", # Person -> Event
    "PART_OF",         # Organization -> Organization (hierarchy)
    "REFERENCES",      # Document -> Document
    "DEFINED_IN",      # Concept -> Document
]


# ============================================================
# Entity Base Class
# ============================================================

class GraphEntity(BaseModel):
    """Base class for all graph entities with disambiguation support."""
    
    name: str = Field(..., description="Primary name/identifier for the entity")
    entity_type: str = Field(..., description="Type of entity (Person, Organization, etc.)")
    description: Optional[str] = Field(None, description="Optional description or summary")
    source_documents: List[str] = Field(
        default_factory=list,
        description="List of document IDs where this entity was mentioned"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for entity extraction (0-1)"
    )
    
    # ========== Disambiguation Fields ==========
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names, abbreviations, or nicknames for this entity"
    )
    canonical_name: Optional[str] = Field(
        None,
        description="Canonical/main entity name if this is an alias or variant (for entity resolution)"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for entity resolution/disambiguation (0-1)"
    )
    # ==========================================
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Core Entity Classes
# ============================================================

class Person(GraphEntity):
    """Represents a person mentioned in documents."""
    
    entity_type: Literal["Person"] = "Person"
    role: Optional[str] = Field(None, description="Primary role or title")
    affiliations: List[str] = Field(
        default_factory=list,
        description="Organizations or groups the person is affiliated with"
    )
    # Note: aliases inherited from GraphEntity base class


class Organization(GraphEntity):
    """Represents an organization, company, or institution."""
    
    entity_type: Literal["Organization"] = "Organization"
    org_type: Optional[str] = Field(
        None,
        description="Type of organization (e.g., 'Company', 'University', 'Government')"
    )
    industry: Optional[str] = Field(None, description="Industry or sector")
    parent_organization: Optional[str] = Field(
        None,
        description="Parent organization if this is a subsidiary or department"
    )
    # Note: aliases inherited from GraphEntity base class


class Event(GraphEntity):
    """Represents a significant event or occurrence."""
    
    entity_type: Literal["Event"] = "Event"
    event_date: Optional[str] = Field(
        None,
        description="Date or time period when the event occurred (flexible format)"
    )
    location: Optional[str] = Field(None, description="Where the event took place")
    participants: List[str] = Field(
        default_factory=list,
        description="People or organizations involved"
    )
    event_type: Optional[str] = Field(
        None,
        description="Category of event (e.g., 'Conference', 'Product Launch', 'Meeting')"
    )


class Document(GraphEntity):
    """Represents a document in the knowledge base."""
    
    entity_type: Literal["Document"] = "Document"
    title: str = Field(..., description="Document title")
    doc_type: Optional[str] = Field(
        None,
        description="Type of document (e.g., 'Technical Manual', 'Research Paper', 'Email')"
    )
    author: Optional[str] = Field(None, description="Primary author")
    source_path: Optional[str] = Field(None, description="Original file path or URL")
    publication_date: Optional[str] = Field(None, description="When the document was published")
    vector_id: Optional[str] = Field(
        None,
        description="ID in the Qdrant vector store for dual-store linkage"
    )


class Concept(GraphEntity):
    """Represents an abstract concept, topic, or technical term."""
    
    entity_type: Literal["Concept"] = "Concept"
    definition: Optional[str] = Field(None, description="Definition of the concept")
    category: Optional[str] = Field(
        None,
        description="Category or domain (e.g., 'Machine Learning', 'Physics', 'Business')"
    )
    related_concepts: List[str] = Field(
        default_factory=list,
        description="Names of related concepts"
    )
    synonyms: List[str] = Field(default_factory=list, description="Alternative terms")


# ============================================================
# Relationship Class
# ============================================================

class Relationship(BaseModel):
    """Represents a relationship between two entities."""
    
    source_entity: str = Field(..., description="Name of the source entity")
    source_type: str = Field(..., description="Type of the source entity")
    target_entity: str = Field(..., description="Name of the target entity")
    target_type: str = Field(..., description="Type of the target entity")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    properties: dict = Field(
        default_factory=dict,
        description="Additional properties for this relationship"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for relationship extraction (0-1)"
    )
    source_document: Optional[str] = Field(
        None,
        description="Document ID where this relationship was identified"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Batch Update Model
# ============================================================

class KnowledgeGraphUpdate(BaseModel):
    """Batch update for knowledge graph containing entities and relationships."""
    
    entities: List[GraphEntity] = Field(
        default_factory=list,
        description="List of entities to add or update"
    )
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of relationships to create"
    )
    document_id: str = Field(..., description="ID of the document being processed")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "entities": [
                    {
                        "name": "John Doe",
                        "entity_type": "Person",
                        "role": "Software Engineer",
                        "affiliations": ["Tech Corp"],
                        "confidence": 0.95
                    }
                ],
                "relationships": [
                    {
                        "source_entity": "John Doe",
                        "source_type": "Person",
                        "target_entity": "Tech Corp",
                        "target_type": "Organization",
                        "relationship_type": "WORKS_FOR",
                        "confidence": 0.9
                    }
                ]
            }
        }


# ============================================================
# Helper Functions
# ============================================================

def get_entity_class(entity_type: str) -> type[GraphEntity]:
    """
    Get the appropriate entity class based on type string.
    
    Args:
        entity_type: String representation of entity type
        
    Returns:
        Corresponding entity class
        
    Raises:
        ValueError: If entity type is not recognized
    """
    entity_map = {
        "Person": Person,
        "Organization": Organization,
        "Event": Event,
        "Document": Document,
        "Concept": Concept,
    }
    
    if entity_type not in entity_map:
        raise ValueError(
            f"Unknown entity type: {entity_type}. "
            f"Valid types: {list(entity_map.keys())}"
        )
    
    return entity_map[entity_type]


def validate_relationship(relationship: Relationship) -> bool:
    """
    Validate that a relationship makes semantic sense.
    
    Args:
        relationship: Relationship to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Define valid relationship patterns
    valid_patterns = {
        "WORKS_FOR": [("Person", "Organization")],
        "MEMBER_OF": [("Person", "Organization")],
        "AUTHORED": [("Person", "Document")],
        "MENTIONS": [
            ("Document", "Person"),
            ("Document", "Organization"),
            ("Document", "Event"),
            ("Document", "Concept"),
        ],
        "RELATES_TO": [("Concept", "Concept")],
        "OCCURRED_AT": [("Event", "Organization")],
        "PARTICIPATED_IN": [("Person", "Event")],
        "PART_OF": [("Organization", "Organization")],
        "REFERENCES": [("Document", "Document")],
        "DEFINED_IN": [("Concept", "Document")],
    }
    
    rel_type = relationship.relationship_type
    source_type = relationship.source_type
    target_type = relationship.target_type
    
    if rel_type not in valid_patterns:
        return False
    
    return (source_type, target_type) in valid_patterns[rel_type]
