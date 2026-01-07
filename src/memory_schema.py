"""Multi-turn conversation memory schema for Lilly-X."""

from datetime import datetime
from typing import List, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ============================================================
# Chat Message Model
# ============================================================

class ChatMessage(BaseModel):
    """Represents a single message in a conversation."""
    
    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Role of the message sender"
    )
    content: str = Field(
        ...,
        description="Message content/text"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the message was created"
    )
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What are the key features of GraphRAG?",
                "timestamp": "2026-01-07T04:47:59Z"
            }
        }


# ============================================================
# Chat Session Model
# ============================================================

class ChatSession(BaseModel):
    """Represents a complete chat session with message history."""
    
    session_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this session"
    )
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Chronologically ordered list of messages"
    )
    last_active: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last time this session was accessed or modified"
    )
    
    def to_string(self, max_turns: int | None = None) -> str:
        """
        Format conversation history as a string for LLM context window.
        
        Args:
            max_turns: Maximum number of recent turns to include (None = all)
            
        Returns:
            Formatted string representation of the conversation
        """
        if not self.messages:
            return ""
        
        # Get recent messages if max_turns is specified
        messages_to_format = self.messages
        if max_turns is not None and max_turns > 0:
            messages_to_format = self.messages[-max_turns:]
        
        # Format each message with role prefix
        formatted_lines: List[str] = []
        for msg in messages_to_format:
            role_prefix = msg.role.upper()
            formatted_lines.append(f"{role_prefix}: {msg.content}")
        
        return "\n\n".join(formatted_lines)
    
    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str
    ) -> None:
        """
        Add a new message to the session and update last_active timestamp.
        
        Args:
            role: Role of the message sender
            content: Message content
        """
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.last_active = datetime.utcnow()
    
    def get_recent_turns(self, n: int) -> List[ChatMessage]:
        """
        Get the N most recent messages.
        
        Args:
            n: Number of recent messages to retrieve
            
        Returns:
            List of recent ChatMessage objects
        """
        if n <= 0:
            return []
        return self.messages[-n:]
    
    def clear_history(self) -> None:
        """Clear all messages from the session."""
        self.messages = []
        self.last_active = datetime.utcnow()
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me about GraphRAG",
                        "timestamp": "2026-01-07T04:45:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "GraphRAG combines knowledge graphs with retrieval...",
                        "timestamp": "2026-01-07T04:45:05Z"
                    }
                ],
                "last_active": "2026-01-07T04:45:05Z"
            }
        }
