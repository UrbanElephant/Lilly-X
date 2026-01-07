"""Memory management for multi-turn conversations in Lilly-X."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

from src.config import settings
from src.memory_schema import ChatSession, ChatMessage


class MemoryManager:
    """
    Manages conversation history with FIFO trimming based on memory window size.
    
    Provides thread-safe operations for storing and retrieving chat sessions.
    Supports both in-memory storage and JSON file persistence.
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        use_persistence: bool = False
    ) -> None:
        """
        Initialize the MemoryManager.
        
        Args:
            storage_path: Path to JSON file for persistent storage.
                         If None, defaults to ./data/memory/sessions.json
            use_persistence: Whether to persist sessions to disk (default: False for in-memory only)
        """
        self._sessions: Dict[UUID, ChatSession] = {}
        self._lock = threading.Lock()
        self._use_persistence = use_persistence
        
        # Set up storage path
        if storage_path is None:
            self._storage_path = Path("./data/memory/sessions.json")
        else:
            self._storage_path = storage_path
        
        # Load existing sessions if persistence is enabled
        if self._use_persistence:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def get_history(self, session_id: UUID) -> ChatSession:
        """
        Retrieve conversation history for a given session.
        
        Creates a new session if one doesn't exist.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ChatSession object containing message history
        """
        with self._lock:
            if session_id not in self._sessions:
                # Create new session
                self._sessions[session_id] = ChatSession(session_id=session_id)
            
            return self._sessions[session_id]
    
    def add_message(
        self,
        session_id: UUID,
        role: str,
        content: str
    ) -> None:
        """
        Add a message to a session and trim history if needed.
        
        Args:
            session_id: Unique session identifier
            role: Message role ('user', 'assistant', or 'system')
            content: Message content/text
            
        Raises:
            ValueError: If role is not valid
        """
        if role not in ["user", "assistant", "system"]:
            raise ValueError(
                f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'"
            )
        
        with self._lock:
            # Get or create session
            if session_id not in self._sessions:
                self._sessions[session_id] = ChatSession(session_id=session_id)
            
            session = self._sessions[session_id]
            
            # Add message
            session.add_message(role=role, content=content)  # type: ignore
            
            # Trim history based on memory window size
            self._trim_history(session)
            
            # Persist if enabled
            if self._use_persistence:
                self._save_to_disk()
    
    def _trim_history(self, session: ChatSession) -> None:
        """
        Implement FIFO trimming based on memory window size.
        
        Keeps only the most recent N messages as defined by
        settings.memory_window_size.
        
        Args:
            session: ChatSession to trim
        """
        max_messages = settings.memory_window_size
        
        if len(session.messages) > max_messages:
            # Keep only the last N messages (FIFO - remove oldest)
            session.messages = session.messages[-max_messages:]
    
    def clear_session(self, session_id: UUID) -> None:
        """
        Clear all messages for a specific session.
        
        Args:
            session_id: Session to clear
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear_history()
                
                if self._use_persistence:
                    self._save_to_disk()
    
    def delete_session(self, session_id: UUID) -> bool:
        """
        Delete a session completely.
        
        Args:
            session_id: Session to delete
            
        Returns:
            True if session was deleted, False if it didn't exist
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                
                if self._use_persistence:
                    self._save_to_disk()
                
                return True
            return False
    
    def list_sessions(self) -> Dict[UUID, datetime]:
        """
        Get all active sessions with their last activity time.
        
        Returns:
            Dictionary mapping session_id to last_active timestamp
        """
        with self._lock:
            return {
                session_id: session.last_active
                for session_id, session in self._sessions.items()
            }
    
    def _save_to_disk(self) -> None:
        """
        Persist all sessions to disk as JSON.
        
        Note: This is a simple implementation. For production use,
        consider using a proper database (PostgreSQL, Redis, etc.)
        """
        try:
            serialized_sessions = {
                str(session_id): session.model_dump(mode='json')
                for session_id, session in self._sessions.items()
            }
            
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(serialized_sessions, f, indent=2, default=str)
        
        except Exception as e:
            print(f"⚠ Warning: Failed to save sessions to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """
        Load sessions from JSON file on disk.
        
        If file doesn't exist or is corrupted, starts with empty sessions.
        """
        if not self._storage_path.exists():
            return
        
        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Deserialize sessions
            for session_id_str, session_data in data.items():
                session_id = UUID(session_id_str)
                self._sessions[session_id] = ChatSession(**session_data)
            
            print(f"✓ Loaded {len(self._sessions)} sessions from {self._storage_path}")
        
        except Exception as e:
            print(f"⚠ Warning: Failed to load sessions from disk: {e}")
            print("  Starting with empty session storage.")


# ============================================================
# TODO: Thread Safety Enhancements
# ============================================================
# 
# Current implementation uses basic threading.Lock for thread safety.
# 
# For production environments, consider:
# 1. Read-write locks (threading.RLock) for better concurrent read performance
# 2. Async support using asyncio.Lock for async operations
# 3. Distributed locking (Redis) for multi-process deployments
# 4. Connection pooling for persistent storage backends
# 5. Batch write operations to reduce I/O overhead
#
# ============================================================
