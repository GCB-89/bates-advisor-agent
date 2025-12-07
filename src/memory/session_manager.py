"""
Session & Memory Management

This module handles:
- Session state management
- Student context memory (major, interests, goals)
- Conversation history
- Persistent memory storage
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class StudentContext:
    """Student context that persists across conversations."""
    major: Optional[str] = None
    year: Optional[str] = None
    interests: List[str] = None
    goals: List[str] = None
    previous_questions: List[str] = None
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.goals is None:
            self.goals = []
        if self.previous_questions is None:
            self.previous_questions = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_context_string(self) -> str:
        """Convert student context to a readable string for agent prompts."""
        parts = []
        
        if self.major:
            parts.append(f"Major: {self.major}")
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests)}")
        if self.goals:
            parts.append(f"Goals: {', '.join(self.goals)}")
        
        return " | ".join(parts) if parts else "No prior context"


class ConversationHistory:
    """Manages conversation history for a session."""
    
    def __init__(self, max_history: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_history * 2:  # *2 for user+assistant pairs
            self.messages = self.messages[-self.max_history * 2:]
    
    def get_recent_context(self, n: int = 5) -> str:
        """Get recent conversation as context string."""
        recent = self.messages[-n*2:] if len(self.messages) >= n*2 else self.messages
        
        context_lines = []
        for msg in recent:
            role = "Student" if msg["role"] == "user" else "Advisor"
            context_lines.append(f"{role}: {msg['content'][:100]}...")
        
        return "\n".join(context_lines) if context_lines else "No prior conversation"
    
    def get_all_messages(self) -> List[Dict]:
        """Get all conversation messages."""
        return self.messages.copy()


class SessionManager:
    """
    Manages sessions and student memory.
    
    Features:
    - Session state tracking
    - Student context memory
    - Conversation history
    - Persistent storage
    """
    
    def __init__(self, storage_dir: str = "memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Active sessions
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            session_id: The created session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "student_context": StudentContext(),
            "conversation_history": ConversationHistory(),
            "metadata": {}
        }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        return self.sessions.get(session_id)
    
    def update_student_context(
        self, 
        session_id: str, 
        major: Optional[str] = None,
        year: Optional[str] = None,
        interests: Optional[List[str]] = None,
        goals: Optional[List[str]] = None
    ):
        """
        Update student context for a session.
        
        This is called when we learn new information about the student.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        context = session["student_context"]
        
        if major:
            context.major = major
        if year:
            context.year = year
        if interests:
            context.interests.extend(interests)
            context.interests = list(set(context.interests))  # Remove duplicates
        if goals:
            context.goals.extend(goals)
            context.goals = list(set(context.goals))
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session["conversation_history"].add_message(role, content)
        
        # Track questions in student context
        if role == "user":
            session["student_context"].previous_questions.append(content)
            # Keep only last 20 questions
            if len(session["student_context"].previous_questions) > 20:
                session["student_context"].previous_questions = \
                    session["student_context"].previous_questions[-20:]
    
    def get_context_for_agent(self, session_id: str) -> Dict[str, str]:
        """
        Get formatted context for agent prompts.
        
        Returns:
            Dict with student_context and conversation_context strings
        """
        session = self.get_session(session_id)
        if not session:
            return {
                "student_context": "No prior context",
                "conversation_context": "No prior conversation"
            }
        
        return {
            "student_context": session["student_context"].to_context_string(),
            "conversation_context": session["conversation_history"].get_recent_context()
        }
    
    def extract_context_from_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Extract student context hints from the query.
        
        Example: "I'm a nursing student..." → major = "Nursing"
        """
        query_lower = query.lower()
        extracted = {}
        
        # Detect major mentions
        programs = [
            "nursing", "carpentry", "welding", "dental", "healthcare",
            "medical", "construction", "automotive", "culinary"
        ]
        for program in programs:
            if program in query_lower:
                extracted["major"] = program.title()
                break
        
        # Detect year mentions
        years = ["freshman", "sophomore", "junior", "senior", "first year", "second year"]
        for year in years:
            if year in query_lower:
                extracted["year"] = year.title()
                break
        
        # Update context if we found anything
        if extracted:
            self.update_student_context(session_id, **extracted)
        
        return extracted
    
    def save_session(self, session_id: str):
        """Persist session to disk."""
        session = self.get_session(session_id)
        if not session:
            return
        
        # Convert to serializable format
        session_data = {
            "session_id": session_id,
            "created_at": session["created_at"],
            "student_context": session["student_context"].to_dict(),
            "conversation_history": session["conversation_history"].get_all_messages(),
            "metadata": session["metadata"]
        }
        
        file_path = self.storage_dir / f"session_{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def load_session(self, session_id: str) -> bool:
        """Load session from disk."""
        file_path = self.storage_dir / f"session_{session_id}.json"
        
        if not file_path.exists():
            return False
        
        with open(file_path, 'r') as f:
            session_data = json.load(f)
        
        # Reconstruct session objects
        context = StudentContext(**session_data["student_context"])
        history = ConversationHistory()
        for msg in session_data["conversation_history"]:
            history.add_message(msg["role"], msg["content"])
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "created_at": session_data["created_at"],
            "student_context": context,
            "conversation_history": history,
            "metadata": session_data.get("metadata", {})
        }
        
        return True
    
    def list_sessions(self) -> List[str]:
        """List all saved sessions."""
        return [f.stem.replace("session_", "") 
                for f in self.storage_dir.glob("session_*.json")]


# Example usage
if __name__ == "__main__":
    # Create session manager
    manager = SessionManager()
    
    # Create a session
    session_id = manager.create_session()
    print(f"Created session: {session_id}")
    
    # Simulate conversation
    manager.add_message(session_id, "user", "I'm a nursing student interested in healthcare programs")
    manager.extract_context_from_query(session_id, "I'm a nursing student interested in healthcare programs")
    
    # Get context
    context = manager.get_context_for_agent(session_id)
    print(f"\nStudent Context: {context['student_context']}")
    print(f"Conversation Context: {context['conversation_context']}")
    
    # Save session
    manager.save_session(session_id)
    print(f"\nSession saved to disk")
