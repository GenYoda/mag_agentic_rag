"""
================================================================================
MEMORY TOOLS - Phase 3 Tool 8/9
================================================================================
Purpose: Manage conversation history and entity memory for contextual responses
Features:
- Conversation history tracking
- Entity memory (patients, doctors, dates, etc.)
- Context window management
- Follow-up question handling
- Memory persistence
================================================================================
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import re

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ConversationTurn:
    """Single conversation turn (Q&A pair)"""
    turn_id: int
    timestamp: datetime
    user_query: str
    assistant_response: str
    entities_mentioned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationTurn':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Entity:
    """Entity tracked across conversation"""
    entity_id: str
    entity_type: str  # person, organization, date, location, etc.
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    mentions: List[int] = field(default_factory=list)  # Turn IDs where mentioned
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['first_seen'] = self.first_seen.isoformat()
        data['last_seen'] = self.last_seen.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        """Create from dictionary"""
        data['first_seen'] = datetime.fromisoformat(data['first_seen'])
        data['last_seen'] = datetime.fromisoformat(data['last_seen'])
        return cls(**data)


# ============================================================================
# Memory Tools Class
# ============================================================================

class MemoryTools:
    """
    Manages conversation history and entity memory for contextual responses
    """
    
    def __init__(
        self,
        session_id: str = "default",
        max_history_turns: int = 20,
        memory_dir: str = "./data/memory",
        enable_persistence: bool = True,
        enable_entity_tracking: bool = True
    ):
        """
        Initialize MemoryTools
        
        Args:
            session_id: Unique identifier for this conversation session
            max_history_turns: Maximum conversation turns to keep in memory
            memory_dir: Directory to store persistent memory
            enable_persistence: Whether to save/load memory from disk
            enable_entity_tracking: Whether to track entities
        """
        self.session_id = session_id
        self.max_history_turns = max_history_turns
        self.memory_dir = Path(memory_dir)
        self.enable_persistence = enable_persistence
        self.enable_entity_tracking = enable_entity_tracking
        
        # Memory storage
        self.conversation_history: List[ConversationTurn] = []
        self.entities: Dict[str, Entity] = {}
        self.turn_counter = 0
        self.session_start = datetime.now()
        
        # Create memory directory
        if self.enable_persistence:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self._load_session()
        
        logger.info(f"MemoryTools initialized (session: {session_id}, "
                   f"entity_tracking: {enable_entity_tracking})")
    
    
    # ========================================================================
    # Core Memory Operations
    # ========================================================================
    
    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a conversation turn to memory
        
        Args:
            user_query: User's question
            assistant_response: Assistant's answer
            metadata: Additional metadata about this turn
            
        Returns:
            ConversationTurn object
        """
        self.turn_counter += 1
        
        # Create conversation turn
        turn = ConversationTurn(
            turn_id=self.turn_counter,
            timestamp=datetime.now(),
            user_query=user_query,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )
        
        # Extract and track entities if enabled
        if self.enable_entity_tracking:
            entities = self._extract_entities(user_query, assistant_response)
            turn.entities_mentioned = [e.entity_id for e in entities]
            
            for entity in entities:
                self._update_entity(entity, self.turn_counter)
        
        # Add to history
        self.conversation_history.append(turn)
        
        # Maintain max history size
        if len(self.conversation_history) > self.max_history_turns:
            removed = self.conversation_history.pop(0)
            logger.debug(f"Removed old turn {removed.turn_id} from history")
        
        # Persist if enabled
        if self.enable_persistence:
            self._save_session()
        
        logger.info(f"Added turn {self.turn_counter} to memory "
                   f"(entities: {len(turn.entities_mentioned)})")
        
        return turn
    
    
    def get_conversation_context(
        self,
        max_turns: Optional[int] = None,
        include_metadata: bool = False
    ) -> str:
        """
        Get formatted conversation history for LLM context
        
        Args:
            max_turns: Maximum recent turns to include (default: all in memory)
            include_metadata: Whether to include turn metadata
            
        Returns:
            Formatted conversation history string
        """
        if not self.conversation_history:
            return "No conversation history available."
        
        # Get recent turns
        turns_to_include = max_turns or len(self.conversation_history)
        recent_turns = self.conversation_history[-turns_to_include:]
        
        # Format conversation
        context_parts = ["=== Conversation History ===\n"]
        
        for turn in recent_turns:
            context_parts.append(f"[Turn {turn.turn_id}]")
            context_parts.append(f"User: {turn.user_query}")
            context_parts.append(f"Assistant: {turn.assistant_response[:500]}...")  # Truncate long responses
            
            if include_metadata and turn.metadata:
                context_parts.append(f"Metadata: {turn.metadata}")
            
            context_parts.append("")  # Blank line
        
        return "\n".join(context_parts)
    
    
    def get_relevant_context(
        self,
        current_query: str,
        max_turns: int = 5
    ) -> str:
        """
        Get relevant conversation context based on current query
        
        Args:
            current_query: Current user query
            max_turns: Maximum turns to return
            
        Returns:
            Relevant conversation history
        """
        if not self.conversation_history:
            return ""
        
        # Simple relevance: check for entity mentions and keyword overlap
        query_lower = current_query.lower()
        scored_turns = []
        
        for turn in self.conversation_history:
            score = 0
            
            # Check for shared entities
            for entity_id in turn.entities_mentioned:
                if entity_id in query_lower:
                    score += 5
            
            # Check for keyword overlap
            turn_text = f"{turn.user_query} {turn.assistant_response}".lower()
            query_words = set(query_lower.split())
            turn_words = set(turn_text.split())
            overlap = len(query_words & turn_words)
            score += overlap
            
            scored_turns.append((score, turn))
        
        # Get top relevant turns
        scored_turns.sort(reverse=True, key=lambda x: x[0])
        relevant_turns = [turn for score, turn in scored_turns[:max_turns] if score > 0]
        
        if not relevant_turns:
            # Return most recent turns if no relevant ones
            relevant_turns = self.conversation_history[-max_turns:]
        
        # Format context
        context_parts = ["=== Relevant Previous Context ===\n"]
        for turn in relevant_turns:
            context_parts.append(f"Previous Q: {turn.user_query}")
            context_parts.append(f"Previous A: {turn.assistant_response[:300]}...\n")
        
        return "\n".join(context_parts)
    
    
    # ========================================================================
    # Entity Memory Operations
    # ========================================================================
    
    def _extract_entities(
        self,
        user_query: str,
        assistant_response: str
    ) -> List[Entity]:
        """
        Extract entities from conversation turn
        (Simple pattern-based extraction for demo)
        
        Args:
            user_query: User's question
            assistant_response: Assistant's answer
            
        Returns:
            List of Entity objects
        """
        entities = []
        text = f"{user_query} {assistant_response}"
        
        # Pattern 1: Proper names (capitalized words)
        name_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b'
        names = re.findall(name_pattern, text)
        
        for name in set(names):
            if len(name.split()) >= 2:  # Multi-word names likely to be people/orgs
                entity_id = name.lower().replace(' ', '_')
                entities.append(Entity(
                    entity_id=entity_id,
                    entity_type='person_or_org',
                    name=name
                ))
        
        # Pattern 2: Dates
        date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        for date in set(dates):
            entity_id = f"date_{date.replace(' ', '_').replace(',', '')}"
            entities.append(Entity(
                entity_id=entity_id,
                entity_type='date',
                name=date,
                attributes={'raw_date': date}
            ))
        
        # Pattern 3: Medical/Legal terms (in uppercase or with specific patterns)
        medical_terms = ['Memorial Health', 'Savannah Health', 'Nurse', 'Doctor', 'Dr.', 'RN']
        for term in medical_terms:
            if term in text:
                entity_id = term.lower().replace(' ', '_')
                entities.append(Entity(
                    entity_id=entity_id,
                    entity_type='organization' if 'Health' in term else 'role',
                    name=term
                ))
        
        return entities
    
    
    def _update_entity(self, entity: Entity, turn_id: int):
        """Update or add entity to memory"""
        if entity.entity_id in self.entities:
            # Update existing entity
            existing = self.entities[entity.entity_id]
            existing.mentions.append(turn_id)
            existing.last_seen = datetime.now()
            existing.attributes.update(entity.attributes)
        else:
            # Add new entity
            entity.mentions.append(turn_id)
            self.entities[entity.entity_id] = entity
    
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    
    def get_all_entities(self) -> List[Entity]:
        """Get all tracked entities"""
        return list(self.entities.values())
    
    
    def get_entity_context(self, entity_id: str) -> str:
        """
        Get context about a specific entity
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Formatted entity information
        """
        entity = self.get_entity(entity_id)
        if not entity:
            return f"No information found about '{entity_id}'"
        
        context_parts = [
            f"=== Entity: {entity.name} ===",
            f"Type: {entity.entity_type}",
            f"First mentioned: {entity.first_seen.strftime('%Y-%m-%d %H:%M')}",
            f"Last mentioned: {entity.last_seen.strftime('%Y-%m-%d %H:%M')}",
            f"Total mentions: {len(entity.mentions)}",
        ]
        
        if entity.attributes:
            context_parts.append(f"Attributes: {entity.attributes}")
        
        return "\n".join(context_parts)
    
    
    # ========================================================================
    # Session Management
    # ========================================================================
    
    def _get_session_path(self) -> Path:
        """Get path to session file"""
        return self.memory_dir / f"session_{self.session_id}.json"
    
    
    def _save_session(self):
        """Save session to disk"""
        session_data = {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'turn_counter': self.turn_counter,
            'conversation_history': [turn.to_dict() for turn in self.conversation_history],
            'entities': {eid: entity.to_dict() for eid, entity in self.entities.items()}
        }
        
        session_path = self._get_session_path()
        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.debug(f"Session saved to {session_path}")
    
    
    def _load_session(self):
        """Load session from disk if exists"""
        session_path = self._get_session_path()
        
        if not session_path.exists():
            logger.info(f"No existing session found for {self.session_id}")
            return
        
        try:
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            
            self.session_start = datetime.fromisoformat(session_data['session_start'])
            self.turn_counter = session_data['turn_counter']
            self.conversation_history = [
                ConversationTurn.from_dict(turn) 
                for turn in session_data['conversation_history']
            ]
            self.entities = {
                eid: Entity.from_dict(edata) 
                for eid, edata in session_data['entities'].items()
            }
            
            logger.info(f"Loaded session {self.session_id} "
                       f"({len(self.conversation_history)} turns, "
                       f"{len(self.entities)} entities)")
        
        except Exception as e:
            logger.error(f"Error loading session: {e}")
    
    
    def clear_memory(self):
        """Clear all memory (conversation and entities)"""
        self.conversation_history.clear()
        self.entities.clear()
        self.turn_counter = 0
        self.session_start = datetime.now()
        
        if self.enable_persistence:
            self._save_session()
        
        logger.info("Memory cleared")
    
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'session_id': self.session_id,
            'session_duration': str(datetime.now() - self.session_start),
            'total_turns': self.turn_counter,
            'turns_in_memory': len(self.conversation_history),
            'total_entities': len(self.entities),
            'entities_by_type': self._count_entities_by_type(),
            'max_history_turns': self.max_history_turns,
            'persistence_enabled': self.enable_persistence,
            'entity_tracking_enabled': self.enable_entity_tracking
        }
    
    
    def _count_entities_by_type(self) -> Dict[str, int]:
        """Count entities by type"""
        counts = {}
        for entity in self.entities.values():
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts
    
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def detect_follow_up(self, query: str) -> bool:
        """
        Detect if query is a follow-up question
        
        Args:
            query: User query to check
            
        Returns:
            True if query appears to be a follow-up
        """
        follow_up_indicators = [
            'what about',
            'and the',
            'also',
            'how about',
            'what else',
            'tell me more',
            'explain that',
            'why',
            'how',
            'when did they',
            'who did',
            'what did',
        ]
        
        query_lower = query.lower()
        
        # Check for pronouns indicating reference to previous context
        pronouns = ['they', 'them', 'their', 'he', 'she', 'his', 'her', 'it', 'that', 'this']
        has_pronoun = any(pronoun in query_lower.split() for pronoun in pronouns)
        
        # Check for follow-up phrases
        has_follow_up_phrase = any(indicator in query_lower for indicator in follow_up_indicators)
        
        # Short questions are often follow-ups
        is_short = len(query.split()) < 10
        
        return (has_pronoun or has_follow_up_phrase) and is_short


# ============================================================================
# Convenience Functions
# ============================================================================

def create_memory_tools(session_id: str = "default", **kwargs) -> MemoryTools:
    """
    Create MemoryTools instance with default settings
    
    Args:
        session_id: Session identifier
        **kwargs: Additional MemoryTools parameters
        
    Returns:
        MemoryTools instance
    """
    return MemoryTools(session_id=session_id, **kwargs)
