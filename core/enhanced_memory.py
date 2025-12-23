"""
Enhanced Multi-Layer Conversation Memory

Handles conversation history with multiple storage strategies:
- Sliding Window: Recent N exchanges
- Summary: Condensed history for long conversations
- Entity-Aware: Integrates with EntityMemory
- Token Management: Prevents context overflow

Designed for medical RAG with long multi-turn conversations.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import deque

from utils.azure_clients import get_chat_completion
from core.entity_memory import EntityMemory, extract_and_store


# ============================================================================
# SECTION 1: Memory Exchange Structure
# ============================================================================

class ConversationExchange:
    """
    Single question-answer exchange in conversation.
    
    Stores:
    - User question
    - Assistant answer
    - Timestamp
    - Metadata (sources, entities, etc.)
    """
    
    def __init__(
        self,
        question: str,
        answer: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationExchange':
        """Create from dictionary."""
        return cls(
            question=data['question'],
            answer=data['answer'],
            timestamp=data.get('timestamp'),
            metadata=data.get('metadata', {})
        )
    
    def get_tokens_estimate(self) -> int:
        """Estimate token count for this exchange."""
        # Rough estimate: 1 token ≈ 4 characters
        total_chars = len(self.question) + len(self.answer)
        return total_chars // 4


# ============================================================================
# SECTION 2: Enhanced Memory Class
# ============================================================================

class EnhancedMemory:
    """
    Multi-layer conversation memory system.
    
    Layers:
    1. Recent Window: Last N exchanges (full detail)
    2. Summary: Condensed version of older exchanges
    3. Entity Memory: Tracked entities across conversation
    
    Features:
    - Automatic token management
    - Sliding window for recent context
    - Summary generation for older context
    - Entity extraction and tracking
    """
    
    def __init__(
        self,
        max_window_size: int = 5,
        max_summary_tokens: int = 500,
        enable_entity_memory: bool = True,
        enable_auto_summary: bool = True
    ):
        """Initialize enhanced memory."""
        self.max_window_size = max_window_size
        self.max_summary_tokens = max_summary_tokens
        self.enable_entity_memory = enable_entity_memory
        self.enable_auto_summary = enable_auto_summary
        
        # Recent exchanges (sliding window)
        self.recent_exchanges: deque = deque(maxlen=max_window_size)
        
        # Older exchanges (for summary generation)
        self.archived_exchanges: List[ConversationExchange] = []
        
        # Current summary of archived exchanges
        self.summary: Optional[str] = None
        
        # Entity memory
        self.entity_memory: Optional[EntityMemory] = None
        if enable_entity_memory:
            self.entity_memory = EntityMemory(max_history=50)
        
        # Rate limiting for summary generation
        self.last_summary_time = 0
        self.summary_cooldown = 60  # seconds

    
    def add_exchange(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        extract_entities: bool = True
    ):
        """
        Add a conversation exchange.
        
        Args:
            question: User question
            answer: Assistant answer
            metadata: Optional metadata (sources, etc.)
            extract_entities: Extract entities from Q&A
        """
        exchange = ConversationExchange(
            question=question,
            answer=answer,
            metadata=metadata
        )
        
        # Extract entities if enabled
        if extract_entities and self.entity_memory:
            # Extract from both question and answer
            combined_text = f"{question}\n{answer}"
            extract_and_store(
                text=combined_text,
                entity_memory=self.entity_memory,
                context=f"Q: {question[:50]}...",
                use_llm=False  # Use regex for now (LLM optional)
            )
        
        # If window is full, archive the oldest exchange
        if len(self.recent_exchanges) == self.max_window_size:
            oldest = self.recent_exchanges[0]
            self.archived_exchanges.append(oldest)
            
            # Trigger summary update if enabled
            if self.enable_auto_summary and len(self.archived_exchanges) % 5 == 0:
                self._update_summary()
        
        # Add to recent window
        self.recent_exchanges.append(exchange)
    
    def get_recent_exchanges(self, limit: Optional[int] = None) -> List[ConversationExchange]:
        """
        Get recent exchanges.
        
        Args:
            limit: Maximum number to return (None = all)
            
        Returns:
            List of recent exchanges
        """
        exchanges = list(self.recent_exchanges)
        if limit:
            exchanges = exchanges[-limit:]
        return exchanges
    
    def get_conversation_string(
        self,
        include_summary: bool = True,
        include_entities: bool = True,
        max_exchanges: Optional[int] = None
    ) -> str:
        """
        Get formatted conversation history string.
        
        Args:
            include_summary: Include summary of archived exchanges
            include_entities: Include entity context
            max_exchanges: Maximum recent exchanges to include
            
        Returns:
            Formatted conversation history
        """
        parts = []
        
        # 1. Summary of older exchanges
        if include_summary and self.summary:
            parts.append("=== Previous Conversation Summary ===")
            parts.append(self.summary)
            parts.append("")
        
        # 2. Entity context
        if include_entities and self.entity_memory:
            entity_context = self.entity_memory.get_recent_context(max_entities=5)
            if entity_context and entity_context != "No entities tracked yet.":
                parts.append("=== Recently Mentioned ===")
                parts.append(entity_context)
                parts.append("")
        
        # 3. Recent exchanges
        recent = self.get_recent_exchanges(limit=max_exchanges)
        if recent:
            parts.append("=== Recent Conversation ===")
            for i, exchange in enumerate(recent, 1):
                parts.append(f"Turn {i}:")
                parts.append(f"User: {exchange.question}")
                parts.append(f"Assistant: {exchange.answer}")
                parts.append("")
        
        return "\n".join(parts)
    
    def get_messages_format(
        self,
        max_exchanges: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history in OpenAI messages format.
        
        Args:
            max_exchanges: Maximum recent exchanges to include
            
        Returns:
            List of message dicts [{"role": "user/assistant", "content": "..."}]
        """
        messages = []
        
        recent = self.get_recent_exchanges(limit=max_exchanges)
        for exchange in recent:
            messages.append({
                "role": "user",
                "content": exchange.question
            })
            messages.append({
                "role": "assistant",
                "content": exchange.answer
            })
        
        return messages
    
    def _update_summary(self):
        """
        Generate summary of archived exchanges with rate limiting.
        Uses LLM to create condensed summary of conversation history.
        """
        import time
        
        # Rate limit check
        current_time = time.time()
        if current_time - self.last_summary_time < self.summary_cooldown:
            time_until_next = int(self.summary_cooldown - (current_time - self.last_summary_time))
            print(f"⏸️ Summary update rate limited (cooldown: {time_until_next}s remaining)")
            return
        
        if not self.archived_exchanges:
            return
        
        # Build text from archived exchanges
        archive_text = []

        for i, exchange in enumerate(self.archived_exchanges, 1):
            archive_text.append(f"Exchange {i}:")
            archive_text.append(f"Q: {exchange.question}")
            archive_text.append(f"A: {exchange.answer}")
        
        archive_str = "\n".join(archive_text)
        
        # Generate summary using LLM
        try:
            system_prompt = """You are a medical conversation summarizer.
Create a concise summary of this conversation history, focusing on:
- Key medical facts mentioned
- Diagnoses and conditions discussed
- Medications and treatments
- Important dates and measurements

Keep the summary under 200 words."""
            
            user_prompt = f"Summarize this conversation:\n\n{archive_str}"
            
            self.summary = get_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            # Update rate limit timestamp
            self.last_summary_time = time.time()
            
        except Exception as e:
            print(f"⚠️ Failed to generate summary: {e}")

    
    def get_token_estimate(self) -> Dict[str, int]:
        """
        Estimate token usage for memory components.
        
        Returns:
            Dict with token estimates for each component
        """
        recent_tokens = sum(ex.get_tokens_estimate() for ex in self.recent_exchanges)
        
        summary_tokens = 0
        if self.summary:
            summary_tokens = len(self.summary) // 4
        
        entity_tokens = 0
        if self.entity_memory:
            entity_context = self.entity_memory.get_recent_context()
            entity_tokens = len(entity_context) // 4
        
        return {
            'recent_exchanges': recent_tokens,
            'summary': summary_tokens,
            'entities': entity_tokens,
            'total': recent_tokens + summary_tokens + entity_tokens
        }
    
    def clear(self):
        """Clear all memory."""
        self.recent_exchanges.clear()
        self.archived_exchanges.clear()
        self.summary = None
        if self.entity_memory:
            self.entity_memory.clear()
    
    def clear_recent(self):
        """Clear only recent exchanges (keep summary and entities)."""
        self.recent_exchanges.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict with statistics
        """
        stats = {
            'recent_exchanges': len(self.recent_exchanges),
            'archived_exchanges': len(self.archived_exchanges),
            'total_exchanges': len(self.recent_exchanges) + len(self.archived_exchanges),
            'has_summary': self.summary is not None,
            'token_estimates': self.get_token_estimate()
        }
        
        if self.entity_memory:
            stats['entity_memory'] = self.entity_memory.get_stats()
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize memory to dictionary.
        
        Returns:
            Dict representation
        """
        return {
            'config': {
                'max_window_size': self.max_window_size,
                'max_summary_tokens': self.max_summary_tokens,
                'enable_entity_memory': self.enable_entity_memory,
                'enable_auto_summary': self.enable_auto_summary
            },
            'recent_exchanges': [ex.to_dict() for ex in self.recent_exchanges],
            'archived_exchanges': [ex.to_dict() for ex in self.archived_exchanges],
            'summary': self.summary,
            'entity_memory': self.entity_memory.to_dict() if self.entity_memory else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedMemory':
        """
        Deserialize memory from dictionary.
        
        Args:
            data: Dict representation
            
        Returns:
            EnhancedMemory instance
        """
        config = data.get('config', {})
        memory = cls(
            max_window_size=config.get('max_window_size', 5),
            max_summary_tokens=config.get('max_summary_tokens', 500),
            enable_entity_memory=config.get('enable_entity_memory', True),
            enable_auto_summary=config.get('enable_auto_summary', True)
        )
        
        # Restore recent exchanges
        for ex_data in data.get('recent_exchanges', []):
            exchange = ConversationExchange.from_dict(ex_data)
            memory.recent_exchanges.append(exchange)
        
        # Restore archived exchanges
        for ex_data in data.get('archived_exchanges', []):
            exchange = ConversationExchange.from_dict(ex_data)
            memory.archived_exchanges.append(exchange)
        
        # Restore summary
        memory.summary = data.get('summary')
        
        # Restore entity memory
        if data.get('entity_memory') and memory.entity_memory:
            memory.entity_memory = EntityMemory.from_dict(data['entity_memory'])
        
        return memory


# ============================================================================
# SECTION 3: Helper Functions
# ============================================================================

def print_memory_stats(memory: EnhancedMemory):
    """
    Print formatted memory statistics.
    
    Args:
        memory: EnhancedMemory instance
    """
    stats = memory.get_stats()
    
    print("\n" + "=" * 80)
    print("ENHANCED MEMORY STATISTICS")
    print("=" * 80)
    
    print(f"\nExchanges:")
    print(f"  • Recent: {stats['recent_exchanges']}")
    print(f"  • Archived: {stats['archived_exchanges']}")
    print(f"  • Total: {stats['total_exchanges']}")
    print(f"  • Has Summary: {stats['has_summary']}")
    
    tokens = stats['token_estimates']
    print(f"\nToken Estimates:")
    print(f"  • Recent Exchanges: {tokens['recent_exchanges']}")
    print(f"  • Summary: {tokens['summary']}")
    print(f"  • Entities: {tokens['entities']}")
    print(f"  • Total: {tokens['total']}")
    
    if 'entity_memory' in stats:
        entity_stats = stats['entity_memory']
        print(f"\nEntity Memory:")
        print(f"  • Total Mentions: {entity_stats['total_mentions']}")
        print(f"  • Unique Entities: {entity_stats['unique_entities']}")
    
    print("\n" + "=" * 80 + "\n")
