"""
Semantic Cache for Agentic Medical RAG

FAISS-based caching system that stores LLM responses and returns cached
answers for semantically similar questions.

Benefits:
- Reduces API costs (reuse answers)
- Faster responses (no LLM call)
- Consistent answers for similar questions

Uses embedding similarity to match questions.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import faiss

from core.embeddings import generate_embedding, validate_embedding


from config.settings import CACHE_PATH, EMBEDDING_DIMENSION

# ============================================================================
# SECTION 1: Cache Entry Structure
# ============================================================================

class CacheEntry:
    """
    Single cache entry storing question, answer, and metadata.
    """
    
    def __init__(
        self,
        question: str,
        answer: str,
        embedding: List[float],
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        hit_count: int = 0
    ):
        self.question = question
        self.answer = answer
        self.embedding = embedding
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}
        self.hit_count = hit_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'question': self.question,
            'answer': self.answer,
            'embedding': self.embedding,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'hit_count': self.hit_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Deserialize from dictionary."""
        return cls(
            question=data['question'],
            answer=data['answer'],
            embedding=data['embedding'],
            timestamp=data.get('timestamp'),
            metadata=data.get('metadata', {}),
            hit_count=data.get('hit_count', 0)
        )
    
    def is_expired(self, ttl_hours: Optional[int] = None) -> bool:
        """
        Check if entry is expired.
        
        Args:
            ttl_hours: Time-to-live in hours (None = never expires)
            
        Returns:
            bool: True if expired
        """
        if ttl_hours is None:
            return False
        
        entry_time = datetime.fromisoformat(self.timestamp)
        age = datetime.now() - entry_time
        return age > timedelta(hours=ttl_hours)


# ============================================================================
# SECTION 2: Semantic Cache Class
# ============================================================================

class SemanticCache:
    """
    FAISS-based semantic cache for LLM responses.
    
    Features:
    - Similarity-based matching
    - Configurable similarity threshold
    - Time-based expiration
    - Hit rate tracking
    - Persistent storage
    """
    
    def __init__(
        self,
        cache_dir: Path = CACHE_PATH,
        similarity_threshold: float = 0.95,
        ttl_hours: Optional[int] = None,
        max_cache_size: int = 1000
    ):
        """
        Initialize semantic cache.
        
        Args:
            cache_dir: Directory to store cache files
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            ttl_hours: Time-to-live for cache entries (None = never expire)
            max_cache_size: Maximum number of cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self.max_cache_size = max_cache_size
        
        # FAISS index for similarity search
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product (cosine similarity)
        
        # Cache entries (ordered list)
        self.entries: List[CacheEntry] = []
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0
        }
        
        # Initialize or load index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        # Inner product index for cosine similarity
        # (assumes embeddings are normalized)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    
    def _normalize_embedding(self, embedding: List[float]) -> np.ndarray:
        """
        Normalize embedding for cosine similarity.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized numpy array
        """
        arr = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(arr)  # In-place normalization
        return arr
    
    def add(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add question-answer pair to cache.
        
        Args:
            question: User question
            answer: LLM answer
            metadata: Optional metadata
            
        Returns:
            bool: True if added successfully
        """
        # Generate embedding for question
        embedding = generate_embedding(question)
        
        if not validate_embedding(embedding):
            return False
        
        # Create cache entry
        entry = CacheEntry(
            question=question,
            answer=answer,
            embedding=embedding,
            metadata=metadata
        )
        
        # Add to FAISS index
        normalized = self._normalize_embedding(embedding)
        self.index.add(normalized)
        
        # Add to entries list
        self.entries.append(entry)
        
        # Evict oldest if cache is full
        if len(self.entries) > self.max_cache_size:
            self._evict_oldest()
        
        return True
    
    def get(self, question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve cached answer for similar question.
        
        Args:
            question: User question
            
        Returns:
            Tuple of (answer, metadata) if cache hit, None if miss
        """
        self.stats['total_queries'] += 1
        
        if len(self.entries) == 0:
            self.stats['misses'] += 1
            return None
        
        # Generate embedding for query
        query_embedding = generate_embedding(question)
        
        if not validate_embedding(query_embedding):
            self.stats['misses'] += 1
            return None
        
        # Search FAISS index
        query_normalized = self._normalize_embedding(query_embedding)
        distances, indices = self.index.search(query_normalized, k=1)
        
        # Check similarity threshold
        similarity = float(distances[0][0])
        best_idx = int(indices[0][0])
        
        if similarity < self.similarity_threshold:
            self.stats['misses'] += 1
            return None
        
        # Get cached entry
        entry = self.entries[best_idx]
        
        # Check expiration
        if entry.is_expired(self.ttl_hours):
            self.stats['misses'] += 1
            return None
        
        # Cache hit!
        self.stats['hits'] += 1
        entry.hit_count += 1
        
        return (entry.answer, {
            'cached_question': entry.question,
            'similarity': similarity,
            'hit_count': entry.hit_count,
            'timestamp': entry.timestamp,
            **entry.metadata
        })
    
    def _evict_oldest(self):
        """Remove oldest cache entry."""
        if not self.entries:
            return
        
        # Remove first entry (oldest)
        self.entries.pop(0)
        
        # Rebuild FAISS index (FAISS doesn't support deletion)
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current entries."""
        self._initialize_index()
        
        if not self.entries:
            return
        
        # Add all entries to new index
        embeddings = [entry.embedding for entry in self.entries]
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
    
    def clear_expired(self):
        """Remove expired entries from cache."""
        if self.ttl_hours is None:
            return
        
        original_count = len(self.entries)
        
        # Filter non-expired entries
        self.entries = [
            entry for entry in self.entries
            if not entry.is_expired(self.ttl_hours)
        ]
        
        removed = original_count - len(self.entries)
        
        if removed > 0:
            self._rebuild_index()
        
        return removed
    
    def clear(self):
        """Clear all cache entries."""
        self.entries.clear()
        self._initialize_index()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0
        }
    
    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            float: Hit rate (0-1)
        """
        if self.stats['total_queries'] == 0:
            return 0.0
        return self.stats['hits'] / self.stats['total_queries']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with statistics
        """
        return {
            'total_entries': len(self.entries),
            'total_queries': self.stats['total_queries'],
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': self.get_hit_rate(),
            'similarity_threshold': self.similarity_threshold,
            'ttl_hours': self.ttl_hours,
            'max_cache_size': self.max_cache_size
        }
    
    def save(self, filename: str = "semantic_cache.json"):
        """
        Save cache to disk.
        
        Args:
            filename: Cache file name
        """
        cache_file = self.cache_dir / filename
        
        data = {
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'ttl_hours': self.ttl_hours,
                'max_cache_size': self.max_cache_size
            },
            'entries': [entry.to_dict() for entry in self.entries],
            'stats': self.stats
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename: str = "semantic_cache.json") -> bool:
        """
        Load cache from disk.
        
        Args:
            filename: Cache file name
            
        Returns:
            bool: True if loaded successfully
        """
        cache_file = self.cache_dir / filename
        
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Restore config
            config = data.get('config', {})
            self.similarity_threshold = config.get('similarity_threshold', 0.95)
            self.ttl_hours = config.get('ttl_hours')
            self.max_cache_size = config.get('max_cache_size', 1000)
            
            # Restore entries
            self.entries = [
                CacheEntry.from_dict(entry_data)
                for entry_data in data.get('entries', [])
            ]
            
            # Restore stats
            self.stats = data.get('stats', {
                'hits': 0,
                'misses': 0,
                'total_queries': 0
            })
            
            # Rebuild FAISS index
            self._rebuild_index()
            
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            return False


# ============================================================================
# SECTION 3: Helper Functions
# ============================================================================

def print_cache_stats(cache: SemanticCache):
    """
    Print formatted cache statistics.
    
    Args:
        cache: SemanticCache instance
    """
    stats = cache.get_stats()
    
    print("\n" + "=" * 80)
    print("SEMANTIC CACHE STATISTICS")
    print("=" * 80)
    
    print(f"\nCache Status:")
    print(f"  • Total Entries: {stats['total_entries']}")
    print(f"  • Max Size: {stats['max_cache_size']}")
    print(f"  • Similarity Threshold: {stats['similarity_threshold']:.2f}")
    print(f"  • TTL: {stats['ttl_hours']} hours" if stats['ttl_hours'] else "  • TTL: Never expires")
    
    print(f"\nQuery Statistics:")
    print(f"  • Total Queries: {stats['total_queries']}")
    print(f"  • Cache Hits: {stats['hits']}")
    print(f"  • Cache Misses: {stats['misses']}")
    print(f"  • Hit Rate: {stats['hit_rate']:.2%}")
    
    print("\n" + "=" * 80 + "\n")
