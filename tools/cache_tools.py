"""
Cache Tools - Semantic Cache Manager

Provides semantic caching for RAG queries to save time and costs:
- Semantic similarity matching (not exact string match)
- TTL-based cache expiration
- FAISS-based fast similarity search
- Thread-safe cache operations

Features:
- Cache query-answer pairs with embeddings
- Retrieve cached answers for similar queries
- Automatic cache cleanup (TTL, max entries)
- Cache statistics and management

Integrates:
- core/embeddings.py (query embeddings)
- utils/azure_clients.py (OpenAI client)
- config/settings.py (cache configuration)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import time
from datetime import datetime, timedelta
import numpy as np
import faiss

from config.settings import (
    ENABLE_SEMANTIC_CACHE,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL_HOURS,
    CACHE_MAX_ENTRIES,
    CACHE_INDEX_FILE,
    CACHE_DATA_FILE,
    CACHE_PATH
)

from core.embeddings import generate_embedding

logger = logging.getLogger(__name__)


class CacheTools:
    """
    Semantic cache manager for RAG queries.
    
    Features:
    - Semantic similarity matching (FAISS)
    - TTL-based expiration
    - LRU eviction when max entries reached
    - Thread-safe operations
    """
    
    def __init__(self):
        """Initialize cache tools."""
        self.cache_enabled = ENABLE_SEMANTIC_CACHE
        self.similarity_threshold = CACHE_SIMILARITY_THRESHOLD
        self.ttl_hours = CACHE_TTL_HOURS
        self.max_entries = CACHE_MAX_ENTRIES
        
        # Cache storage
        self.cache_index = None  # FAISS index for query embeddings
        self.cache_data = []  # List of cache entries
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0,
            'saves': 0
        }
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"CacheTools initialized (enabled: {self.cache_enabled})")
    
    def _load_cache(self):
        """Load cache from disk."""
        if not self.cache_enabled:
            logger.info("Semantic cache disabled")
            return
        
        try:
            # Load FAISS index
            if CACHE_INDEX_FILE.exists():
                self.cache_index = faiss.read_index(str(CACHE_INDEX_FILE))
                logger.info(f"Loaded cache index: {self.cache_index.ntotal} entries")
            else:
                # Create new index (dimension = 3072 for text-embedding-3-large)
                self.cache_index = faiss.IndexFlatL2(3072)
                logger.info("Created new cache index")
            
            # Load cache data
            if CACHE_DATA_FILE.exists():
                with open(CACHE_DATA_FILE, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
                logger.info(f"Loaded cache data: {len(self.cache_data)} entries")
            else:
                self.cache_data = []
                logger.info("Created new cache data")
            
            # Clean expired entries
            self._clean_expired()
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            # Create new cache on error
            self.cache_index = faiss.IndexFlatL2(3072)
            self.cache_data = []
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.cache_enabled:
            return
        
        try:
            # Ensure cache directory exists
            CACHE_PATH.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.cache_index, str(CACHE_INDEX_FILE))
            
            # Save cache data
            with open(CACHE_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Cache saved: {len(self.cache_data)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _clean_expired(self):
        """Remove expired cache entries."""
        if not self.cache_data:
            return
        
        now = datetime.now()
        original_count = len(self.cache_data)
        
        # Filter out expired entries
        valid_indices = []
        valid_data = []
        
        for i, entry in enumerate(self.cache_data):
            created_at = datetime.fromisoformat(entry['created_at'])
            expiry = created_at + timedelta(hours=self.ttl_hours)
            
            if now < expiry:
                valid_indices.append(i)
                valid_data.append(entry)
        
        # Rebuild index if entries were removed
        if len(valid_indices) < original_count:
            logger.info(f"Cleaning {original_count - len(valid_indices)} expired entries")
            
            # Rebuild FAISS index
            self.cache_index = faiss.IndexFlatL2(3072)
            if valid_data:
                embeddings = np.array([entry['embedding'] for entry in valid_data], dtype=np.float32)
                self.cache_index.add(embeddings)
            
            self.cache_data = valid_data
            self._save_cache()
    
    def _evict_lru(self):
        """Evict least recently used entry if cache is full."""
        if len(self.cache_data) < self.max_entries:
            return
        
        logger.info(f"Cache full ({len(self.cache_data)} entries). Evicting LRU entry.")
        
        # Find least recently used entry
        lru_index = 0
        lru_time = datetime.now()
        
        for i, entry in enumerate(self.cache_data):
            last_used = datetime.fromisoformat(entry['last_used'])
            if last_used < lru_time:
                lru_time = last_used
                lru_index = i
        
        # Remove from cache
        self.cache_data.pop(lru_index)
        
        # Rebuild FAISS index
        self.cache_index = faiss.IndexFlatL2(3072)
        if self.cache_data:
            embeddings = np.array([entry['embedding'] for entry in self.cache_data], dtype=np.float32)
            self.cache_index.add(embeddings)
        
        self._save_cache()
    
    def get_cached_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached answer for semantically similar query.
        
        Args:
            query: User query
            
        Returns:
            Cached answer dict if found, None otherwise
        """
        if not self.cache_enabled:
            return None
        
        self.stats['total_queries'] += 1
        
        if not self.cache_data:
            self.stats['misses'] += 1
            return None
        
        try:
            # Get query embedding
            query_embedding = generate_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search in FAISS index
            distances, indices = self.cache_index.search(query_vector, k=1)
            
            if len(indices[0]) == 0:
                self.stats['misses'] += 1
                return None
            
            # Get closest match
            best_idx = indices[0][0]
            best_distance = distances[0][0]
            
            # Convert L2 distance to similarity (lower distance = higher similarity)
            # Using normalized vectors, similarity ≈ 1 - (distance/2)
            similarity = 1.0 - (best_distance / 2.0)
            
            logger.debug(f"Cache search: similarity={similarity:.4f}, threshold={self.similarity_threshold}")
            
            # Check if similarity meets threshold
            if similarity >= self.similarity_threshold:
                cached_entry = self.cache_data[best_idx]
                
                # Update last used time
                cached_entry['last_used'] = datetime.now().isoformat()
                cached_entry['hit_count'] = cached_entry.get('hit_count', 0) + 1
                self._save_cache()
                
                self.stats['hits'] += 1
                
                logger.info(f"✅ Cache HIT (similarity: {similarity:.4f})")
                logger.info(f"   Original query: {cached_entry['query']}")
                logger.info(f"   Current query: {query}")
                
                return {
                    'answer': cached_entry['answer'],
                    'sources': cached_entry.get('sources', []),
                    'cached': True,
                    'similarity': similarity,
                    'original_query': cached_entry['query'],
                    'hit_count': cached_entry['hit_count']
                }
            else:
                self.stats['misses'] += 1
                logger.debug(f"❌ Cache MISS (similarity {similarity:.4f} < threshold {self.similarity_threshold})")
                return None
                
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            self.stats['misses'] += 1
            return None
    
    def cache_answer(
        self,
        query: str,
        answer: str,
        sources: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache a query-answer pair.
        
        Args:
            query: User query
            answer: Generated answer
            sources: Source citations
            metadata: Additional metadata
            
        Returns:
            True if cached successfully
        """
        if not self.cache_enabled:
            return False
        
        try:
            # Evict LRU if cache is full
            self._evict_lru()
            
            # Get query embedding
            query_embedding = generate_embedding(query)
            
            # Create cache entry
            entry = {
                'query': query,
                'answer': answer,
                'sources': sources or [],
                'metadata': metadata or {},
                'embedding': query_embedding,
                'created_at': datetime.now().isoformat(),
                'last_used': datetime.now().isoformat(),
                'hit_count': 0
            }
            
            # Add to cache
            self.cache_data.append(entry)
            
            # Add to FAISS index
            vector = np.array([query_embedding], dtype=np.float32)
            self.cache_index.add(vector)
            
            # Save to disk
            self._save_cache()
            
            self.stats['saves'] += 1
            
            logger.info(f"💾 Cached answer for query: '{query[:50]}...'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache answer: {e}")
            return False
    
    def clear_cache(self):
        """Clear all cache entries."""
        logger.info("Clearing cache...")
        
        self.cache_index = faiss.IndexFlatL2(3072)
        self.cache_data = []
        self._save_cache()
        
        logger.info("✅ Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dict
        """
        hit_rate = (
            self.stats['hits'] / self.stats['total_queries']
            if self.stats['total_queries'] > 0
            else 0.0
        )
        
        return {
            'enabled': self.cache_enabled,
            'total_entries': len(self.cache_data),
            'max_entries': self.max_entries,
            'total_queries': self.stats['total_queries'],
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'saves': self.stats['saves'],
            'similarity_threshold': self.similarity_threshold,
            'ttl_hours': self.ttl_hours
        }
    
    def get_cache_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent cache entries (for debugging).
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of cache entries (without embeddings)
        """
        entries = []
        for entry in self.cache_data[-limit:]:
            entries.append({
                'query': entry['query'],
                'answer': entry['answer'][:100] + '...',
                'created_at': entry['created_at'],
                'last_used': entry['last_used'],
                'hit_count': entry.get('hit_count', 0)
            })
        return entries


# ============================================================================
# Convenience Functions
# ============================================================================

def get_cached_answer(query: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get cached answer.
    
    Args:
        query: User query
        
    Returns:
        Cached answer if found
    """
    cache = CacheTools()
    return cache.get_cached_answer(query)


def cache_answer(
    query: str,
    answer: str,
    sources: Optional[List[str]] = None
) -> bool:
    """
    Convenience function to cache an answer.
    
    Args:
        query: User query
        answer: Generated answer
        sources: Source citations
        
    Returns:
        True if cached successfully
    """
    cache = CacheTools()
    return cache.cache_answer(query, answer, sources)
