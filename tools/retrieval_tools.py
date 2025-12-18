"""
Retrieval Tools - Semantic Search Specialist

Provides semantic search over FAISS knowledge base with:
- Single query search
- Multi-query search (query variations)
- Hybrid search (semantic + keyword)
- Distance filtering
- Top-K retrieval with configurable parameters

Integrates:
- FAISS (vector similarity search)
- core/embeddings.py (query embedding)
- KBTools (loads index)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
import faiss

from core.embeddings import generate_embedding
from tools.kb_tools import KBTools
from config.settings import (
    DEFAULT_TOP_K,
    RETRIEVAL_BUFFER,
    MAX_DISTANCE_THRESHOLD
)

logger = logging.getLogger(__name__)


class RetrievalTools:
    """
    Semantic search over FAISS knowledge base.
    
    Features:
    - Query embedding generation
    - FAISS similarity search
    - Distance-based filtering
    - Result ranking and formatting
    """
    
    def __init__(self, kb_tools: Optional[KBTools] = None):
        """
        Initialize retrieval tools.
        
        Args:
            kb_tools: KBTools instance with loaded index (optional, will create if None)
        """
        if kb_tools is None:
            # Create and load KB
            kb_tools = KBTools()
            load_result = kb_tools.load_index()
            
            if not load_result['success']:
                raise ValueError(f"Failed to load knowledge base: {load_result.get('error')}")
        
        self.kb_tools = kb_tools
        self.index = kb_tools.index
        self.chunks = kb_tools.chunks
        
        if self.index is None or not self.chunks:
            raise ValueError("Knowledge base not loaded. Build index first.")
        
        logger.info(f"RetrievalTools initialized with {len(self.chunks)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        max_distance: Optional[float] = MAX_DISTANCE_THRESHOLD,
        return_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Semantic search for query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            max_distance: Maximum L2 distance threshold (filter far results)
            return_scores: Include similarity scores in results
            
        Returns:
            {
                'success': bool,
                'query': str,
                'results': list,  # List of matching chunks with metadata
                'total_results': int,
                'filtered_count': int,  # How many filtered by distance
                'error': str | None
            }
        """
        try:
            logger.info(f"Searching for: '{query[:50]}...' (top_k={top_k})")
            
            # Step 1: Generate query embedding
            query_embedding = generate_embedding(query)
            
            if not query_embedding or len(query_embedding) == 0:
                return {
                    'success': False,
                    'query': query,
                    'error': 'Failed to generate query embedding'
                }
            
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Step 2: Search FAISS index
            # Retrieve more than needed for distance filtering
            search_k = min(top_k + RETRIEVAL_BUFFER, len(self.chunks))
            
            distances, indices = self.index.search(query_vector, search_k)
            
            # Step 3: Format results and filter by distance
            results = []
            filtered_count = 0
            
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                # Check distance threshold
                if max_distance is not None and distance > max_distance:
                    filtered_count += 1
                    continue
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
                
                # Get chunk
                chunk = self.chunks[idx]
                
                # Build result
                result = {
                    'rank': len(results) + 1,
                    'text': chunk['text'],
                    'metadata': chunk.get('metadata', {}),
                }
                
                if return_scores:
                    # Convert L2 distance to similarity score (0-1, higher is better)
                    # Using: similarity = 1 / (1 + distance)
                    similarity = 1.0 / (1.0 + float(distance))
                    result['distance'] = float(distance)
                    result['similarity'] = round(similarity, 4)
                
                results.append(result)
            
            logger.info(f"Found {len(results)} results (filtered {filtered_count} by distance)")
            
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'filtered_count': filtered_count
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return {
                'success': False,
                'query': query,
                'error': f'Search failed: {str(e)}'
            }
    
    def search_multiple_queries(
        self,
        queries: List[str],
        top_k: int = DEFAULT_TOP_K,
        max_distance: Optional[float] = MAX_DISTANCE_THRESHOLD,
        fusion_method: str = "reciprocal_rank"
    ) -> Dict[str, Any]:
        """
        Search multiple query variations and fuse results.
        
        Useful for query expansion / HyDE / multi-query retrieval.
        
        Args:
            queries: List of query variations
            top_k: Number of final results
            max_distance: Distance threshold
            fusion_method: "reciprocal_rank" or "max_score"
            
        Returns:
            Fused results dict
        """
        try:
            logger.info(f"Multi-query search with {len(queries)} variations")
            
            # Search each query
            all_results = []
            for query in queries:
                result = self.search(
                    query=query,
                    top_k=top_k * 2,  # Get more for fusion
                    max_distance=max_distance,
                    return_scores=True
                )
                
                if result['success']:
                    all_results.append(result['results'])
            
            if not all_results:
                return {
                    'success': False,
                    'queries': queries,
                    'error': 'All queries failed'
                }
            
            # Fuse results
            if fusion_method == "reciprocal_rank":
                fused = self._reciprocal_rank_fusion(all_results, top_k)
            else:
                fused = self._max_score_fusion(all_results, top_k)
            
            return {
                'success': True,
                'queries': queries,
                'results': fused,
                'total_results': len(fused),
                'fusion_method': fusion_method
            }
            
        except Exception as e:
            logger.error(f"Multi-query search failed: {e}", exc_info=True)
            return {
                'success': False,
                'queries': queries,
                'error': str(e)
            }
    
    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Dict]],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) for combining multiple result lists.
        
        RRF score = sum(1 / (k + rank)) for each occurrence
        """
        # Track chunks by text (assuming text is unique identifier)
        chunk_scores = {}
        chunk_data = {}
        
        for results in results_list:
            for rank, result in enumerate(results, 1):
                text = result['text']
                
                # RRF score contribution
                score = 1.0 / (k + rank)
                
                if text not in chunk_scores:
                    chunk_scores[text] = 0.0
                    chunk_data[text] = result
                
                chunk_scores[text] += score
        
        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build final results
        fused = []
        for i, (text, score) in enumerate(sorted_chunks, 1):
            result = chunk_data[text].copy()
            result['rank'] = i
            result['fusion_score'] = round(score, 4)
            fused.append(result)
        
        return fused
    
    def _max_score_fusion(
        self,
        results_list: List[List[Dict]],
        top_k: int
    ) -> List[Dict]:
        """
        Max score fusion - take highest similarity score for each chunk.
        """
        chunk_scores = {}
        chunk_data = {}
        
        for results in results_list:
            for result in results:
                text = result['text']
                score = result.get('similarity', 0.0)
                
                if text not in chunk_scores or score > chunk_scores[text]:
                    chunk_scores[text] = score
                    chunk_data[text] = result
        
        # Sort by max score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build final results
        fused = []
        for i, (text, score) in enumerate(sorted_chunks, 1):
            result = chunk_data[text].copy()
            result['rank'] = i
            fused.append(result)
        
        return fused
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval system statistics.
        
        Returns:
            Statistics dict
        """
        return {
            'index_size': len(self.chunks),
            'embedding_dimension': self.index.d if self.index else None,
            'default_top_k': DEFAULT_TOP_K,
            'max_distance_threshold': MAX_DISTANCE_THRESHOLD,
            'retrieval_buffer': RETRIEVAL_BUFFER
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def search_knowledge_base(
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """
    Convenience function for single query search.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        Search results dict
    """
    retrieval = RetrievalTools()
    return retrieval.search(query=query, top_k=top_k)
