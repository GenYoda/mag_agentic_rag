"""
Reranking Tools - Result Reranking Specialist

Provides reranking of retrieved chunks using:
- Cross-encoder models (local, free, fast)
- LLM-based reranking (optional, expensive, high quality)

Features:
- Local cross-encoder reranking (default)
- Score normalization
- Batch processing
- Optional LLM fallback
- Model caching (FIXED: Singleton pattern)

Integrates:
- sentence-transformers cross-encoder models
- utils/azure_clients.py LLM reranking
- config/settings.py configuration
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from crewai.tools import tool
import logging
import numpy as np
import threading

from config.settings import (
    RERANKING_MODEL,
    RERANKING_METHOD,
    ENABLE_LLM_RERANKING_FALLBACK,
    RERANKING_BATCH_SIZE,
    RERANKING_SCORE_NORMALIZATION,
    LLM_RERANKING_TEMPERATURE,
    LLM_RERANKING_MAX_TOKENS,
    CROSS_ENCODER_CACHE_DIR
)

logger = logging.getLogger(__name__)


class RerankingTools:
    """Reranks retrieved chunks using cross-encoder or LLM."""
    
    def __init__(self, method: str = RERANKING_METHOD, model_name: str = RERANKING_MODEL):
        """Initialize reranking tools."""
        self.method = method
        self.model_name = model_name
        self.crossencoder = None  # Lazy load
        logger.info(f"RerankingTools initialized with method: {method}")

    def load_crossencoder(self):
        """Lazy load cross-encoder model."""
        if self.crossencoder is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            from pathlib import Path
            
            local_model_path = Path("models/ms-marco-MiniLM-L-6-v2")
            
            if local_model_path.exists() and local_model_path.is_dir():
                logger.info(f"Found local model at {local_model_path}")
                model_path = str(local_model_path)
            else:
                logger.info(f"Attempting download from HuggingFace: {self.model_name}")
                model_path = self.model_name
            
            self.crossencoder = CrossEncoder(
                model_path,
                max_length=512,
                device='cpu'
            )
            logger.info(f"✅ Cross-encoder model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise ImportError("sentence-transformers required for cross-encoder reranking")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        normalize_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """Rerank chunks using specified method."""
        if not chunks:
            return []
        
        # Normalize chunks to dict format
        normalized_chunks = []
        for chunk in chunks:
            if isinstance(chunk, str):
                normalized_chunks.append({'text': chunk, 'metadata': {}})
            elif isinstance(chunk, dict):
                if 'text' not in chunk:
                    chunk['text'] = str(chunk.get('chunk', ''))
                normalized_chunks.append(chunk)
            else:
                normalized_chunks.append({'text': str(chunk), 'metadata': {}})
        
        chunks = normalized_chunks
        method = method or self.method
        
        logger.info(f"Reranking {len(chunks)} chunks using method: {method}")
        
        try:
            if method == "crossencoder":
                reranked = self._rerank_with_crossencoder(
                    query=query,
                    chunks=chunks,
                    normalize=normalize_scores
                )
            elif method == "llm":
                reranked = self._rerank_with_llm(
                    query=query,
                    chunks=chunks,
                    normalize=normalize_scores
                )
            else:
                logger.warning(f"Unknown reranking method: {method}")
                return chunks
            
            if top_k is not None and top_k > 0:
                reranked = reranked[:top_k]
            
            logger.info(f"Reranking complete. Returning {len(reranked)} chunks")
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            return chunks

    def _rerank_with_crossencoder(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """Rerank using local cross-encoder model."""
        self.load_crossencoder()
        
        pairs = [(query, chunk['text']) for chunk in chunks]
        
        logger.info(f"Computing cross-encoder scores for {len(pairs)} pairs...")
        scores = self.crossencoder.predict(pairs, batch_size=RERANKING_BATCH_SIZE, show_progress_bar=False)
        
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        
        if normalize:
            scores = self._normalize_scores(scores, method=RERANKING_SCORE_NORMALIZATION)
        
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)
        
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Cross-encoder reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
        return reranked

    def _rerank_with_llm(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """Rerank using LLM (expensive but high quality)."""
        from utils.azure_clients import get_chat_completion
        
        logger.info("Using LLM reranking (API cost incurred)")
        
        chunks_text = "\n\n".join([
            f"{i+1}. {chunk['text'][:300]}..."
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""You are a relevance scoring expert. Score each document's relevance to the query on a scale of 0-10.

Query: {query}

Documents:
{chunks_text}

Instructions:
- Score each document from 0 (not relevant) to 10 (highly relevant)
- Consider semantic relevance, not just keyword matching
- Return ONLY a JSON array of scores in order: [score1, score2, ...]

Scores:"""
        
        try:
            response = get_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_RERANKING_TEMPERATURE,
                max_tokens=LLM_RERANKING_MAX_TOKENS
            )
            
            import json
            scores = json.loads(response.strip())
            
            if len(scores) != len(chunks):
                logger.warning(f"Score count mismatch: {len(scores)} vs {len(chunks)}")
                return self._rerank_with_crossencoder(query, chunks, normalize)
            
            if normalize:
                scores = [s / 10.0 for s in scores]
            
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)
            
            reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"LLM reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
            return reranked
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}. Falling back to cross-encoder")
            return self._rerank_with_crossencoder(query, chunks, normalize)

    def _normalize_scores(self, scores: List[float], method: str = "minmax") -> List[float]:
        """Normalize scores to 0-1 range."""
        if method == "none":
            return scores
        
        scores = np.array(scores)
        
        if method == "minmax":
            min_score = scores.min()
            max_score = scores.max()
            if max_score - min_score == 0:
                return [0.5] * len(scores)
            normalized = (scores - min_score) / (max_score - min_score)
        elif method == "sigmoid":
            normalized = 1 / (1 + np.exp(-scores))
        else:
            logger.warning(f"Unknown normalization method: {method}. Using minmax")
            return self._normalize_scores(scores, method="minmax")
        
        return normalized.tolist()

    def rerank_with_fallback(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        fallback_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Smart reranking with automatic LLM fallback."""
        logger.info(f"Smart reranking with fallback enabled")
        
        try:
            reranked = self.rerank(
                query=query,
                chunks=chunks,
                top_k=None,
                method="crossencoder"
            )
            
            if not reranked:
                logger.warning("Cross-encoder returned no results. Falling back to LLM")
                return self._fallback_to_llm(query, chunks, top_k, reason="no_results")
            
            if isinstance(reranked[0], dict):
                top_score = reranked[0].get('rerank_score', 0)
            else:
                top_score = 0
            
            if top_score < fallback_threshold:
                logger.warning(f"Cross-encoder top score {top_score:.4f} below threshold. Falling back to LLM")
                return self._fallback_to_llm(query, chunks, top_k, reason="low_quality")
            
            logger.info(f"✅ Cross-encoder reranking successful (top score: {top_score:.4f})")
            return {
                'success': True,
                'reranked_chunks': reranked[:top_k],
                'method_used': 'crossencoder',
                'top_score': top_score,
                'fallback_triggered': False
            }
            
        except Exception as e:
            logger.error(f"Cross-encoder failed: {e}. Falling back to LLM")
            return self._fallback_to_llm(query, chunks, top_k, reason="error")

    def _fallback_to_llm(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        reason: str = "unknown"
    ) -> Dict[str, Any]:
        """Fallback to LLM reranking."""
        logger.info(f"🔄 Falling back to LLM reranking (reason: {reason})")
        
        if not ENABLE_LLM_RERANKING_FALLBACK:
            logger.warning("LLM fallback disabled in config. Returning original chunks")
            return {
                'success': False,
                'reranked_chunks': chunks[:top_k],
                'method_used': 'none',
                'fallback_triggered': True,
                'fallback_reason': reason,
                'error': 'LLM fallback disabled'
            }
        
        try:
            reranked = self.rerank(query=query, chunks=chunks, top_k=top_k, method="llm")
            top_score = reranked[0].get('rerank_score', 0) if reranked else 0
            
            logger.info(f"✅ LLM reranking successful (top score: {top_score:.4f})")
            return {
                'success': True,
                'reranked_chunks': reranked,
                'method_used': 'llm',
                'top_score': top_score,
                'fallback_triggered': True,
                'fallback_reason': reason
            }
        except Exception as e:
            logger.error(f"LLM reranking also failed: {e}")
            return {
                'success': False,
                'reranked_chunks': chunks[:top_k],
                'method_used': 'none',
                'fallback_triggered': True,
                'fallback_reason': reason,
                'error': str(e)
            }


# ============================================================================
# MODULE-LEVEL SINGLETON (FIX: Prevent model reload on every tool call)
# ============================================================================

_reranker_instance = None
_reranker_lock = threading.Lock()

def get_reranker() -> RerankingTools:
    """
    Get or create singleton RerankingTools instance.
    Prevents 900MB model reload on every query.
    """
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                _reranker_instance = RerankingTools()
    return _reranker_instance


# ============================================================================
# CrewAI Tool Wrappers
# ============================================================================

@tool("Rerank with Cross-Encoder")
def rerank_crossencoder_tool(query: str, chunks: list, top_k: int = 5) -> dict:
    """
    Rerank chunks using cross-encoder model (fast, local).
    
    Args:
        query: User query
        chunks: Retrieved chunks to rerank
        top_k: Number of top results to return
    
    Returns:
        dict with success, reranked_chunks, method, top_score
    """
    reranker = get_reranker()
    try:
        reranked = reranker.rerank(query=query, chunks=chunks, top_k=top_k, method="crossencoder")
        return {
            'success': True,
            'reranked_chunks': reranked,
            'method': 'crossencoder',
            'top_score': reranked[0].get('rerank_score', 0) if reranked else 0
        }
    except Exception as e:
        logger.error(f"Cross-encoder reranking failed: {e}")
        return {'success': False, 'reranked_chunks': chunks[:top_k], 'method': 'none', 'error': str(e)}


@tool("Rerank with LLM")
def rerank_llm_tool(query: str, chunks: list, top_k: int = 5) -> dict:
    """
    Rerank chunks using LLM-based scoring (higher quality, slower).
    
    Args:
        query: User query
        chunks: Retrieved chunks to rerank
        top_k: Number of top results to return
    
    Returns:
        dict with success, reranked_chunks, method, top_score
    """
    reranker = get_reranker()
    try:
        reranked = reranker.rerank(query=query, chunks=chunks, top_k=top_k, method="llm")
        return {
            'success': True,
            'reranked_chunks': reranked,
            'method': 'llm',
            'top_score': reranked[0].get('rerank_score', 0) if reranked else 0
        }
    except Exception as e:
        logger.error(f"LLM reranking failed: {e}")
        return {'success': False, 'reranked_chunks': chunks[:top_k], 'method': 'none', 'error': str(e)}


@tool("Smart Rerank with Fallback")
def rerank_smart_fallback_tool(query: str, chunks: list, top_k: int = 5, fallback_threshold: float = 0.5) -> dict:
    """
    Smart reranking with automatic LLM fallback.
    Tries cross-encoder first, falls back to LLM if scores are low.
    
    Args:
        query: User query
        chunks: Retrieved chunks to rerank
        top_k: Number of top results to return
        fallback_threshold: If top score < this, trigger LLM fallback
    
    Returns:
        dict with success, reranked_chunks, method_used, fallback_triggered
    """
    reranker = get_reranker()
    return reranker.rerank_with_fallback(query, chunks, top_k, fallback_threshold)
