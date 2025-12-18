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
- Model caching

Integrates:
- sentence-transformers (cross-encoder models)
- utils/azure_clients.py (LLM reranking)
- config/settings.py (configuration)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import numpy as np

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
    """
    Reranks retrieved chunks using cross-encoder or LLM.
    
    Features:
    - Local cross-encoder reranking (fast, free)
    - Optional LLM reranking (expensive, high quality)
    - Score normalization
    - Batch processing
    """
    
    def __init__(
        self,
        method: str = RERANKING_METHOD,
        model_name: str = RERANKING_MODEL
    ):
        """
        Initialize reranking tools.
        
        Args:
            method: "cross_encoder" or "llm"
            model_name: Cross-encoder model name
        """
        self.method = method
        self.model_name = model_name
        self.cross_encoder = None  # Lazy load
        
        logger.info(f"RerankingTools initialized with method: {method}")
    

    def _load_cross_encoder(self):
        """
        Lazy load cross-encoder model.
        
        Supports both:
        - Local model path (for corporate networks)
        - HuggingFace download (for open networks)
        """
        if self.cross_encoder is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            from pathlib import Path
            
            # Check for local model first (corporate network fix)
            local_model_path = Path("models") / "ms-marco-MiniLM-L-6-v2"
            
            if local_model_path.exists() and local_model_path.is_dir():
                logger.info(f"✅ Found local model at: {local_model_path}")
                logger.info(f"📦 Loading cross-encoder from local path...")
                model_path = str(local_model_path)
            else:
                logger.info(f"⚠️  Local model not found at: {local_model_path.absolute()}")
                logger.info(f"🌐 Attempting download from HuggingFace: {self.model_name}")
                logger.info("⏳ First-time download may take ~1 minute...")
                model_path = self.model_name
            
            # Load model (from local or remote)
            self.cross_encoder = CrossEncoder(
                model_path,
                max_length=512,
                device='cpu'  # Use CPU (change to 'cuda' if GPU available)
            )
            
            logger.info(f"✅ Cross-encoder model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise ImportError(
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise


# Add this method to RerankingTools class:

    def rerank_with_fallback(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
        fallback_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Smart reranking with automatic LLM fallback.
        
        Falls back to LLM reranking if:
        1. Cross-encoder fails
        2. All rerank scores are below threshold (poor quality)
        3. Cross-encoder not available
        
        Args:
            query: User query
            chunks: Retrieved chunks
            top_k: Return top K
            fallback_threshold: If top score < this, use LLM fallback
            
        Returns:
            Dict with reranked chunks and metadata
        """
        logger.info(f"Smart reranking with fallback enabled")
        
        # Try cross-encoder first
        try:
            reranked = self.rerank(
                query=query,
                chunks=chunks,
                top_k=None,  # Get all scores first
                method="cross_encoder"
            )
            
            if not reranked:
                logger.warning("Cross-encoder returned no results. Falling back to LLM.")
                return self._fallback_to_llm(query, chunks, top_k, reason="no_results")
            
            # Check quality of top result
            top_score = reranked[0].get('rerank_score', 0)
            
            if top_score < fallback_threshold:
                logger.warning(
                    f"Cross-encoder top score ({top_score:.4f}) below threshold "
                    f"({fallback_threshold}). Falling back to LLM for better quality."
                )
                return self._fallback_to_llm(query, chunks, top_k, reason="low_quality")
            
            # Cross-encoder worked well
            logger.info(f"✅ Cross-encoder reranking successful (top score: {top_score:.4f})")
            return {
                'success': True,
                'reranked_chunks': reranked[:top_k],
                'method_used': 'cross_encoder',
                'top_score': top_score,
                'fallback_triggered': False
            }
            
        except Exception as e:
            logger.error(f"Cross-encoder failed: {e}. Falling back to LLM.")
            return self._fallback_to_llm(query, chunks, top_k, reason="error")


    def _fallback_to_llm(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        reason: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Fallback to LLM reranking.
        
        Args:
            query: User query
            chunks: Chunks to rerank
            top_k: Number to return
            reason: Why fallback was triggered
            
        Returns:
            Dict with LLM reranked results
        """
        logger.info(f"🔄 Falling back to LLM reranking (reason: {reason})")
        
        if not ENABLE_LLM_RERANKING_FALLBACK:
            logger.warning("LLM fallback disabled in config. Returning original chunks.")
            return {
                'success': False,
                'reranked_chunks': chunks[:top_k],
                'method_used': 'none',
                'fallback_triggered': True,
                'fallback_reason': reason,
                'error': 'LLM fallback disabled'
            }
        
        try:
            reranked = self.rerank(
                query=query,
                chunks=chunks,
                top_k=top_k,
                method="llm"
            )
            
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


    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        normalize_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using specified method.
        
        Args:
            query: User query
            chunks: Retrieved chunks to rerank
            top_k: Return top K after reranking (None = all)
            method: Override default method ("cross_encoder" or "llm")
            normalize_scores: Normalize scores to [0, 1]
            
        Returns:
            Reranked chunks with 'rerank_score' field
        """
        if not chunks:
            return []
        
        method = method or self.method
        
        logger.info(f"Reranking {len(chunks)} chunks using method: {method}")
        
        try:
            if method == "cross_encoder":
                reranked = self._rerank_with_cross_encoder(
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
                logger.warning(f"Unknown reranking method: {method}. Returning original order.")
                return chunks
            
            # Limit to top_k if specified
            if top_k is not None and top_k > 0:
                reranked = reranked[:top_k]
            
            logger.info(f"Reranking complete. Returning {len(reranked)} chunks.")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback: return original chunks
            logger.warning("Returning original chunks due to reranking failure")
            return chunks
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank using local cross-encoder model.
        
        Fast and free! No API calls.
        """
        # Lazy load model
        self._load_cross_encoder()
        
        # Prepare query-text pairs
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get scores (batch processing)
        logger.info(f"Computing cross-encoder scores for {len(pairs)} pairs...")
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=RERANKING_BATCH_SIZE,
            show_progress_bar=False
        )
        
        # Convert to list if numpy array
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        
        # Normalize scores if requested
        if normalize:
            scores = self._normalize_scores(scores, method=RERANKING_SCORE_NORMALIZATION)
        
        # Add scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)
        
        # Sort by rerank score (descending)
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Cross-encoder reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
        
        return reranked
    
    def _rerank_with_llm(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank using LLM (expensive but high quality).
        
        Uses Azure OpenAI to score relevance.
        """
        from utils.azure_clients import get_chat_completion
        
        logger.info("Using LLM reranking (this will incur API costs)")
        
        # Build reranking prompt
        chunks_text = "\n\n".join([
            f"[{i+1}] {chunk['text'][:300]}..."  # Truncate for token efficiency
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
            
            # Parse scores
            import json
            scores = json.loads(response.strip())
            
            if len(scores) != len(chunks):
                logger.warning(f"Score count mismatch: {len(scores)} vs {len(chunks)}. Using fallback.")
                return self._rerank_with_cross_encoder(query, chunks, normalize)
            
            # Normalize to [0, 1] if requested
            if normalize:
                scores = [s / 10.0 for s in scores]
            
            # Add scores to chunks
            for chunk, score in zip(chunks, scores):
                chunk['rerank_score'] = float(score)
            
            # Sort by score
            reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"LLM reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
            
            return reranked
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}. Falling back to cross-encoder.")
            return self._rerank_with_cross_encoder(query, chunks, normalize)
    
    def _normalize_scores(
        self,
        scores: List[float],
        method: str = "minmax"
    ) -> List[float]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: Raw scores
            method: "minmax", "sigmoid", or "none"
            
        Returns:
            Normalized scores
        """
        if method == "none":
            return scores
        
        scores = np.array(scores)
        
        if method == "minmax":
            # Min-max normalization
            min_score = scores.min()
            max_score = scores.max()
            
            if max_score - min_score == 0:
                # All scores are the same
                return [0.5] * len(scores)
            
            normalized = (scores - min_score) / (max_score - min_score)
            
        elif method == "sigmoid":
            # Sigmoid normalization (maps to 0-1)
            normalized = 1 / (1 + np.exp(-scores))
            
        else:
            logger.warning(f"Unknown normalization method: {method}. Using minmax.")
            return self._normalize_scores(scores, method="minmax")
        
        return normalized.tolist()
    
    def compare_methods(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare cross-encoder vs LLM reranking.
        
        Useful for evaluating which method works better for your use case.
        
        Args:
            query: Test query
            chunks: Chunks to rerank
            top_k: Number of results to compare
            
        Returns:
            Comparison results
        """
        logger.info("Comparing cross-encoder vs LLM reranking...")
        
        # Cross-encoder reranking
        ce_chunks = self._rerank_with_cross_encoder(
            query=query,
            chunks=[c.copy() for c in chunks],  # Deep copy
            normalize=True
        )[:top_k]
        
        # LLM reranking (if enabled)
        if ENABLE_LLM_RERANKING_FALLBACK:
            llm_chunks = self._rerank_with_llm(
                query=query,
                chunks=[c.copy() for c in chunks],  # Deep copy
                normalize=True
            )[:top_k]
        else:
            llm_chunks = None
        
        return {
            'query': query,
            'cross_encoder_results': [
                {
                    'rank': i + 1,
                    'text': c['text'][:100] + '...',
                    'score': c['rerank_score'],
                    'source': c['metadata'].get('source', 'unknown')
                }
                for i, c in enumerate(ce_chunks)
            ],
            'llm_results': [
                {
                    'rank': i + 1,
                    'text': c['text'][:100] + '...',
                    'score': c['rerank_score'],
                    'source': c['metadata'].get('source', 'unknown')
                }
                for i, c in enumerate(llm_chunks)
            ] if llm_chunks else None
        }
    
    def get_reranking_stats(self) -> Dict[str, Any]:
        """
        Get reranking configuration and statistics.
        
        Returns:
            Statistics dict
        """
        return {
            'method': self.method,
            'model': self.model_name if self.method == 'cross_encoder' else 'Azure OpenAI',
            'model_loaded': self.cross_encoder is not None,
            'batch_size': RERANKING_BATCH_SIZE,
            'normalization': RERANKING_SCORE_NORMALIZATION,
            'llm_fallback_enabled': ENABLE_LLM_RERANKING_FALLBACK
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def rerank_results(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function for quick reranking.
    
    Args:
        query: User query
        chunks: Retrieved chunks
        top_k: Return top K
        
    Returns:
        Reranked chunks
    """
    reranker = RerankingTools()
    return reranker.rerank(query=query, chunks=chunks, top_k=top_k)
