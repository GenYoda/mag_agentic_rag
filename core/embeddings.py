"""
Embedding Generation for Agentic Medical RAG

Handles:
- Batch embedding generation using Azure OpenAI
- Token estimation and management
- Retry logic with exponential backoff
- Rate limiting awareness
- Embedding caching (optional)

Uses text-embedding-3-large (3072 dimensions) as configured.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from utils.azure_clients import get_openai_client, get_single_embedding
from config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_DELAY,
    EMBEDDING_DIMENSION,
    azure_settings
)


# ============================================================================
# SECTION 1: Token Estimation
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text.
    
    Rough approximation: 1 token ≈ 4 characters for English text.
    This is conservative - actual token count may be lower.
    
    Args:
        text: Input text
        
    Returns:
        int: Estimated token count
        
    Example:
        >>> estimate_tokens("Hello, world!")
        3
    """
    # Simple approximation: ~4 chars per token
    return len(text) // 4


def check_text_length(text: str, max_tokens: int = 8191) -> Tuple[bool, int]:
    """
    Check if text exceeds token limit for embedding model.
    
    Azure OpenAI embedding models have token limits:
    - text-embedding-3-small/large: 8191 tokens
    
    Args:
        text: Input text
        max_tokens: Maximum allowed tokens (default: 8191)
        
    Returns:
        tuple: (is_valid, estimated_tokens)
        
    Example:
        >>> is_valid, tokens = check_text_length("Short text")
        >>> print(is_valid)
        True
    """
    estimated = estimate_tokens(text)
    return (estimated <= max_tokens, estimated)


def truncate_text(text: str, max_tokens: int = 8191) -> str:
    """
    Truncate text to fit within token limit.
    
    Conservative truncation based on character count.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens allowed
        
    Returns:
        str: Truncated text
    """
    max_chars = max_tokens * 4  # Conservative estimate
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ============================================================================
# SECTION 2: Single Embedding Generation
# ============================================================================

def generate_embedding(
    text: str,
    model: Optional[str] = None,
    max_retries: int = EMBEDDING_MAX_RETRIES,
    truncate_if_needed: bool = True
) -> Optional[List[float]]:
    """
    Generate embedding for a single text with retry logic.
    
    Args:
        text: Text to embed
        model: Embedding model (default: from config)
        max_retries: Maximum retry attempts
        truncate_if_needed: Truncate text if too long (default: True)
        
    Returns:
        List[float]: Embedding vector, or None if failed
        
    Example:
        >>> embedding = generate_embedding("Patient diagnosed with diabetes")
        >>> len(embedding)
        3072  # text-embedding-3-large dimension
    """
    if not text or not text.strip():
        return None
    
    model = model or azure_settings.azure_openai_embedding_deployment
    
    # Check/truncate text length
    is_valid, tokens = check_text_length(text)
    if not is_valid:
        if truncate_if_needed:
            text = truncate_text(text)
        else:
            raise ValueError(f"Text too long: {tokens} tokens (max: 8191)")
    
    # Retry loop
    last_error = None
    for attempt in range(max_retries):
        try:
            client = get_openai_client()
            
            response = client.embeddings.create(
                input=text,
                model=model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            last_error = e
            
            if attempt < max_retries - 1:
                wait_time = EMBEDDING_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
            else:
                print(f"⚠️  Failed to generate embedding after {max_retries} attempts: {e}")
    
    return None


# ============================================================================
# SECTION 3: Batch Embedding Generation
# ============================================================================

def generate_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
    truncate_if_needed: bool = True
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.
    
    Processes texts in batches to:
    - Respect API rate limits
    - Show progress for large datasets
    - Handle failures gracefully
    
    Args:
        texts: List of texts to embed
        model: Embedding model (default: from config)
        batch_size: Texts per batch (default: 100)
        show_progress: Show progress bar (default: True)
        truncate_if_needed: Truncate long texts (default: True)
        
    Returns:
        List[List[float]]: List of embedding vectors
        
    Example:
        >>> texts = ["Text 1", "Text 2", "Text 3"]
        >>> embeddings = generate_embeddings_batch(texts)
        >>> len(embeddings) == len(texts)
        True
    """
    if not texts:
        return []
    
    model = model or azure_settings.azure_openai_embedding_deployment
    client = get_openai_client()
    
    all_embeddings = []
    failed_indices = []
    
    # Process in batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=total_batches, desc="Generating embeddings")
    
    for i in iterator:
        batch = texts[i:i + batch_size]
        
        # Filter empty texts
        batch_with_indices = [(idx, text) for idx, text in enumerate(batch, start=i) if text and text.strip()]
        
        if not batch_with_indices:
            # All texts in batch are empty
            all_embeddings.extend([None] * len(batch))
            continue
        
        indices, valid_texts = zip(*batch_with_indices)
        
        # Truncate if needed
        if truncate_if_needed:
            valid_texts = [truncate_text(t) for t in valid_texts]
        
        # Generate embeddings for batch
        try:
            response = client.embeddings.create(
                input=list(valid_texts),
                model=model
            )
            
            # Map embeddings back to original positions
            embeddings_dict = {idx: response.data[j].embedding for j, idx in enumerate(indices)}
            
            # Fill in results (None for empty texts)
            for idx in range(i, i + len(batch)):
                if idx in embeddings_dict:
                    all_embeddings.append(embeddings_dict[idx])
                else:
                    all_embeddings.append(None)
                    failed_indices.append(idx)
            
        except Exception as e:
            print(f"\n⚠️  Batch {i//batch_size + 1} failed: {e}")
            # Add None for all texts in failed batch
            all_embeddings.extend([None] * len(batch))
            failed_indices.extend(range(i, i + len(batch)))
    
    if failed_indices:
        print(f"\n⚠️  Failed to generate embeddings for {len(failed_indices)} texts")
    
    return all_embeddings


# ============================================================================
# SECTION 4: Embedding Validation
# ============================================================================

def validate_embedding(embedding: Optional[List[float]]) -> bool:
    """
    Validate embedding vector.
    
    Checks:
    - Not None
    - Correct dimension
    - Contains valid numbers (no NaN/Inf)
    
    Args:
        embedding: Embedding vector to validate
        
    Returns:
        bool: True if valid
    """
    if embedding is None:
        return False
    
    if len(embedding) != EMBEDDING_DIMENSION:
        return False
    
    # Check for NaN or Inf
    arr = np.array(embedding)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False
    
    return True


def validate_embeddings_batch(embeddings: List[Optional[List[float]]]) -> Dict[str, Any]:
    """
    Validate batch of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        dict: Validation report
            {
                'total': int,
                'valid': int,
                'invalid': int,
                'invalid_indices': List[int],
                'all_valid': bool
            }
    """
    total = len(embeddings)
    invalid_indices = []
    
    for i, emb in enumerate(embeddings):
        if not validate_embedding(emb):
            invalid_indices.append(i)
    
    valid = total - len(invalid_indices)
    
    return {
        'total': total,
        'valid': valid,
        'invalid': len(invalid_indices),
        'invalid_indices': invalid_indices,
        'all_valid': len(invalid_indices) == 0
    }


# ============================================================================
# SECTION 5: Embedding Statistics
# ============================================================================

def calculate_embedding_stats(embeddings: List[List[float]]) -> Dict[str, Any]:
    """
    Calculate statistics for embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        dict: Statistics
            {
                'count': int,
                'dimension': int,
                'mean_norm': float,
                'std_norm': float,
                'min_norm': float,
                'max_norm': float
            }
    """
    if not embeddings:
        return {
            'count': 0,
            'dimension': 0,
            'mean_norm': 0.0,
            'std_norm': 0.0,
            'min_norm': 0.0,
            'max_norm': 0.0
        }
    
    # Filter valid embeddings
    valid_embeddings = [e for e in embeddings if e is not None]
    
    if not valid_embeddings:
        return {
            'count': 0,
            'dimension': 0,
            'mean_norm': 0.0,
            'std_norm': 0.0,
            'min_norm': 0.0,
            'max_norm': 0.0
        }
    
    # Calculate norms
    arr = np.array(valid_embeddings)
    norms = np.linalg.norm(arr, axis=1)
    
    return {
        'count': len(valid_embeddings),
        'dimension': len(valid_embeddings[0]),
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'min_norm': float(np.min(norms)),
        'max_norm': float(np.max(norms))
    }


# ============================================================================
# SECTION 6: Convenience Functions
# ============================================================================

def embed_chunks(
    chunks: List[Dict[str, Any]],
    text_key: str = 'text',
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Add embeddings to chunk dictionaries.
    
    Args:
        chunks: List of chunk dicts (must have text_key)
        text_key: Key containing text to embed (default: 'text')
        batch_size: Batch size for embedding generation
        show_progress: Show progress bar
        
    Returns:
        List[Dict]: Chunks with 'embedding' key added
        
    Example:
        >>> chunks = [{'text': 'Sample text', 'metadata': {...}}]
        >>> chunks_with_embeddings = embed_chunks(chunks)
        >>> 'embedding' in chunks_with_embeddings[0]
        True
    """
    # Extract texts
    texts = [chunk.get(text_key, '') for chunk in chunks]
    
    # Generate embeddings
    embeddings = generate_embeddings_batch(
        texts,
        batch_size=batch_size,
        show_progress=show_progress
    )
    
    # Add embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i]
    
    return chunks


def get_embedding_info() -> Dict[str, Any]:
    """
    Get information about configured embedding model.
    
    Returns:
        dict: Model information
    """
    return {
        'model': azure_settings.azure_openai_embedding_deployment,
        'dimension': EMBEDDING_DIMENSION,
        'batch_size': EMBEDDING_BATCH_SIZE,
        'max_retries': EMBEDDING_MAX_RETRIES,
        'endpoint': azure_settings.azure_openai_endpoint
    }


def print_embedding_info():
    """Print formatted embedding configuration."""
    info = get_embedding_info()
    
    print("\n" + "=" * 80)
    print("EMBEDDING CONFIGURATION")
    print("=" * 80)
    print(f"Model: {info['model']}")
    print(f"Dimension: {info['dimension']}")
    print(f"Batch Size: {info['batch_size']}")
    print(f"Max Retries: {info['max_retries']}")
    print(f"Endpoint: {info['endpoint']}")
    print("=" * 80 + "\n")
