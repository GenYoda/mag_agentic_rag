"""
Embeddings Module - Text-to-Vector Conversion

Handles embedding generation with:
- Azure OpenAI text-embedding-3-large (3072 dimensions)
- Retry logic with exponential backoff
- Text length validation and truncation
- Batch processing optimization
- Dimension validation (ADDED)

Integrates:
- utils/azure_clients.py for OpenAI client
- config/settings.py for configuration
"""

from typing import List, Optional
import time
import logging

from utils.azure_clients import get_openai_client
from config.settings import (
    azure_settings,
    EMBEDDING_DIMENSION,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_RETRY_DELAY
)

# Default value if not in settings
EMBEDDING_MAX_TOKENS = 8191  # Max tokens for text-embedding-3-large


logger = logging.getLogger(__name__)


def check_text_length(text: str, max_tokens: int = EMBEDDING_MAX_TOKENS) -> tuple[bool, int]:
    """
    Check if text is within token limit.
    Simple heuristic: ~4 chars per token.
    """
    estimated_tokens = len(text) // 4
    return estimated_tokens <= max_tokens, estimated_tokens


def truncate_text(text: str, max_tokens: int = EMBEDDING_MAX_TOKENS) -> str:
    """Truncate text to fit within token limit."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def generate_embedding(
    text: str,
    model: Optional[str] = None,
    max_retries: int = EMBEDDING_MAX_RETRIES,
    truncate_if_needed: bool = True
) -> Optional[List[float]]:
    """
    Generate embedding for a single text with retry logic and dimension validation.
    
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
            
            embedding = response.data[0].embedding
            
            # Validate dimension matches configuration
            if len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding dimension mismatch! "
                    f"Expected: {EMBEDDING_DIMENSION}, Got: {len(embedding)}. "
                    f"Model '{model}' returns {len(embedding)} dimensions. "
                    f"Update EMBEDDING_DIMENSION in config/settings.py to match your model."
                )
            
            return embedding
            
        except ValueError:
            # Re-raise dimension validation errors immediately (don't retry)
            raise
            
        except Exception as e:
            last_error = e
            
            if attempt < max_retries - 1:
                wait_time = EMBEDDING_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Embedding generation failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to generate embedding after {max_retries} attempts: {e}")
    
    return None


def generate_batch_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 32,
    max_retries: int = EMBEDDING_MAX_RETRIES,
    truncate_if_needed: bool = True
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches with dimension validation.
    
    Args:
        texts: List of texts to embed
        model: Embedding model (default: from config)
        batch_size: Number of texts per API call
        max_retries: Maximum retry attempts per batch
        truncate_if_needed: Truncate texts if too long
        
    Returns:
        List of embedding vectors
        
    Example:
        >>> texts = ["First document", "Second document"]
        >>> embeddings = generate_batch_embeddings(texts)
        >>> len(embeddings)
        2
    """
    if not texts:
        return []
    
    model = model or azure_settings.azure_openai_embedding_deployment
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Truncate if needed
        if truncate_if_needed:
            batch = [truncate_text(text) if text else "" for text in batch]
        
        # Retry loop for this batch
        last_error = None
        for attempt in range(max_retries):
            try:
                client = get_openai_client()
                
                response = client.embeddings.create(
                    input=batch,
                    model=model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                
                # Validate first embedding dimension (all should match)
                if batch_embeddings and len(batch_embeddings[0]) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Embedding dimension mismatch! "
                        f"Expected: {EMBEDDING_DIMENSION}, Got: {len(batch_embeddings[0])}. "
                        f"Model '{model}' returns {len(batch_embeddings[0])} dimensions. "
                        f"Update EMBEDDING_DIMENSION in config/settings.py to match your model."
                    )
                
                all_embeddings.extend(batch_embeddings)
                break  # Success - exit retry loop
                
            except ValueError:
                # Re-raise dimension validation errors immediately
                raise
                
            except Exception as e:
                last_error = e
                
                if attempt < max_retries - 1:
                    wait_time = EMBEDDING_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Batch embedding failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate batch embeddings after {max_retries} attempts: {e}")
                    # Return partial results
                    return all_embeddings
    
    return all_embeddings


def get_embedding_dimension() -> int:
    """Get configured embedding dimension."""
    return EMBEDDING_DIMENSION


def validate_embedding(embedding: List[float]) -> bool:
    """
    Validate that embedding has correct dimension.
    
    Args:
        embedding: Embedding vector to validate
        
    Returns:
        bool: True if valid dimension
    """
    return len(embedding) == EMBEDDING_DIMENSION


# Convenience functions
def embed_query(query: str) -> Optional[List[float]]:
    """Convenience function to embed a query."""
    return generate_embedding(query)


def embed_documents(documents: List[str], batch_size: int = 32) -> List[List[float]]:
    """Convenience function to embed multiple documents."""
    return generate_batch_embeddings(documents, batch_size=batch_size)


# Alias for backward compatibility
def generate_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 32,
    max_retries: int = EMBEDDING_MAX_RETRIES,
    truncate_if_needed: bool = True
) -> List[List[float]]:
    """
    Alias for generate_batch_embeddings.
    Maintains backward compatibility with different naming convention.
    """
    return generate_batch_embeddings(
        texts=texts,
        model=model,
        batch_size=batch_size,
        max_retries=max_retries,
        truncate_if_needed=truncate_if_needed
    )
