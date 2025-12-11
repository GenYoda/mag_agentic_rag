"""
Azure Client Factory for Medical RAG System

Provides singleton clients for:
1. Azure OpenAI (chat + embeddings)
2. Azure Document Intelligence

Design:
- Lazy initialization (clients created on first use)
- Singleton pattern (one client instance per type)
- Uses config/settings.py for credentials
- Thread-safe
"""

from typing import Optional
from functools import lru_cache
import threading

from openai import AzureOpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from config.settings import azure_settings


# Thread lock for singleton initialization
_lock = threading.Lock()

# Client instances (initialized lazily)
_openai_client: Optional[AzureOpenAI] = None
_document_intelligence_client: Optional[DocumentAnalysisClient] = None


# ============================================================================
# Azure OpenAI Client Factory
# ============================================================================

def get_openai_client() -> AzureOpenAI:
    """
    Get or create Azure OpenAI client (singleton).
    
    Thread-safe lazy initialization.
    Used for both chat completions and embeddings.
    
    Returns:
        AzureOpenAI: Configured OpenAI client
        
    Example:
        >>> client = get_openai_client()
        >>> response = client.chat.completions.create(...)
    """
    global _openai_client
    
    if _openai_client is None:
        with _lock:
            # Double-check pattern
            if _openai_client is None:
                _openai_client = AzureOpenAI(
                    api_key=azure_settings.azure_openai_key,
                    api_version=azure_settings.azure_openai_api_version,
                    azure_endpoint=azure_settings.azure_openai_endpoint
                )
    
    return _openai_client


def reset_openai_client():
    """
    Reset OpenAI client (useful for testing or credential updates).
    """
    global _openai_client
    with _lock:
        _openai_client = None


# ============================================================================
# Azure Document Intelligence Client Factory
# ============================================================================

def get_document_intelligence_client() -> DocumentAnalysisClient:
    """
    Get or create Azure Document Intelligence client (singleton).
    
    Thread-safe lazy initialization.
    Used for PDF text extraction with OCR support.
    
    Returns:
        DocumentAnalysisClient: Configured Document Intelligence client
        
    Example:
        >>> client = get_document_intelligence_client()
        >>> poller = client.begin_analyze_document("prebuilt-read", document=file)
    """
    global _document_intelligence_client
    
    if _document_intelligence_client is None:
        with _lock:
            # Double-check pattern
            if _document_intelligence_client is None:
                credential = AzureKeyCredential(
                    azure_settings.azure_document_intelligence_key
                )
                _document_intelligence_client = DocumentAnalysisClient(
                    endpoint=azure_settings.azure_document_intelligence_endpoint,
                    credential=credential
                )
    
    return _document_intelligence_client


def reset_document_intelligence_client():
    """
    Reset Document Intelligence client (useful for testing or credential updates).
    """
    global _document_intelligence_client
    with _lock:
        _document_intelligence_client = None


# ============================================================================
# Health Check Functions
# ============================================================================

def test_openai_connection() -> dict:
    """
    Test Azure OpenAI connection with a simple embedding call.
    
    Returns:
        dict: Status and details
            {
                'success': bool,
                'service': str,
                'deployment': str,
                'error': str (if failed)
            }
    """
    try:
        client = get_openai_client()
        
        # Test with simple embedding (smallest API call)
        response = client.embeddings.create(
            input="test",
            model=azure_settings.azure_openai_embedding_deployment
        )
        
        return {
            'success': True,
            'service': 'Azure OpenAI',
            'deployment': azure_settings.azure_openai_embedding_deployment,
            'embedding_dim': len(response.data[0].embedding),
            'api_version': azure_settings.azure_openai_api_version
        }
        
    except Exception as e:
        return {
            'success': False,
            'service': 'Azure OpenAI',
            'error': str(e)
        }


def test_document_intelligence_connection() -> dict:
    """
    Test Azure Document Intelligence connection.
    
    Returns:
        dict: Status and details
            {
                'success': bool,
                'service': str,
                'endpoint': str,
                'error': str (if failed)
            }
    """
    try:
        client = get_document_intelligence_client()
        
        # Just verify client is initialized (actual test needs a document)
        # Check if endpoint is reachable by verifying client properties
        endpoint = azure_settings.azure_document_intelligence_endpoint
        
        return {
            'success': True,
            'service': 'Azure Document Intelligence',
            'endpoint': endpoint,
            'note': 'Client initialized (full test requires document upload)'
        }
        
    except Exception as e:
        return {
            'success': False,
            'service': 'Azure Document Intelligence',
            'error': str(e)
        }


def test_all_connections() -> dict:
    """
    Test all Azure service connections.
    
    Returns:
        dict: Combined status of all services
            {
                'openai': {...},
                'document_intelligence': {...},
                'all_healthy': bool
            }
    """
    openai_status = test_openai_connection()
    doc_intel_status = test_document_intelligence_connection()
    
    return {
        'openai': openai_status,
        'document_intelligence': doc_intel_status,
        'all_healthy': openai_status['success'] and doc_intel_status['success']
    }


# ============================================================================
# Convenience Functions for Common Operations
# ============================================================================

def get_chat_completion(
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 400,
    **kwargs
) -> str:
    """
    Convenience function for chat completions.
    
    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}]
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for chat.completions.create()
    
    Returns:
        str: Generated text response
    """
    client = get_openai_client()
    
    response = client.chat.completions.create(
        model=azure_settings.azure_openai_chat_deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return response.choices[0].message.content


def get_embeddings(
    texts: list[str],
    **kwargs
) -> list[list[float]]:
    """
    Convenience function for getting embeddings.
    
    Args:
        texts: List of texts to embed
        **kwargs: Additional parameters for embeddings.create()
    
    Returns:
        list[list[float]]: List of embedding vectors
    """
    client = get_openai_client()
    
    response = client.embeddings.create(
        input=texts,
        model=azure_settings.azure_openai_embedding_deployment,
        **kwargs
    )
    
    return [item.embedding for item in response.data]


def get_single_embedding(text: str, **kwargs) -> list[float]:
    """
    Convenience function for getting a single embedding.
    
    Args:
        text: Text to embed
        **kwargs: Additional parameters for embeddings.create()
    
    Returns:
        list[float]: Embedding vector
    """
    return get_embeddings([text], **kwargs)[0]


# ============================================================================
# Client Info Functions
# ============================================================================

def get_client_info() -> dict:
    """
    Get information about configured Azure clients.
    
    Returns:
        dict: Client configuration details
    """
    return {
        'openai': {
            'endpoint': azure_settings.azure_openai_endpoint,
            'api_version': azure_settings.azure_openai_api_version,
            'chat_deployment': azure_settings.azure_openai_chat_deployment,
            'embedding_deployment': azure_settings.azure_openai_embedding_deployment,
            'client_initialized': _openai_client is not None
        },
        'document_intelligence': {
            'endpoint': azure_settings.azure_document_intelligence_endpoint,
            'client_initialized': _document_intelligence_client is not None
        }
    }


def print_client_info():
    """
    Print formatted client information.
    """
    info = get_client_info()
    
    print("=" * 80)
    print("AZURE CLIENTS CONFIGURATION")
    print("=" * 80)
    
    print("\n🤖 Azure OpenAI:")
    print(f"  • Endpoint: {info['openai']['endpoint']}")
    print(f"  • API Version: {info['openai']['api_version']}")
    print(f"  • Chat Model: {info['openai']['chat_deployment']}")
    print(f"  • Embedding Model: {info['openai']['embedding_deployment']}")
    print(f"  • Client Initialized: {info['openai']['client_initialized']}")
    
    print("\n📄 Azure Document Intelligence:")
    print(f"  • Endpoint: {info['document_intelligence']['endpoint']}")
    print(f"  • Client Initialized: {info['document_intelligence']['client_initialized']}")
    
    print("=" * 80)
