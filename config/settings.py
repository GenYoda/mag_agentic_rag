"""
Configuration and Settings for Agentic Medical RAG System

Design:
- Pydantic Settings loads ONLY secrets/keys from .env (7 variables)
- All other settings are Python constants (toggles, paths, configs)
- Matches legacy config.py pattern while adding type safety
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal


# ============================================================================
# SECTION 1: Azure Credentials & Endpoints (from .env)
# ============================================================================

class AzureSettings(BaseSettings):
    """
    Load Azure credentials from environment variables.
    These are the ONLY settings that come from .env
    """
    
    # Azure Document Intelligence
    azure_document_intelligence_endpoint: str = Field(
        ...,
        alias="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        description="Azure Document Intelligence endpoint URL"
    )
    azure_document_intelligence_key: str = Field(
        ...,
        alias="AZURE_DOCUMENT_INTELLIGENCE_KEY",
        description="Azure Document Intelligence API key"
    )
    
    # Azure OpenAI
    azure_openai_endpoint: str = Field(
        ...,
        alias="AZURE_OPENAI_ENDPOINT",
        description="Azure OpenAI endpoint URL"
    )
    azure_openai_key: str = Field(
        ...,
        alias="AZURE_OPENAI_KEY",
        description="Azure OpenAI API key"
    )
    azure_openai_api_version: str = Field(
        ...,
        alias="AZURE_OPENAI_API_VERSION",
        description="Azure OpenAI API version"
    )
    azure_openai_embedding_deployment: str = Field(
        ...,
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        description="Azure OpenAI embedding model deployment name"
    )
    azure_openai_chat_deployment: str = Field(
        ...,
        alias="AZURE_OPENAI_CHAT_DEPLOYMENT",
        description="Azure OpenAI chat model deployment name"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
azure_settings = AzureSettings()


# ============================================================================
# SECTION 2: Paths & Directories
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Data subdirectories
INPUT_FOLDER = DATA_DIR / "input"
INDEX_PATH = DATA_DIR / "knowledge_base"
CACHE_PATH = DATA_DIR / "cache"
MEMORY_PATH = DATA_DIR / "memory"

# KB specific paths
FAISS_INDEX_FILE = INDEX_PATH / "faiss_index.bin"
CHUNKS_FILE = INDEX_PATH / "chunks.json"
METADATA_FILE = INDEX_PATH / "metadata.json"
TRACKER_FILE = INDEX_PATH / "pdf_tracker.json"

# Cache specific paths
CACHE_INDEX_FILE = CACHE_PATH / "cache_index.bin"
CACHE_DATA_FILE = CACHE_PATH / "cache_data.json"


# Memory specific paths
MEMORY_DB_FILE = MEMORY_PATH / "conversation_memory.db"
ENTITY_MEMORY_FILE = MEMORY_PATH / "entity_memory.json"


# ============================================================================
# SECTION 3: PDF Processing & Chunking
# ============================================================================

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50  # Skip chunks smaller than this

# Azure Document Intelligence settings
DOC_INTEL_MODEL = "prebuilt-read"  # For medical documents with handwriting
DOC_INTEL_PAGES = None  # Process all pages
DOC_INTEL_LOCALE = "en-US"


# ============================================================================
# SECTION 4: Embeddings
# ============================================================================

EMBEDDING_DIMENSION = 3072  # text-embedding-3-small
EMBEDDING_BATCH_SIZE = 100  # Batch size for embedding generation
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_DELAY = 2  # seconds


# ============================================================================
# SECTION 5: Knowledge Base Settings
# ============================================================================

# Deduplication
ENABLE_PDF_DEDUPLICATION = True
ENABLE_CHUNK_DEDUPLICATION = True

# Autosync
ENABLE_AUTOSYNC = True
AUTOSYNC_ON_STARTUP = False  # Set to True for automatic sync on app start


# ============================================================================
# SECTION 6: Retrieval Settings
# ============================================================================

DEFAULT_TOP_K = 5
RETRIEVAL_BUFFER = 10  # Retrieve more, then filter/rerank
MAX_DISTANCE_THRESHOLD = 1.5  # FAISS L2 distance threshold
MIN_CHUNKS_RETURNED = 3  # Always return at least this many

# Multi-strategy retrieval
ENABLE_MULTI_STRATEGY = True
RECIPROCAL_RANK_K = 60  # RRF constant


# ============================================================================
# SECTION 7: Query Enhancement Settings
# ============================================================================

ENABLE_QUERY_ENHANCEMENT = True
ENABLE_QUERY_CLASSIFICATION = True

# HyDE settings
ENABLE_HYDE = True
HYDE_FOR_SIMPLE_QUERIES = False  # Only use for complex/high-level queries

# Query variations
ENABLE_QUERY_VARIATIONS = True
MAX_QUERY_VARIATIONS = 3


# ============================================================================
# SECTION 8: Reranking Settings
# ============================================================================

ENABLE_RERANKING = True
RERANKING_METHOD: Literal["crossencoder", "llm_simple", "llm_detailed", "llm_pairwise"] = "llm_simple"
RERANKING_TOP_K = 5

# Cross-encoder model (if using crossencoder method)
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================================
# SECTION 9: LLM Answer Generation Settings
# ============================================================================

# LLM Configuration (PRESERVED from legacy)
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 400
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 2


# ============================================================================
# SECTION 10: Semantic Cache Settings
# ============================================================================

ENABLE_SEMANTIC_CACHE = True
CACHE_SIMILARITY_THRESHOLD = 0.95  # Must be very similar to return cached answer
CACHE_TTL_HOURS = 24  # Cache entry expires after 24 hours
CACHE_MAX_ENTRIES = 1000  # Maximum cache size


# ============================================================================
# SECTION 11: Memory Settings
# ============================================================================

# Short-term memory (sliding window)
SHORT_TERM_WINDOW_SIZE = 5  # Last N exchanges

# Long-term memory
ENABLE_LONG_TERM_MEMORY = True
LONG_TERM_SUMMARY_TRIGGER = 10  # Summarize after N exchanges

# Entity memory
ENABLE_ENTITY_MEMORY = True

# Semantic memory (FAISS-based conversation search)
ENABLE_SEMANTIC_MEMORY = True
SEMANTIC_MEMORY_TOP_K = 3


# ============================================================================
# SECTION 12: Validation & Self-Healing Settings
# ============================================================================

ENABLE_VALIDATION = True
ENABLE_SELF_HEALING = True
MAX_SELF_HEALING_ITERATIONS = 2

# Confidence scoring
CONFIDENCE_THRESHOLD_HIGH = 0.8
CONFIDENCE_THRESHOLD_MEDIUM = 0.6
CONFIDENCE_THRESHOLD_LOW = 0.4


# ============================================================================
# SECTION 13: Agentic Mode vs Classic Mode
# ============================================================================

# Feature flag for agentic system
ENABLE_AGENTIC_MODE = True  # Set to False to use legacy pipeline

# CrewAI settings
CREW_VERBOSE = True
CREW_MEMORY = True  # CrewAI's built-in memory
CREW_MAX_RPM = 60  # Rate limit


# ============================================================================
# SECTION 14: Logging
# ============================================================================

LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE_PATH = BASE_DIR / "logs" / "medical_rag.log"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"


# ============================================================================
# SECTION 15: Helper Functions
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        INPUT_FOLDER,
        INDEX_PATH,
        CACHE_PATH,
        MEMORY_PATH,
        BASE_DIR / "logs"
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_kb_paths():
    """Return all KB-related paths as a dict"""
    return {
        "index": FAISS_INDEX_FILE,
        "chunks": CHUNKS_FILE,
        "metadata": METADATA_FILE,
        "tracker": TRACKER_FILE
    }


def get_cache_paths():
    """Return all cache-related paths as a dict"""
    return {
        "index": CACHE_INDEX_FILE,
        "data": CACHE_DATA_FILE
    }


def get_memory_paths():
    """Return all memory-related paths as a dict"""
    return {
        "db": MEMORY_DB_FILE,
        "entities": ENTITY_MEMORY_FILE
    }


def print_config_summary():
    """Print configuration summary for debugging"""
    print("=" * 80)
    print("AGENTIC MEDICAL RAG - CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\n🔐 Azure Settings:")
    print(f"  ✓ Document Intelligence: {azure_settings.azure_document_intelligence_endpoint}")
    print(f"  ✓ OpenAI Endpoint: {azure_settings.azure_openai_endpoint}")
    print(f"  ✓ Chat Deployment: {azure_settings.azure_openai_chat_deployment}")
    print(f"  ✓ Embedding Deployment: {azure_settings.azure_openai_embedding_deployment}")
    print(f"  ✓ API Version: {azure_settings.azure_openai_api_version}")
    
    print(f"\n📁 Paths:")
    print(f"  • Input Folder: {INPUT_FOLDER}")
    print(f"  • Knowledge Base: {INDEX_PATH}")
    print(f"  • Cache: {CACHE_PATH}")
    print(f"  • Memory: {MEMORY_PATH}")
    
    print(f"\n⚙️  Feature Flags:")
    print(f"  • Agentic Mode: {ENABLE_AGENTIC_MODE}")
    print(f"  • Semantic Cache: {ENABLE_SEMANTIC_CACHE}")
    print(f"  • Query Enhancement: {ENABLE_QUERY_ENHANCEMENT}")
    print(f"  • Reranking: {ENABLE_RERANKING} ({RERANKING_METHOD})")
    print(f"  • Validation: {ENABLE_VALIDATION}")
    print(f"  • Self-Healing: {ENABLE_SELF_HEALING}")
    
    print(f"\n🔧 Core Settings:")
    print(f"  • Chunk Size: {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print(f"  • Top-K: {DEFAULT_TOP_K}")
    print(f"  • LLM Temperature: {LLM_TEMPERATURE}")
    print(f"  • Cache Threshold: {CACHE_SIMILARITY_THRESHOLD}")
    print("=" * 80)


# ============================================================================
# SECTION 16: Initialization
# ============================================================================

# Auto-create directories on import
ensure_directories()
