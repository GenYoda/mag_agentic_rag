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
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

EMBEDDING_DIMENSION = 3072  # text-embedding-3-large
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
# SECTION 7: Reranking Settings
# ============================================================================

ENABLE_RERANKING = True

# Reranking method: "cross_encoder" (free, local) or "llm" (expensive, cloud)
RERANKING_METHOD: Literal["cross_encoder", "llm"] = "cross_encoder"
RERANKING_TOP_K = 5

# Cross-encoder model (local, free, ~900MB)
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Batch processing settings
RERANKING_BATCH_SIZE = 32

# Score normalization method
RERANKING_SCORE_NORMALIZATION: Literal["minmax", "sigmoid", "none"] = "minmax"

# Smart reranking with LLM fallback
ENABLE_LLM_RERANKING_FALLBACK = True
RERANKING_FALLBACK_THRESHOLD = 0.5  # If score < this, fallback to LLM

# LLM reranking settings (if using "llm" method or fallback)
LLM_RERANKING_TEMPERATURE = 0.1
LLM_RERANKING_MAX_TOKENS = 200

# Model caching directory (None = use default)
CROSS_ENCODER_CACHE_DIR = None

# ============================================================================
# SECTION 8: Query Enhancement Settings
# ============================================================================

ENABLE_QUERY_ENHANCEMENT = True

# Query Classification
ENABLE_QUERY_CLASSIFICATION = True

# Query Decomposition (for complex multi-part queries)
ENABLE_QUERY_DECOMPOSITION = True
MAX_SUBQUERIES = 3  # Maximum sub-queries from decomposition

# HyDE (Hypothetical Document Embeddings)
ENABLE_HYDE = True
HYDE_FOR_SIMPLE_QUERIES = False  # Only use HyDE for complex queries
HYDE_TEMPERATURE = 0.4  # Higher = more creative hypothetical answers
HYDE_MAX_TOKENS = 200

# Query Variations (alternative phrasings)
ENABLE_QUERY_VARIATIONS = True
MAX_QUERY_VARIATIONS = 3  # Number of variations to generate
QUERY_VARIATION_METHOD: Literal["llm", "template"] = "llm"  # "llm" or "template"

# Query enhancement LLM settings
QUERY_ENHANCEMENT_TEMPERATURE = 0.3
QUERY_ENHANCEMENT_MAX_TOKENS = 300

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
CACHE_SIMILARITY_THRESHOLD = 0.90  # Must be very similar to return cached answer
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

# Validation
ENABLE_VALIDATION = True

# ✅ NEW: Self-Healing Configuration
ENABLE_SELF_HEALING = True
SELF_HEAL_MAX_RETRIES = 2  # Default: 2 retries before returning best attempt
SELF_HEAL_MIN_QUALITY_SCORE = 0.80  # Threshold to trigger healing (0.0-1.0)

# Self-healing strategies (in order of preference)
SELF_HEAL_STRATEGIES = [
    "regenerate_with_emphasis",  # Add stronger instructions for citations/grounding
    "retrieve_more_context",      # Get additional chunks if incomplete
    "rephrase_query",             # Try different query formulation
    "rerank_context"              # Rerank with different method
]

# Track best attempt (return highest scoring if all retries fail)
SELF_HEAL_TRACK_BEST_ATTEMPT = True

# Confidence scoring (legacy - kept for compatibility)
CONFIDENCE_THRESHOLD_HIGH = 0.8
CONFIDENCE_THRESHOLD_MEDIUM = 0.6
CONFIDENCE_THRESHOLD_LOW = 0.4

# ============================================================================
# SECTION 13: LLM-as-Judge Configuration (Validation)
# ============================================================================

# Enable LLM-based validation
ENABLE_LLM_JUDGE = True

# ✅ UPDATED: LLM Judge Configuration (pluggable design)
# Validation LLM provider: "azure_openai", "gemini", "openai", "claude"
VALIDATION_LLM_PROVIDER = os.getenv("VALIDATION_LLM_PROVIDER", "azure_openai")

# For Azure OpenAI: Use deployment name from .env
VALIDATION_LLM_DEPLOYMENT = os.getenv("VALIDATION_LLM_DEPLOYMENT", "gpt-4o-mini")

# Use existing Azure OpenAI credentials (or separate ones from .env)
VALIDATION_LLM_API_KEY = os.getenv("VALIDATION_LLM_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
VALIDATION_LLM_ENDPOINT = os.getenv("VALIDATION_LLM_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
VALIDATION_LLM_API_VERSION = os.getenv("VALIDATION_LLM_API_VERSION", "2024-12-01-preview")

# For Gemini (if using Gemini as judge)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set in .env if using Gemini
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")

# For Claude (if using Claude as judge)
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")  # Set in .env if using Claude
CLAUDE_MODEL_NAME = os.getenv("CLAUDE_MODEL_NAME", "claude-3-opus-20240229")

# Fallback to rule-based if LLM fails
VALIDATION_FALLBACK_TO_RULES = True

# Validation thresholds
MIN_ANSWER_LENGTH = 50
MAX_ANSWER_LENGTH = 2000
MIN_CITATIONS = 1
MIN_QUALITY_SCORE = 0.6

# PII Detection (DISABLED - returns unfiltered answers)
ENABLE_PII_DETECTION = False

# Strict mode
VALIDATION_STRICT_MODE = False

# ============================================================================
# SECTION 14: Agentic Mode vs Classic Mode
# ============================================================================

# Feature flag for agentic system
ENABLE_AGENTIC_MODE = True  # Set to False to use legacy pipeline

# CrewAI settings
CREW_VERBOSE = True
CREW_MEMORY = True  # CrewAI's built-in memory
CREW_MAX_RPM = 60  # Rate limit

# ============================================================================
# SECTION 15: Logging
# ============================================================================

LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE_PATH = BASE_DIR / "logs" / "medical_rag.log"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"

# ============================================================================
# SECTION 16: Helper Functions
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
    print(f"   ✓ Document Intelligence: {azure_settings.azure_document_intelligence_endpoint}")
    print(f"   ✓ OpenAI Endpoint: {azure_settings.azure_openai_endpoint}")
    print(f"   ✓ Chat Deployment: {azure_settings.azure_openai_chat_deployment}")
    print(f"   ✓ Embedding Deployment: {azure_settings.azure_openai_embedding_deployment}")
    print(f"   ✓ API Version: {azure_settings.azure_openai_api_version}")
    
    print(f"\n📁 Paths:")
    print(f"   • Input Folder: {INPUT_FOLDER}")
    print(f"   • Knowledge Base: {INDEX_PATH}")
    print(f"   • Cache: {CACHE_PATH}")
    print(f"   • Memory: {MEMORY_PATH}")
    
    print(f"\n⚙️  Feature Flags:")
    print(f"   • Agentic Mode: {ENABLE_AGENTIC_MODE}")
    print(f"   • Semantic Cache: {ENABLE_SEMANTIC_CACHE}")
    print(f"   • Query Enhancement: {ENABLE_QUERY_ENHANCEMENT}")
    print(f"   • Reranking: {ENABLE_RERANKING} ({RERANKING_METHOD})")
    print(f"   • Validation: {ENABLE_VALIDATION}")
    print(f"   • Self-Healing: {ENABLE_SELF_HEALING} (max {SELF_HEAL_MAX_RETRIES} retries)")
    
    print(f"\n🔧 Core Settings:")
    print(f"   • Chunk Size: {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
    print(f"   • Top-K: {DEFAULT_TOP_K}")
    print(f"   • LLM Temperature: {LLM_TEMPERATURE}")
    print(f"   • Cache Threshold: {CACHE_SIMILARITY_THRESHOLD}")
    print(f"   • Validation LLM: {VALIDATION_LLM_PROVIDER} ({VALIDATION_LLM_DEPLOYMENT})")
    print(f"   • Self-Heal Threshold: {SELF_HEAL_MIN_QUALITY_SCORE}")
    print("=" * 80)


# ============================================================================
# SECTION 17: Initialization (MUST BE AT END!)
# ============================================================================

# Auto-create directories on import
ensure_directories()

# Define model name using azure_settings (which is now available)
QUERY_ENHANCEMENT_MODEL = azure_settings.azure_openai_chat_deployment  # pra-poc-gpt-4o

# """
# Configuration and Settings for Agentic Medical RAG System

# Design:
# - Pydantic Settings loads ONLY secrets/keys from .env (7 variables)
# - All other settings are Python constants (toggles, paths, configs)
# - Matches legacy config.py pattern while adding type safety
# """

# from pydantic import Field
# from pydantic_settings import BaseSettings
# from pathlib import Path
# from typing import Literal

# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# # ============================================================================
# # SECTION 1: Azure Credentials & Endpoints (from .env)
# # ============================================================================

# class AzureSettings(BaseSettings):
#     """
#     Load Azure credentials from environment variables.
#     These are the ONLY settings that come from .env
#     """
    
#     # Azure Document Intelligence
#     azure_document_intelligence_endpoint: str = Field(
#         ...,
#         alias="AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
#         description="Azure Document Intelligence endpoint URL"
#     )
    
#     azure_document_intelligence_key: str = Field(
#         ...,
#         alias="AZURE_DOCUMENT_INTELLIGENCE_KEY",
#         description="Azure Document Intelligence API key"
#     )
    
#     # Azure OpenAI
#     azure_openai_endpoint: str = Field(
#         ...,
#         alias="AZURE_OPENAI_ENDPOINT",
#         description="Azure OpenAI endpoint URL"
#     )
    
#     azure_openai_key: str = Field(
#         ...,
#         alias="AZURE_OPENAI_KEY",
#         description="Azure OpenAI API key"
#     )
    
#     azure_openai_api_version: str = Field(
#         ...,
#         alias="AZURE_OPENAI_API_VERSION",
#         description="Azure OpenAI API version"
#     )
    
#     azure_openai_embedding_deployment: str = Field(
#         ...,
#         alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
#         description="Azure OpenAI embedding model deployment name"
#     )
    
#     azure_openai_chat_deployment: str = Field(
#         ...,
#         alias="AZURE_OPENAI_CHAT_DEPLOYMENT",
#         description="Azure OpenAI chat model deployment name"
#     )
    
#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"
#         case_sensitive = False

# # Singleton instance
# azure_settings = AzureSettings()

# # ============================================================================
# # SECTION 2: Paths & Directories
# # ============================================================================

# # Base directories
# BASE_DIR = Path(__file__).parent.parent
# DATA_DIR = BASE_DIR / "data"

# # Data subdirectories
# INPUT_FOLDER = DATA_DIR / "input"
# INDEX_PATH = DATA_DIR / "knowledge_base"
# CACHE_PATH = DATA_DIR / "cache"
# MEMORY_PATH = DATA_DIR / "memory"

# # KB specific paths
# FAISS_INDEX_FILE = INDEX_PATH / "faiss_index.bin"
# CHUNKS_FILE = INDEX_PATH / "chunks.json"
# METADATA_FILE = INDEX_PATH / "metadata.json"
# TRACKER_FILE = INDEX_PATH / "pdf_tracker.json"

# # Cache specific paths
# CACHE_INDEX_FILE = CACHE_PATH / "cache_index.bin"
# CACHE_DATA_FILE = CACHE_PATH / "cache_data.json"

# # Memory specific paths
# MEMORY_DB_FILE = MEMORY_PATH / "conversation_memory.db"
# ENTITY_MEMORY_FILE = MEMORY_PATH / "entity_memory.json"

# # ============================================================================
# # SECTION 3: PDF Processing & Chunking
# # ============================================================================

# CHUNK_SIZE = 1024
# CHUNK_OVERLAP = 200
# MIN_CHUNK_SIZE = 50  # Skip chunks smaller than this

# # Azure Document Intelligence settings
# DOC_INTEL_MODEL = "prebuilt-read"  # For medical documents with handwriting
# DOC_INTEL_PAGES = None  # Process all pages
# DOC_INTEL_LOCALE = "en-US"

# # ============================================================================
# # SECTION 4: Embeddings
# # ============================================================================

# EMBEDDING_DIMENSION = 3072  # text-embedding-3-large
# EMBEDDING_BATCH_SIZE = 100  # Batch size for embedding generation
# EMBEDDING_MAX_RETRIES = 3
# EMBEDDING_RETRY_DELAY = 2  # seconds

# # ============================================================================
# # SECTION 5: Knowledge Base Settings
# # ============================================================================

# # Deduplication
# ENABLE_PDF_DEDUPLICATION = True
# ENABLE_CHUNK_DEDUPLICATION = True

# # Autosync
# ENABLE_AUTOSYNC = True
# AUTOSYNC_ON_STARTUP = False  # Set to True for automatic sync on app start

# # ============================================================================
# # SECTION 6: Retrieval Settings
# # ============================================================================

# DEFAULT_TOP_K = 5
# RETRIEVAL_BUFFER = 10  # Retrieve more, then filter/rerank
# MAX_DISTANCE_THRESHOLD = 1.5  # FAISS L2 distance threshold
# MIN_CHUNKS_RETURNED = 3  # Always return at least this many

# # Multi-strategy retrieval
# ENABLE_MULTI_STRATEGY = True
# RECIPROCAL_RANK_K = 60  # RRF constant

# # ============================================================================
# # SECTION 7: Reranking Settings
# # ============================================================================

# ENABLE_RERANKING = True

# # Reranking method: "cross_encoder" (free, local) or "llm" (expensive, cloud)
# RERANKING_METHOD: Literal["cross_encoder", "llm"] = "cross_encoder"

# RERANKING_TOP_K = 5

# # Cross-encoder model (local, free, ~900MB)
# RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# # Batch processing settings
# RERANKING_BATCH_SIZE = 32

# # Score normalization method
# RERANKING_SCORE_NORMALIZATION: Literal["minmax", "sigmoid", "none"] = "minmax"

# # Smart reranking with LLM fallback
# ENABLE_LLM_RERANKING_FALLBACK = True
# RERANKING_FALLBACK_THRESHOLD = 0.5  # If score < this, fallback to LLM

# # LLM reranking settings (if using "llm" method or fallback)
# LLM_RERANKING_TEMPERATURE = 0.1
# LLM_RERANKING_MAX_TOKENS = 200

# # Model caching directory (None = use default)
# CROSS_ENCODER_CACHE_DIR = None

# # ============================================================================
# # SECTION 8: Query Enhancement Settings
# # ============================================================================

# ENABLE_QUERY_ENHANCEMENT = True

# # Query Classification
# ENABLE_QUERY_CLASSIFICATION = True

# # Query Decomposition (for complex multi-part queries)
# ENABLE_QUERY_DECOMPOSITION = True
# MAX_SUBQUERIES = 3  # Maximum sub-queries from decomposition

# # HyDE (Hypothetical Document Embeddings)
# ENABLE_HYDE = True
# HYDE_FOR_SIMPLE_QUERIES = False  # Only use HyDE for complex queries
# HYDE_TEMPERATURE = 0.4  # Higher = more creative hypothetical answers
# HYDE_MAX_TOKENS = 200

# # Query Variations (alternative phrasings)
# ENABLE_QUERY_VARIATIONS = True
# MAX_QUERY_VARIATIONS = 3  # Number of variations to generate
# QUERY_VARIATION_METHOD: Literal["llm", "template"] = "llm"  # "llm" or "template"

# # Query enhancement LLM settings
# QUERY_ENHANCEMENT_TEMPERATURE = 0.3
# QUERY_ENHANCEMENT_MAX_TOKENS = 300

# # ============================================================================
# # SECTION 9: LLM Answer Generation Settings
# # ============================================================================

# # LLM Configuration (PRESERVED from legacy)
# LLM_TEMPERATURE = 0.3
# LLM_MAX_TOKENS = 400
# LLM_MAX_RETRIES = 3
# LLM_RETRY_DELAY = 2

# # ============================================================================
# # SECTION 10: Semantic Cache Settings
# # ============================================================================

# ENABLE_SEMANTIC_CACHE = True
# CACHE_SIMILARITY_THRESHOLD = 0.90  # Must be very similar to return cached answer
# CACHE_TTL_HOURS = 24  # Cache entry expires after 24 hours
# CACHE_MAX_ENTRIES = 1000  # Maximum cache size

# # ============================================================================
# # SECTION 11: Memory Settings
# # ============================================================================

# # Short-term memory (sliding window)
# SHORT_TERM_WINDOW_SIZE = 5  # Last N exchanges

# # Long-term memory
# ENABLE_LONG_TERM_MEMORY = True
# LONG_TERM_SUMMARY_TRIGGER = 10  # Summarize after N exchanges

# # Entity memory
# ENABLE_ENTITY_MEMORY = True

# # Semantic memory (FAISS-based conversation search)
# ENABLE_SEMANTIC_MEMORY = True
# SEMANTIC_MEMORY_TOP_K = 3

# # ============================================================================
# # SECTION 12: Validation & Self-Healing Settings
# # ============================================================================

# ENABLE_VALIDATION = True
# ENABLE_SELF_HEALING = True
# MAX_SELF_HEALING_ITERATIONS = 2

# # Confidence scoring
# CONFIDENCE_THRESHOLD_HIGH = 0.8
# CONFIDENCE_THRESHOLD_MEDIUM = 0.6
# CONFIDENCE_THRESHOLD_LOW = 0.4

# # ============================================================================
# # SECTION 13: Agentic Mode vs Classic Mode
# # ============================================================================

# # Feature flag for agentic system
# ENABLE_AGENTIC_MODE = True  # Set to False to use legacy pipeline

# # CrewAI settings
# CREW_VERBOSE = True
# CREW_MEMORY = True  # CrewAI's built-in memory
# CREW_MAX_RPM = 60  # Rate limit

# # ============================================================================
# # SECTION 14: Logging
# # ============================================================================

# LOG_LEVEL = "INFO"
# LOG_TO_FILE = True
# LOG_FILE_PATH = BASE_DIR / "logs" / "medical_rag.log"
# LOG_ROTATION = "10 MB"
# LOG_RETENTION = "30 days"

# # ============================================================================
# # SECTION 15: Helper Functions
# # ============================================================================

# def ensure_directories():
#     """Create all necessary directories if they don't exist"""
#     directories = [
#         INPUT_FOLDER,
#         INDEX_PATH,
#         CACHE_PATH,
#         MEMORY_PATH,
#         BASE_DIR / "logs"
#     ]
    
#     for directory in directories:
#         directory.mkdir(parents=True, exist_ok=True)


# def get_kb_paths():
#     """Return all KB-related paths as a dict"""
#     return {
#         "index": FAISS_INDEX_FILE,
#         "chunks": CHUNKS_FILE,
#         "metadata": METADATA_FILE,
#         "tracker": TRACKER_FILE
#     }


# def get_cache_paths():
#     """Return all cache-related paths as a dict"""
#     return {
#         "index": CACHE_INDEX_FILE,
#         "data": CACHE_DATA_FILE
#     }


# def get_memory_paths():
#     """Return all memory-related paths as a dict"""
#     return {
#         "db": MEMORY_DB_FILE,
#         "entities": ENTITY_MEMORY_FILE
#     }


# def print_config_summary():
#     """Print configuration summary for debugging"""
#     print("=" * 80)
#     print("AGENTIC MEDICAL RAG - CONFIGURATION SUMMARY")
#     print("=" * 80)
#     print(f"\n🔐 Azure Settings:")
#     print(f"   ✓ Document Intelligence: {azure_settings.azure_document_intelligence_endpoint}")
#     print(f"   ✓ OpenAI Endpoint: {azure_settings.azure_openai_endpoint}")
#     print(f"   ✓ Chat Deployment: {azure_settings.azure_openai_chat_deployment}")
#     print(f"   ✓ Embedding Deployment: {azure_settings.azure_openai_embedding_deployment}")
#     print(f"   ✓ API Version: {azure_settings.azure_openai_api_version}")
    
#     print(f"\n📁 Paths:")
#     print(f"   • Input Folder: {INPUT_FOLDER}")
#     print(f"   • Knowledge Base: {INDEX_PATH}")
#     print(f"   • Cache: {CACHE_PATH}")
#     print(f"   • Memory: {MEMORY_PATH}")
    
#     print(f"\n⚙️  Feature Flags:")
#     print(f"   • Agentic Mode: {ENABLE_AGENTIC_MODE}")
#     print(f"   • Semantic Cache: {ENABLE_SEMANTIC_CACHE}")
#     print(f"   • Query Enhancement: {ENABLE_QUERY_ENHANCEMENT}")
#     print(f"   • Reranking: {ENABLE_RERANKING} ({RERANKING_METHOD})")
#     print(f"   • Validation: {ENABLE_VALIDATION}")
#     print(f"   • Self-Healing: {ENABLE_SELF_HEALING}")
    
#     print(f"\n🔧 Core Settings:")
#     print(f"   • Chunk Size: {CHUNK_SIZE} (overlap: {CHUNK_OVERLAP})")
#     print(f"   • Top-K: {DEFAULT_TOP_K}")
#     print(f"   • LLM Temperature: {LLM_TEMPERATURE}")
#     print(f"   • Cache Threshold: {CACHE_SIMILARITY_THRESHOLD}")
#     print("=" * 80)




# # ============================================================================
# # SECTION 16: validation
# # ============================================================================
# # ============================================================================
# # SECTION 11: LLM-as-Judge Configuration (for ValidationTools)
# # ============================================================================

# # Enable LLM-based validation
# ENABLE_LLM_JUDGE = True

# # LLM provider for validation (azure_openai, gemini, openai)
# VALIDATION_LLM_PROVIDER = os.getenv("VALIDATION_LLM_PROVIDER", "azure_openai")

# # For Azure OpenAI: Use deployment name from env
# VALIDATION_LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")

# # Use existing Azure OpenAI credentials
# VALIDATION_LLM_API_KEY = os.getenv("AZURE_OPENAI_KEY")  # ← Your env var
# VALIDATION_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# VALIDATION_LLM_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# # Fallback to rule-based if LLM fails
# VALIDATION_FALLBACK_TO_RULES = True

# # Validation thresholds
# MIN_ANSWER_LENGTH = 50
# MAX_ANSWER_LENGTH = 2000
# MIN_CITATIONS = 1
# MIN_QUALITY_SCORE = 0.6

# # PII Detection (DISABLED - returns unfiltered answers)
# ENABLE_PII_DETECTION = False

# # Strict mode
# VALIDATION_STRICT_MODE = False




# # ============================================================================
# # SECTION 17: Initialization & Model Names (MUST BE AT END!)
# # ============================================================================

# # Auto-create directories on import
# ensure_directories()

# # Define model name using azure_settings (which is now available)
# QUERY_ENHANCEMENT_MODEL = azure_settings.azure_openai_chat_deployment  # pra-poc-gpt-4o