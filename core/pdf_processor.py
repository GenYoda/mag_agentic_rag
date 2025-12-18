"""
PDF Text Extraction using Azure Document Intelligence

Wraps Azure Document Intelligence API for medical document processing.

Supports:
- Handwritten text (critical for medical forms)
- Complex layouts
- Page-by-page extraction
- Metadata preservation
- Automatic retry on network failures

Uses prebuilt-read model optimized for medical documents.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

from utils.azure_clients import get_document_intelligence_client
from core.deduplication import calculate_chunk_signature
from utils.file_utils import calculate_file_hash, get_file_info
from config.settings import (
    DOC_INTEL_MODEL,
    DOC_INTEL_PAGES,
    DOC_INTEL_LOCALE,
    MIN_CHUNK_SIZE
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
RETRY_BACKOFF = 2  # exponential backoff multiplier


# ============================================================================
# SECTION 1: PDF Text Extraction with Retry Logic
# ============================================================================

def extract_text_from_pdf(
    pdf_path: Path,
    model: str = DOC_INTEL_MODEL,
    pages: Optional[str] = DOC_INTEL_PAGES,
    locale: str = DOC_INTEL_LOCALE,
    max_retries: int = MAX_RETRIES,
    retry_delay: int = RETRY_DELAY
) -> Dict[str, Any]:
    """
    Extract text from PDF using Azure Document Intelligence with automatic retry.
    
    Features:
    - Automatic retry on network/DNS failures
    - Exponential backoff
    - Detailed error logging
    
    Args:
        pdf_path: Path to PDF file
        model: Document Intelligence model to use
        pages: Pages to extract (None = all)
        locale: Document locale
        max_retries: Maximum retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 5)
        
    Returns:
        dict: Extraction result with success/error status
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        return {
            'success': False,
            'error': f'File not found: {pdf_path}'
        }
    
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.info(f"🔄 Retry attempt {attempt}/{max_retries} for {pdf_path.name}")
            
            # Call internal extraction function
            result = _extract_text_internal(
                pdf_path=pdf_path,
                model=model,
                pages=pages,
                locale=locale
            )
            
            # Success!
            if result['success']:
                if attempt > 1:
                    logger.info(f"✅ Extraction succeeded on attempt {attempt}")
                return result
            else:
                # Extraction failed but not due to network - don't retry
                return result
                
        except (ServiceRequestError, ConnectionError, TimeoutError, OSError) as e:
            # Network/DNS errors - these are retryable
            error_str = str(e)
            last_error = error_str
            
            # Identify error type
            if "Failed to resolve" in error_str or "getaddrinfo failed" in error_str:
                error_type = "🌐 DNS resolution failure"
            elif "Connection" in error_str:
                error_type = "🔌 Connection error"
            elif "Timeout" in error_str:
                error_type = "⏱️ Timeout"
            else:
                error_type = "🔗 Network error"
            
            logger.warning(
                f"{error_type} on attempt {attempt}/{max_retries} for {pdf_path.name}: {error_str}"
            )
            
            if attempt < max_retries:
                # Calculate exponential backoff delay
                delay = retry_delay * (RETRY_BACKOFF ** (attempt - 1))
                logger.info(f"⏳ Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                logger.error(
                    f"❌ All {max_retries} attempts failed for {pdf_path.name}"
                )
                return {
                    'success': False,
                    'error': f'Extraction failed after {max_retries} attempts: {last_error}',
                    'attempts': max_retries
                }
        
        except HttpResponseError as e:
            # Azure API errors (quota, auth, etc.) - don't retry
            logger.error(f"❌ Azure API error (non-retryable): {e}")
            return {
                'success': False,
                'error': f'Azure API error: {str(e)}'
            }
        
        except Exception as e:
            # Unknown errors - don't retry
            logger.error(f"❌ Unexpected error: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    # Should not reach here, but safety fallback
    return {
        'success': False,
        'error': f'Extraction failed after {max_retries} attempts: {last_error}',
        'attempts': max_retries
    }


def _extract_text_internal(
    pdf_path: Path,
    model: str,
    pages: Optional[str],
    locale: str
) -> Dict[str, Any]:
    """
    Internal extraction function (called by retry wrapper).
    
    This contains the actual Azure Document Intelligence API call.
    Separated to keep retry logic clean.
    """
    start_time = time.time()
    
    try:
        client = get_document_intelligence_client()
        
        # Read PDF content as bytes (NEW SDK requires this)
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        
        # Call API with body parameter (NEW SDK signature)
        poller = client.begin_analyze_document(
            model_id=model,
            body=pdf_content,
            content_type="application/pdf",
            pages=pages,
            locale=locale
        )
        
        result = poller.result()
        
        # Extract text page by page
        page_texts = {}
        full_text_parts = []
        
        if result.pages:
            for page in result.pages:
                page_num = page.page_number
                page_text = []
                
                if page.lines:
                    for line in page.lines:
                        page_text.append(line.content)
                
                page_content = "\n".join(page_text)
                page_texts[page_num] = page_content
                full_text_parts.append(page_content)
        
        full_text = "\n\n".join(full_text_parts)
        
        # Calculate hashes and metadata
        file_hash = calculate_file_hash(pdf_path, algorithm="md5")
        content_signature = calculate_chunk_signature(full_text)
        file_info = get_file_info(pdf_path)
        extraction_time = time.time() - start_time
        
        return {
            'success': True,
            'full_text': full_text,
            'page_texts': page_texts,
            'total_pages': len(page_texts),
            'file_hash': file_hash,
            'content_signature': content_signature,
            'metadata': {
                'filename': pdf_path.name,
                'size_mb': file_info['size_mb'],
                'extraction_time_seconds': round(extraction_time, 2),
                'model_used': model
            }
        }
        
    except Exception as e:
        # Re-raise to let retry wrapper handle it
        raise


# ============================================================================
# SECTION 2: Batch Extraction (with retry support)
# ============================================================================

def extract_batch(
    pdf_paths: List[Path],
    show_progress: bool = True,
    max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    Extract text from multiple PDFs with progress tracking and retry.
    
    Args:
        pdf_paths: List of PDF file paths
        show_progress: Print progress messages (default: True)
        max_retries: Maximum retry attempts per PDF
        
    Returns:
        dict: Batch extraction results
        {
            'results': {filename: extraction_result},
            'summary': {
                'total': int,
                'successful': int,
                'failed': int,
                'retried': int,
                'total_time_seconds': float
            }
        }
    """
    start_time = time.time()
    results = {}
    successful = 0
    failed = 0
    retried = 0
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        if show_progress:
            print(f"Processing {i}/{len(pdf_paths)}: {pdf_path.name}...", end=" ")
        
        result = extract_text_from_pdf(pdf_path, max_retries=max_retries)
        results[pdf_path.name] = result
        
        if result['success']:
            successful += 1
            if show_progress:
                print(f"✅ ({result['total_pages']} pages)")
            
            # Track if it was retried
            if result.get('attempts', 1) > 1:
                retried += 1
        else:
            failed += 1
            if show_progress:
                error_msg = result.get('error', 'Unknown error')
                # Truncate long error messages
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                print(f"❌ {error_msg}")
    
    total_time = time.time() - start_time
    
    return {
        'results': results,
        'summary': {
            'total': len(pdf_paths),
            'successful': successful,
            'failed': failed,
            'retried': retried,
            'total_time_seconds': round(total_time, 2)
        }
    }


# ============================================================================
# SECTION 3: Text Chunking
# ============================================================================

def chunk_text(
    text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for indexing.
    
    Uses simple character-based chunking with overlap to preserve context.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size (smaller chunks are skipped)
        metadata: Optional metadata to attach to each chunk
        
    Returns:
        List[dict]: List of chunk dictionaries
        {
            'text': str,
            'chunk_index': int,
            'start_char': int,
            'end_char': int,
            'metadata': dict
        }
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    text_length = len(text)
    start = 0
    chunk_index = 0
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        # Extract chunk
        chunk_text = text[start:end]
        
        # Skip if chunk is too small
        if len(chunk_text.strip()) < min_chunk_size:
            break
        
        # Create chunk dict
        chunk_dict = {
            'text': chunk_text,
            'chunk_index': chunk_index,
            'start_char': start,
            'end_char': min(end, text_length),
            'metadata': metadata.copy() if metadata else {}
        }
        
        chunks.append(chunk_dict)
        
        # Move start position (with overlap)
        start = end - chunk_overlap
        chunk_index += 1
        
        # Prevent infinite loop if overlap >= chunk_size
        if chunk_overlap >= chunk_size:
            break
    
    return chunks


def chunk_pdf_by_pages(
    page_texts: Dict[int, str],
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    base_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Chunk PDF text while preserving page information.
    
    Each chunk's metadata includes the page number(s) it spans.
    
    Args:
        page_texts: Dict of {page_num: text}
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size
        base_metadata: Base metadata to include in all chunks
        
    Returns:
        List[dict]: List of chunks with page metadata
    """
    all_chunks = []
    
    for page_num, page_text in sorted(page_texts.items()):
        # Create metadata for this page
        page_metadata = base_metadata.copy() if base_metadata else {}
        page_metadata['page_numbers'] = [page_num]
        page_metadata['page'] = page_num
        
        # Chunk the page text
        page_chunks = chunk_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            metadata=page_metadata
        )
        
        all_chunks.extend(page_chunks)
    
    return all_chunks


# ============================================================================
# SECTION 4: Complete PDF Processing Pipeline
# ============================================================================

def process_pdf(
    pdf_path: Path,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    extract_chunks: bool = True,
    max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    Complete pipeline: Extract text + chunk + calculate signatures.
    
    This is the main function to use for processing PDFs.
    Now includes automatic retry on network failures.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Chunk size for text splitting
        chunk_overlap: Overlap between chunks
        extract_chunks: Whether to chunk text (default: True)
        max_retries: Maximum retry attempts (default: 3)
        
    Returns:
        dict: Complete processing results
        {
            'success': bool,
            'extraction': {extraction_result},
            'chunks': [list of chunks] (if extract_chunks=True),
            'total_chunks': int,
            'error': str (if failed)
        }
    """
    # Step 1: Extract text (with retry)
    extraction_result = extract_text_from_pdf(pdf_path, max_retries=max_retries)
    
    if not extraction_result['success']:
        return {
            'success': False,
            'extraction': extraction_result,
            'error': extraction_result.get('error')
        }
    
    # Step 2: Chunk text (if requested)
    chunks = []
    if extract_chunks:
        base_metadata = {
            'source': pdf_path.name,
            'file_hash': extraction_result['file_hash'],
            'content_signature': extraction_result['content_signature']
        }
        
        chunks = chunk_pdf_by_pages(
            page_texts=extraction_result['page_texts'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_metadata=base_metadata
        )
    
    return {
        'success': True,
        'extraction': extraction_result,
        'chunks': chunks,
        'total_chunks': len(chunks)
    }


# ============================================================================
# SECTION 5: Validation and Stats
# ============================================================================

def validate_extraction(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extraction result quality.
    
    Checks:
    - Text extracted
    - Reasonable length
    - Pages processed
    
    Args:
        extraction_result: Result from extract_text_from_pdf()
        
    Returns:
        dict: Validation results
        {
            'valid': bool,
            'warnings': [list of warnings],
            'stats': {statistics}
        }
    """
    warnings = []
    
    if not extraction_result.get('success'):
        return {
            'valid': False,
            'warnings': ['Extraction failed'],
            'stats': {}
        }
    
    full_text = extraction_result.get('full_text', '')
    total_pages = extraction_result.get('total_pages', 0)
    
    # Check if text was extracted
    if not full_text or len(full_text.strip()) < 10:
        warnings.append('Very little text extracted (possible scan/image PDF)')
    
    # Check pages
    if total_pages == 0:
        warnings.append('No pages processed')
    
    # Check average text per page
    if total_pages > 0:
        avg_chars_per_page = len(full_text) / total_pages
        if avg_chars_per_page < 100:
            warnings.append(f'Low text density ({avg_chars_per_page:.0f} chars/page)')
    
    stats = {
        'total_characters': len(full_text),
        'total_words': len(full_text.split()),
        'total_pages': total_pages,
        'avg_chars_per_page': len(full_text) / total_pages if total_pages > 0 else 0
    }
    
    return {
        'valid': len(warnings) == 0,
        'warnings': warnings,
        'stats': stats
    }


def print_extraction_summary(result: Dict[str, Any]):
    """
    Print formatted extraction summary.
    
    Args:
        result: Result from process_pdf()
    """
    print("\n" + "=" * 80)
    print("PDF EXTRACTION SUMMARY")
    print("=" * 80)
    
    if not result['success']:
        print(f"❌ Extraction failed: {result.get('error')}")
        return
    
    extraction = result['extraction']
    metadata = extraction['metadata']
    
    print(f"\n📄 File: {metadata['filename']}")
    print(f"📊 Size: {metadata['size_mb']} MB")
    print(f"📖 Pages: {extraction['total_pages']}")
    print(f"⏱️ Extraction Time: {metadata['extraction_time_seconds']}s")
    print(f"🔧 Model: {metadata['model_used']}")
    
    if 'chunks' in result:
        print(f"\n📝 Chunks Generated: {result['total_chunks']}")
    
    print(f"\n🔑 File Hash: {extraction['file_hash'][:16]}...")
    print(f"🔑 Content Signature: {extraction['content_signature'][:16]}...")
    
    # Validation
    validation = validate_extraction(extraction)
    if validation['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    print(f"\n📈 Statistics:")
    for key, value in validation['stats'].items():
        print(f"   • {key}: {value}")
    
    print("=" * 80 + "\n")
