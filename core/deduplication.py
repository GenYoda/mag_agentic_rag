"""
Deduplication System for Agentic Medical RAG

Handles two-level deduplication:
1. PDF-level: Detect duplicate/modified PDFs using file_hash + content_signature
2. Chunk-level: Deduplicate chunks within and across documents using chunk_signature

PDF Tracker Structure:
{
    "filename.pdf": {
        "file_hash": "md5_hash_of_bytes",
        "content_signature": "sha256_hash_of_normalized_text",
        "last_indexed": "ISO_timestamp",
        "path": "full_path_to_file",
        "total_pages": 10,
        "total_chunks": 45,
        "is_canonical": true,  # First occurrence of this content
        "duplicate_of": null   # If duplicate, points to canonical filename
    }
}
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
import json

from utils.file_utils import (
    calculate_file_hash,
    calculate_content_hash,
    normalize_text_for_hashing,
    read_json,
    write_json
)
from config.settings import TRACKER_FILE, ENABLE_PDF_DEDUPLICATION, ENABLE_CHUNK_DEDUPLICATION


# ============================================================================
# SECTION 1: PDF Tracker Management
# ============================================================================

class PDFTracker:
    """
    Manages PDF tracking for deduplication.
    
    Tracks:
    - File hashes (byte-level changes)
    - Content signatures (text-level deduplication)
    - Indexing status and metadata
    """
    
    def __init__(self, tracker_file: Path = TRACKER_FILE):
        """
        Initialize PDF tracker.
        
        Args:
            tracker_file: Path to tracker JSON file
        """
        self.tracker_file = Path(tracker_file)
        self.tracker_data: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self) -> bool:
        """
        Load tracker from file.
        
        Returns:
            bool: True if loaded successfully
        """
        data = read_json(self.tracker_file, default={})
        if data:
            self.tracker_data = data
            return True
        return False
    
    def save(self) -> bool:
        """
        Save tracker to file.
        
        Returns:
            bool: True if saved successfully
        """
        return write_json(self.tracker_data, self.tracker_file)
    
    def add_or_update_pdf(
        self,
        filename: str,
        file_hash: str,
        content_signature: str,
        path: str,
        total_pages: int = 0,
        total_chunks: int = 0,
        is_canonical: bool = True,
        duplicate_of: Optional[str] = None
    ) -> bool:
        """
        Add or update PDF entry in tracker.
        
        Args:
            filename: PDF filename
            file_hash: MD5 hash of file bytes
            content_signature: SHA256 hash of normalized text content
            path: Full path to file
            total_pages: Number of pages
            total_chunks: Number of chunks generated
            is_canonical: Whether this is the first occurrence of this content
            duplicate_of: If duplicate, the canonical filename
            
        Returns:
            bool: True if successful
        """
        self.tracker_data[filename] = {
            "file_hash": file_hash,
            "content_signature": content_signature,
            "last_indexed": datetime.now().isoformat(),
            "path": path,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "is_canonical": is_canonical,
            "duplicate_of": duplicate_of
        }
        return self.save()
    
    def get_pdf_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get PDF info from tracker.
        
        Args:
            filename: PDF filename
            
        Returns:
            dict: PDF info or None if not found
        """
        return self.tracker_data.get(filename)
    
    def find_by_content_signature(self, content_signature: str) -> Optional[str]:
        """
        Find filename with matching content signature.
        
        Args:
            content_signature: Content signature to search for
            
        Returns:
            str: Canonical filename with this signature, or None if not found
        """
        for filename, info in self.tracker_data.items():
            if info.get("content_signature") == content_signature and info.get("is_canonical"):
                return filename
        return None
    
    def get_all_content_signatures(self) -> Set[str]:
        """
        Get set of all content signatures in tracker.
        
        Returns:
            Set[str]: All content signatures
        """
        return {info["content_signature"] for info in self.tracker_data.values() if "content_signature" in info}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.
        
        Returns:
            dict: Statistics about tracked PDFs
        """
        total = len(self.tracker_data)
        canonical = sum(1 for info in self.tracker_data.values() if info.get("is_canonical", False))
        duplicates = total - canonical
        
        return {
            "total_pdfs_tracked": total,
            "canonical_pdfs": canonical,
            "duplicate_pdfs": duplicates,
            "total_chunks_indexed": sum(info.get("total_chunks", 0) for info in self.tracker_data.values() if info.get("is_canonical", False))
        }
    
    def remove_pdf(self, filename: str) -> bool:
        """
        Remove PDF from tracker.
        
        Args:
            filename: PDF filename
            
        Returns:
            bool: True if removed
        """
        if filename in self.tracker_data:
            del self.tracker_data[filename]
            return self.save()
        return False
    
    def clear(self) -> bool:
        """
        Clear all tracker data.
        
        Returns:
            bool: True if successful
        """
        self.tracker_data = {}
        return self.save()


# ============================================================================
# SECTION 2: PDF Status Checking
# ============================================================================

def check_pdf_status(
    pdf_path: Path,
    tracker: PDFTracker
) -> Dict[str, Any]:
    """
    Check PDF status (new, modified, duplicate, or unchanged).
    
    Args:
        pdf_path: Path to PDF file
        tracker: PDFTracker instance
        
    Returns:
        dict: Status information
            {
                'status': 'new' | 'modified' | 'duplicate_content' | 'unchanged',
                'filename': str,
                'file_hash': str,
                'existing_info': dict (if exists),
                'recommendation': str
            }
    """
    filename = pdf_path.name
    file_hash = calculate_file_hash(pdf_path, algorithm="md5")
    
    existing_info = tracker.get_pdf_info(filename)
    
    # Case 1: New file (not in tracker)
    if existing_info is None:
        return {
            'status': 'new',
            'filename': filename,
            'file_hash': file_hash,
            'existing_info': None,
            'recommendation': 'Process and index this PDF'
        }
    
    # Case 2: Unchanged (same file hash)
    if existing_info['file_hash'] == file_hash:
        return {
            'status': 'unchanged',
            'filename': filename,
            'file_hash': file_hash,
            'existing_info': existing_info,
            'recommendation': 'Skip - already indexed'
        }
    
    # Case 3: Modified (different file hash)
    return {
        'status': 'modified',
        'filename': filename,
        'file_hash': file_hash,
        'existing_info': existing_info,
        'recommendation': 'Re-extract and re-index'
    }


def check_content_duplication(
    content_signature: str,
    filename: str,
    tracker: PDFTracker
) -> Dict[str, Any]:
    """
    Check if content signature already exists (content-level duplicate).
    
    This is checked AFTER PDF extraction, when we have the text content.
    
    Args:
        content_signature: SHA256 hash of normalized text
        filename: Current PDF filename
        tracker: PDFTracker instance
        
    Returns:
        dict: Duplication check result
            {
                'is_duplicate': bool,
                'canonical_filename': str (if duplicate),
                'recommendation': str
            }
    """
    if not ENABLE_PDF_DEDUPLICATION:
        return {
            'is_duplicate': False,
            'canonical_filename': None,
            'recommendation': 'PDF deduplication disabled - proceed with indexing'
        }
    
    canonical = tracker.find_by_content_signature(content_signature)
    
    if canonical and canonical != filename:
        return {
            'is_duplicate': True,
            'canonical_filename': canonical,
            'recommendation': f'Skip indexing - duplicate of {canonical}'
        }
    
    return {
        'is_duplicate': False,
        'canonical_filename': None,
        'recommendation': 'Unique content - proceed with indexing'
    }


# ============================================================================
# SECTION 3: Chunk-Level Deduplication
# ============================================================================

def calculate_chunk_signature(chunk_text: str) -> str:
    """
    Calculate signature for a text chunk.
    
    Uses same normalization as content signature for consistency.
    
    Args:
        chunk_text: Raw chunk text
        
    Returns:
        str: SHA256 hash of normalized chunk
    """
    return calculate_content_hash(chunk_text, algorithm="sha256")


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    existing_signatures: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], Set[str], int]:
    """
    Remove duplicate chunks based on content signature.
    
    Handles two scenarios:
    1. Within-document deduplication (e.g., repeated headers/footers)
    2. Cross-document deduplication (comparing against existing KB chunks)
    
    Args:
        chunks: List of chunk dicts with 'text' or 'chunk' key
        existing_signatures: Set of chunk signatures already in KB (optional)
        
    Returns:
        tuple: (deduplicated_chunks, new_signatures, duplicates_removed)
    """
    if not ENABLE_CHUNK_DEDUPLICATION:
        # Return all chunks unchanged
        return chunks, set(), 0
    
    seen_signatures = existing_signatures.copy() if existing_signatures else set()
    unique_chunks = []
    duplicates_removed = 0
    
    for chunk in chunks:
        # Get chunk text
        chunk_text = chunk.get('text', chunk.get('chunk', ''))
        
        if not chunk_text.strip():
            continue  # Skip empty chunks
        
        # Calculate signature
        signature = calculate_chunk_signature(chunk_text)
        
        # Check if already seen
        if signature in seen_signatures:
            duplicates_removed += 1
            continue
        
        # Add signature and keep chunk
        seen_signatures.add(signature)
        
        # Add signature to chunk metadata
        if 'metadata' not in chunk:
            chunk['metadata'] = {}
        chunk['metadata']['chunk_signature'] = signature
        
        unique_chunks.append(chunk)
    
    new_signatures = seen_signatures - (existing_signatures or set())
    
    return unique_chunks, new_signatures, duplicates_removed


def get_existing_chunk_signatures(chunks_data: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract chunk signatures from existing chunks data.
    
    Args:
        chunks_data: List of existing chunk dicts from KB
        
    Returns:
        Set[str]: Set of chunk signatures
    """
    signatures = set()
    
    for chunk in chunks_data:
        metadata = chunk.get('metadata', {})
        signature = metadata.get('chunk_signature')
        
        if signature:
            signatures.add(signature)
        else:
            # If no signature stored, calculate it
            chunk_text = chunk.get('text', chunk.get('chunk', ''))
            if chunk_text.strip():
                signature = calculate_chunk_signature(chunk_text)
                signatures.add(signature)
    
    return signatures


# ============================================================================
# SECTION 4: Batch Operations
# ============================================================================

def scan_directory_for_changes(
    input_dir: Path,
    tracker: PDFTracker,
    pattern: str = "*.pdf"
) -> Dict[str, List[Path]]:
    """
    Scan directory and categorize PDFs by status.
    
    Args:
        input_dir: Directory containing PDFs
        tracker: PDFTracker instance
        pattern: File pattern (default: "*.pdf")
        
    Returns:
        dict: Categorized PDF paths
            {
                'new': [list of new PDFs],
                'modified': [list of modified PDFs],
                'unchanged': [list of unchanged PDFs]
            }
    """
    from utils.file_utils import list_files
    
    categorized = {
        'new': [],
        'modified': [],
        'unchanged': []
    }
    
    pdf_files = list_files(input_dir, pattern=pattern)
    
    for pdf_path in pdf_files:
        status_info = check_pdf_status(pdf_path, tracker)
        status = status_info['status']
        
        if status in categorized:
            categorized[status].append(pdf_path)
    
    return categorized


def generate_deduplication_report(
    tracker: PDFTracker,
    chunks_deduplicated: int = 0,
    pdfs_processed: int = 0,
    pdfs_skipped_duplicate: int = 0
) -> Dict[str, Any]:
    """
    Generate deduplication report.
    
    Args:
        tracker: PDFTracker instance
        chunks_deduplicated: Number of duplicate chunks removed
        pdfs_processed: Number of PDFs processed
        pdfs_skipped_duplicate: Number of PDFs skipped as duplicates
        
    Returns:
        dict: Deduplication statistics
    """
    tracker_stats = tracker.get_stats()
    
    return {
        'pdf_deduplication': {
            'enabled': ENABLE_PDF_DEDUPLICATION,
            'total_pdfs_tracked': tracker_stats['total_pdfs_tracked'],
            'canonical_pdfs': tracker_stats['canonical_pdfs'],
            'duplicate_pdfs': tracker_stats['duplicate_pdfs'],
            'pdfs_skipped_this_run': pdfs_skipped_duplicate
        },
        'chunk_deduplication': {
            'enabled': ENABLE_CHUNK_DEDUPLICATION,
            'chunks_deduplicated': chunks_deduplicated,
            'total_chunks_indexed': tracker_stats['total_chunks_indexed']
        },
        'processing_summary': {
            'pdfs_processed': pdfs_processed,
            'pdfs_skipped': pdfs_skipped_duplicate
        }
    }


def print_deduplication_report(report: Dict[str, Any]):
    """
    Print formatted deduplication report.
    
    Args:
        report: Report dict from generate_deduplication_report()
    """
    print("\n" + "=" * 80)
    print("DEDUPLICATION REPORT")
    print("=" * 80)
    
    pdf_dedup = report['pdf_deduplication']
    print(f"\n📄 PDF-Level Deduplication: {'ENABLED' if pdf_dedup['enabled'] else 'DISABLED'}")
    print(f"  • Total PDFs Tracked: {pdf_dedup['total_pdfs_tracked']}")
    print(f"  • Canonical PDFs: {pdf_dedup['canonical_pdfs']}")
    print(f"  • Duplicate PDFs: {pdf_dedup['duplicate_pdfs']}")
    print(f"  • PDFs Skipped (This Run): {pdf_dedup['pdfs_skipped_this_run']}")
    
    chunk_dedup = report['chunk_deduplication']
    print(f"\n📝 Chunk-Level Deduplication: {'ENABLED' if chunk_dedup['enabled'] else 'DISABLED'}")
    print(f"  • Chunks Deduplicated: {chunk_dedup['chunks_deduplicated']}")
    print(f"  • Total Chunks Indexed: {chunk_dedup['total_chunks_indexed']}")
    
    summary = report['processing_summary']
    print(f"\n📊 Processing Summary:")
    print(f"  • PDFs Processed: {summary['pdfs_processed']}")
    print(f"  • PDFs Skipped: {summary['pdfs_skipped']}")
    
    print("=" * 80 + "\n")


# ============================================================================
# SECTION 5: Helper Functions
# ============================================================================

def create_pdf_entry(
    pdf_path: Path,
    full_text: str,
    total_pages: int,
    total_chunks: int
) -> Dict[str, Any]:
    """
    Create a complete PDF tracker entry.
    
    Args:
        pdf_path: Path to PDF
        full_text: Extracted full text
        total_pages: Number of pages
        total_chunks: Number of chunks created
        
    Returns:
        dict: Complete entry data for tracker
    """
    file_hash = calculate_file_hash(pdf_path, algorithm="md5")
    content_signature = calculate_content_hash(full_text, algorithm="sha256")
    
    return {
        "filename": pdf_path.name,
        "file_hash": file_hash,
        "content_signature": content_signature,
        "path": str(pdf_path.absolute()),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "is_canonical": True,
        "duplicate_of": None
    }


def mark_as_duplicate(
    tracker: PDFTracker,
    duplicate_filename: str,
    canonical_filename: str,
    pdf_path: Path
) -> bool:
    """
    Mark a PDF as duplicate of another.
    
    Args:
        tracker: PDFTracker instance
        duplicate_filename: Filename to mark as duplicate
        canonical_filename: Canonical (original) filename
        pdf_path: Path to duplicate file
        
    Returns:
        bool: True if successful
    """
    canonical_info = tracker.get_pdf_info(canonical_filename)
    
    if not canonical_info:
        return False
    
    file_hash = calculate_file_hash(pdf_path, algorithm="md5")
    
    return tracker.add_or_update_pdf(
        filename=duplicate_filename,
        file_hash=file_hash,
        content_signature=canonical_info['content_signature'],
        path=str(pdf_path.absolute()),
        total_pages=canonical_info.get('total_pages', 0),
        total_chunks=0,  # No chunks indexed for duplicate
        is_canonical=False,
        duplicate_of=canonical_filename
    )
