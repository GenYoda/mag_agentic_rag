"""
Extraction Tools - Document Intelligence Specialist

Wraps core/pdf_processor.py for PDF text extraction using Azure Document Intelligence.
Plain Python class (no CrewAI decorators yet - Phase 4).

Provides:
- Single PDF extraction
- Batch extraction
- Complete processing pipeline (extract + chunk)
- Content signature calculation
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Import from Phase 2 core modules
from core.pdf_processor import (
    extract_text_from_pdf,
    chunk_text,
    process_pdf
)
from core.deduplication import calculate_chunk_signature
from utils.file_utils import calculate_file_hash, calculate_content_hash

logger = logging.getLogger(__name__)


class ExtractionTools:
    """
    Wraps Azure Document Intelligence extraction with deduplication support.
    
    All methods return dicts for easy JSON serialization and tool compatibility.
    """
    
    def __init__(self):
        """Initialize extraction tools."""
        logger.info("ExtractionTools initialized")
    
    def extract_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from single PDF using Azure Document Intelligence.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            {
                'success': bool,
                'pdf_path': str,
                'full_text': str,
                'page_texts': dict,  # {page_num: text}
                'total_pages': int,
                'word_count': int,
                'file_hash': str,
                'content_signature': str,
                'metadata': dict,
                'error': str | None
            }
        """
        try:
            pdf_path_obj = Path(pdf_path)
            
            if not pdf_path_obj.exists():
                return {
                    'success': False,
                    'pdf_path': str(pdf_path),
                    'error': f'PDF file not found: {pdf_path}'
                }
            
            logger.info(f"Extracting text from: {pdf_path_obj.name}")
            
            # Call core PDF processor (uses Azure Document Intelligence)
            extraction_result = extract_text_from_pdf(pdf_path_obj)
            
            if not extraction_result.get('success'):
                return {
                    'success': False,
                    'pdf_path': str(pdf_path),
                    'error': extraction_result.get('error', 'Unknown extraction error')
                }
            
            # Calculate hashes for deduplication
            file_hash = calculate_file_hash(pdf_path_obj, algorithm='md5')
            full_text = extraction_result['full_text']
            content_signature = calculate_content_hash(full_text, algorithm='sha256')
            
            # Calculate word count
            word_count = len(full_text.split())
            
            result = {
                'success': True,
                'pdf_path': str(pdf_path_obj),
                'full_text': full_text,
                'page_texts': extraction_result.get('page_texts', {}),
                'total_pages': extraction_result.get('total_pages', 0),
                'word_count': word_count,
                'file_hash': file_hash,
                'content_signature': content_signature,
                'metadata': extraction_result.get('metadata', {}),
                'error': None
            }
            
            logger.info(f"✅ Extracted {word_count} words from {pdf_path_obj.name} "
                       f"({result['total_pages']} pages)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'pdf_path': str(pdf_path),
                'error': f'Extraction failed: {str(e)}'
            }
    
    def batch_extract_pdfs(
        self, 
        pdf_paths: List[str],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            show_progress: Show progress messages (default: True)
            
        Returns:
            {
                'success': bool,
                'total_pdfs': int,
                'successful': int,
                'failed': int,
                'results': list,  # List of extraction results
                'errors': dict    # {pdf_path: error_msg}
            }
        """
        logger.info(f"Batch extracting {len(pdf_paths)} PDFs")
        
        results = []
        errors = {}
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            if show_progress:
                print(f"Processing {i}/{len(pdf_paths)}: {Path(pdf_path).name}...", end=" ")
            
            result = self.extract_pdf(pdf_path)
            results.append(result)
            
            if result['success']:
                successful += 1
                if show_progress:
                    print(f"✅ ({result['total_pages']} pages)")
            else:
                failed += 1
                errors[pdf_path] = result.get('error', 'Unknown error')
                if show_progress:
                    print(f"❌ {result.get('error', 'Unknown error')}")
        
        summary = {
            'success': failed == 0,
            'total_pdfs': len(pdf_paths),
            'successful': successful,
            'failed': failed,
            'results': results,
            'errors': errors
        }
        
        logger.info(f"Batch extraction complete: {successful}/{len(pdf_paths)} successful")
        
        return summary
    
    def process_pdf_with_chunking(
        self,
        pdf_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        extract_chunks: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: extract + chunk + calculate signatures.
        
        Uses core/pdf_processor.process_pdf() which handles everything.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Characters per chunk (uses config default if None)
            chunk_overlap: Overlap between chunks (uses config default if None)
            extract_chunks: If False, only extract text without chunking
            
        Returns:
            {
                'success': bool,
                'pdf_path': str,
                'full_text': str,
                'file_hash': str,
                'content_signature': str,
                'chunks': list | None,  # Only if extract_chunks=True
                'chunk_count': int,
                'metadata': dict,
                'error': str | None
            }
        """
        try:
            from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
            
            pdf_path_obj = Path(pdf_path)
            
            if not pdf_path_obj.exists():
                return {
                    'success': False,
                    'pdf_path': str(pdf_path),
                    'error': f'PDF file not found: {pdf_path}'
                }
            
            # Use settings defaults if not specified
            chunk_size = chunk_size or CHUNK_SIZE
            chunk_overlap = chunk_overlap or CHUNK_OVERLAP
            
            logger.info(f"Processing {pdf_path_obj.name} with chunking "
                       f"(size={chunk_size}, overlap={chunk_overlap})")
            
            # Use core process_pdf which handles extraction + chunking
            result = process_pdf(
                pdf_path=pdf_path_obj,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                extract_chunks=extract_chunks
            )
            
            if result['success']:
                chunk_count = result.get('total_chunks', 0)
                logger.info(f"✅ Processed {pdf_path_obj.name}: {chunk_count} chunks created")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'pdf_path': str(pdf_path),
                'error': f'Processing failed: {str(e)}'
            }
    
    def calculate_signature(self, text: str) -> str:
        """
        Calculate SHA256 content signature for deduplication.
        
        Args:
            text: Text content to hash
            
        Returns:
            SHA256 hash string
        """
        return calculate_content_hash(text, algorithm='sha256')
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get extraction statistics (placeholder for future enhancement).
        
        Returns:
            Statistics dict
        """
        return {
            'tool_name': 'ExtractionTools',
            'status': 'operational',
            'capabilities': [
                'single_pdf_extraction',
                'batch_extraction',
                'chunking',
                'signature_calculation'
            ]
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_single_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Convenience function for single PDF extraction.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extraction result dict
    """
    tools = ExtractionTools()
    return tools.extract_pdf(pdf_path)


def extract_multiple_pdfs(pdf_paths: List[str]) -> Dict[str, Any]:
    """
    Convenience function for batch PDF extraction.
    
    Args:
        pdf_paths: List of PDF file paths
        
    Returns:
        Batch extraction result dict
    """
    tools = ExtractionTools()
    return tools.batch_extract_pdfs(pdf_paths)
