"""
KB Tools - Knowledge Base Manager

Builds and maintains FAISS knowledge base with:
- PDF-level deduplication (file hash + content signature)
- Chunk-level deduplication (chunk signatures)
- Incremental updates (add new PDFs without rebuilding)
- Auto-sync (detect changes and update)
- PDF tracking with metadata

Integrates:
- ExtractionTools (PDF extraction)
- core/deduplication.py (PDFTracker, deduplication logic)
- core/embeddings.py (embedding generation)
- FAISS (vector indexing)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import numpy as np
import faiss

# Phase 3 Tool 1
from tools.extraction_tools import ExtractionTools

# Phase 2 Core
from core.deduplication import (
    PDFTracker,
    check_pdf_status,
    check_content_duplication,
    deduplicate_chunks,
    get_existing_chunk_signatures
)
from core.embeddings import generate_embeddings_batch

# Phase 1 Utils
from utils.file_utils import list_files, read_json, write_json
from config.settings import (
    INDEX_PATH,
    INPUT_FOLDER,
    FAISS_INDEX_FILE,
    CHUNKS_FILE,
    METADATA_FILE,
    TRACKER_FILE,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)


class KBTools:
    """
    Knowledge Base management with deduplication and tracking.
    
    Features:
    - Build FAISS index from PDFs
    - PDF-level deduplication (content signatures)
    - Chunk-level deduplication (chunk signatures)
    - Incremental updates
    - Auto-sync (detect new/modified PDFs)
    """
    
    def __init__(self):
        """Initialize KB Tools."""
        self.extraction_tools = ExtractionTools()
        self.tracker = PDFTracker(tracker_file=TRACKER_FILE)
        
        # FAISS index and data
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
        logger.info("KBTools initialized")
    
    def check_pdf_status(self, pdf_path: str) -> Dict[str, Any]:
        """
        Check if PDF is new, modified, duplicate, or unchanged.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            {
                'status': 'new' | 'modified' | 'duplicate_content' | 'unchanged',
                'filename': str,
                'file_hash': str,
                'existing_info': dict | None,
                'recommendation': str
            }
        """
        pdf_path_obj = Path(pdf_path)
        
        if not pdf_path_obj.exists():
            return {
                'status': 'error',
                'filename': pdf_path_obj.name,
                'error': f'File not found: {pdf_path}'
            }
        
        # Use core deduplication check
        status_info = check_pdf_status(pdf_path_obj, self.tracker)
        
        logger.info(f"PDF status for {pdf_path_obj.name}: {status_info['status']}")
        
        return status_info
    

    def build_index(
        self,
        input_folder: Optional[str] = None,
        force: bool = False,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Build FAISS index from scratch with deduplication.
        """
        import time
        start_time = time.time()
        
        input_folder = Path(input_folder) if input_folder else INPUT_FOLDER
        
        logger.info(f"Building knowledge base from: {input_folder}")
        
        if not input_folder.exists():
            return {
                'success': False,
                'error': f'Input folder not found: {input_folder}'
            }
        
        # Check if index already exists
        if FAISS_INDEX_FILE.exists() and not force:
            logger.warning("Index already exists. Use force=True to rebuild.")
            return {
                'success': False,
                'error': 'Index already exists. Use force=True to rebuild.'
            }
        
        # Find all PDFs
        pdf_files = list_files(input_folder, pattern="*.pdf", recursive=recursive)
        
        if not pdf_files:
            return {
                'success': False,
                'error': f'No PDFs found in {input_folder}'
            }
        
        logger.info(f"Found {len(pdf_files)} PDF(s) to process")
        
        # Reset tracker if force rebuild
        if force:
            self.tracker.clear()
            self.tracker.save()
        
        # Stats
        stats = {
            'pdfs_processed': 0,
            'pdfs_new': 0,
            'pdfs_duplicate': 0,
            'pdfs_skipped': 0,
            'chunks_created': 0,
            'chunks_deduplicated': 0,
            'errors': {}
        }
        
        all_chunks = []
        seen_signatures = set()
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            try:
                # Step 1: Check PDF status
                status_info = self.check_pdf_status(str(pdf_path))
                
                if status_info['status'] == 'unchanged':
                    print(f"   ⏭️  Skipped (unchanged)")
                    stats['pdfs_skipped'] += 1
                    continue
                
                # Step 2: Extract text with chunking
                result = self.extraction_tools.process_pdf_with_chunking(
                    pdf_path=str(pdf_path),
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                if not result['success']:
                    stats['errors'][str(pdf_path)] = result['error']
                    print(f"   ❌ Extraction failed: {result['error']}")
                    continue
                
                # Access nested structure correctly
                extraction_data = result.get('extraction', {})
                file_hash = extraction_data.get('file_hash')
                content_sig = extraction_data.get('content_signature')
                total_pages = extraction_data.get('total_pages', 0)
                
                # Fallback: if extraction is not nested, try top level
                if not content_sig:
                    content_sig = result.get('content_signature')
                    file_hash = result.get('file_hash')
                    total_pages = result.get('total_pages', 0)
                
                if not content_sig:
                    error_msg = "Missing content_signature in extraction result"
                    stats['errors'][str(pdf_path)] = error_msg
                    print(f"   ❌ {error_msg}")
                    logger.error(f"Result structure: {list(result.keys())}")
                    continue
                
                # Step 3: Check content duplication
                dup_check = check_content_duplication(
                    content_signature=content_sig,
                    filename=pdf_path.name,
                    tracker=self.tracker
                )
                
                if dup_check['is_duplicate']:
                    print(f"   🔄 Duplicate of: {dup_check['canonical_filename']}")
                    # Mark as duplicate in tracker
                    self.tracker.add_or_update_pdf(
                        filename=pdf_path.name,
                        file_hash=file_hash,
                        content_signature=content_sig,
                        path=str(pdf_path.absolute()),
                        total_pages=total_pages,
                        total_chunks=0,
                        is_canonical=False,
                        duplicate_of=dup_check['canonical_filename']
                    )
                    stats['pdfs_duplicate'] += 1
                    continue
                
                # Step 4: Deduplicate chunks
                pdf_chunks = result.get('chunks', [])
                dedup_chunks, new_sigs, dup_count = deduplicate_chunks(
                    chunks=pdf_chunks,
                    existing_signatures=seen_signatures
                )
                
                # Update seen signatures
                seen_signatures.update(new_sigs)
                
                # Add to all chunks
                all_chunks.extend(dedup_chunks)
                
                print(f"   ✅ Extracted: {total_pages} pages, "
                      f"{len(dedup_chunks)} chunks (removed {dup_count} duplicates)")
                
                # Step 5: Update tracker
                self.tracker.add_or_update_pdf(
                    filename=pdf_path.name,
                    file_hash=file_hash,
                    content_signature=content_sig,
                    path=str(pdf_path.absolute()),
                    total_pages=total_pages,
                    total_chunks=len(dedup_chunks),
                    is_canonical=True
                )
                
                stats['pdfs_processed'] += 1
                stats['pdfs_new'] += 1
                stats['chunks_created'] += len(pdf_chunks)
                stats['chunks_deduplicated'] += dup_count
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
                stats['errors'][str(pdf_path)] = str(e)
        
        # Check if we have chunks to index
        if not all_chunks:
            return {
                'success': False,
                'error': 'No chunks to index (all PDFs were duplicates or failed)',
                **stats
            }
        
        print(f"\n📊 Total chunks to index: {len(all_chunks)}")
        
        # Step 6: Generate embeddings
        print(f"🔢 Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        
        try:
            embeddings = generate_embeddings_batch(
                texts=chunk_texts,
                show_progress=True
            )
            
            # generate_embeddings_batch returns list of embeddings directly
            if not embeddings or len(embeddings) == 0:
                return {
                    'success': False,
                    'error': 'Embedding generation returned empty results',
                    **stats
                }
            
            if len(embeddings) != len(chunk_texts):
                return {
                    'success': False,
                    'error': f'Embedding count mismatch: {len(embeddings)} vs {len(chunk_texts)} chunks',
                    **stats
                }
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Embedding generation failed: {str(e)}',
                **stats
            }
        
        # Step 7: Build FAISS index
        print(f"🏗️  Building FAISS index...")
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # L2 distance index
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        index.add(embeddings_array)
        
        self.index = index
        self.chunks = all_chunks
        
        # Step 8: Save everything
        print(f"💾 Saving index and metadata...")
        
        # Ensure directory exists
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        
        # Save chunks
        write_json(self.chunks, CHUNKS_FILE)
        
        # Save metadata
        self.metadata = {
            'total_pdfs': stats['pdfs_processed'],
            'total_chunks': len(all_chunks),
            'embedding_dimension': EMBEDDING_DIMENSION,
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'kb_version': self._calculate_kb_version(),
            'created_at': self._get_timestamp()
        }
        write_json(self.metadata, METADATA_FILE)
        
        # Save tracker
        self.tracker.save()
        
        processing_time = time.time() - start_time
        
        result = {
            'success': True,
            'pdfs_processed': stats['pdfs_processed'],
            'pdfs_new': stats['pdfs_new'],
            'pdfs_duplicate': stats['pdfs_duplicate'],
            'pdfs_skipped': stats['pdfs_skipped'],
            'chunks_created': stats['chunks_created'],
            'chunks_deduplicated': stats['chunks_deduplicated'],
            'chunks_indexed': len(all_chunks),
            'index_path': str(FAISS_INDEX_FILE),
            'kb_version': self.metadata['kb_version'],
            'errors': stats['errors'],
            'processing_time': round(processing_time, 2)
        }
        
        logger.info(f"✅ Knowledge base built successfully in {processing_time:.2f}s")
        
        return result




    def load_index(self) -> Dict[str, Any]:
        """
        Load existing FAISS index from disk.
        
        Returns:
            {
                'success': bool,
                'total_chunks': int,
                'kb_version': str,
                'metadata': dict,
                'error': str | None
            }
        """
        try:
            if not FAISS_INDEX_FILE.exists():
                return {
                    'success': False,
                    'error': 'Index file not found. Build index first.'
                }
            
            # Load FAISS index
            self.index = faiss.read_index(str(FAISS_INDEX_FILE))
            
            # Load chunks
            self.chunks = read_json(CHUNKS_FILE, default=[])
            
            # Load metadata
            self.metadata = read_json(METADATA_FILE, default={})
            
            # Load tracker
            self.tracker.load()
            
            logger.info(f"✅ Loaded knowledge base: {len(self.chunks)} chunks")
            
            return {
                'success': True,
                'total_chunks': len(self.chunks),
                'kb_version': self.metadata.get('kb_version', 'unknown'),
                'metadata': self.metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading index: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Failed to load index: {str(e)}'
            }
    
    def get_kb_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Statistics dict
        """
        tracker_stats = self.tracker.get_stats()
        
        return {
            'index_loaded': self.index is not None,
            'total_chunks': len(self.chunks),
            'kb_version': self.metadata.get('kb_version', 'unknown'),
            'tracker_stats': tracker_stats,
            'metadata': self.metadata
        }
    
    def _calculate_kb_version(self) -> str:
        """Calculate KB version hash based on tracker state."""
        import hashlib
        
        # Create version string from tracker data
        version_data = json.dumps(self.tracker.tracker_data, sort_keys=True)
        version_hash = hashlib.sha256(version_data.encode()).hexdigest()
        
        return version_hash[:16]  # Short version
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


# ============================================================================
# Convenience Functions
# ============================================================================

def build_knowledge_base(
    input_folder: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to build knowledge base.
    
    Args:
        input_folder: Folder containing PDFs
        force: Force rebuild
        
    Returns:
        Build result dict
    """
    kb = KBTools()
    return kb.build_index(input_folder=input_folder, force=force)


def load_knowledge_base() -> Tuple[KBTools, Dict[str, Any]]:
    """
    Convenience function to load knowledge base.
    
    Returns:
        Tuple of (KBTools instance, load result dict)
    """
    kb = KBTools()
    result = kb.load_index()
    return kb, result
