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
from crewai.tools import tool

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


    def reset_kb(self) -> Dict[str, Any]:
        """
        🆕 Reset the entire knowledge base - clear all data and start fresh.
        
        Deletes:
        - FAISS index file
        - Chunks file
        - Metadata file
        - Tracker file (PDF hashes)
        
        Returns:
            {'success': bool, 'message': str, 'files_deleted': list}
        """
        try:
            deleted_files = []
            
            # Delete FAISS index
            if FAISS_INDEX_FILE.exists():
                FAISS_INDEX_FILE.unlink()
                deleted_files.append(str(FAISS_INDEX_FILE))
                logger.info(f"✅ Deleted: {FAISS_INDEX_FILE}")
            
            # Delete chunks
            if CHUNKS_FILE.exists():
                CHUNKS_FILE.unlink()
                deleted_files.append(str(CHUNKS_FILE))
                logger.info(f"✅ Deleted: {CHUNKS_FILE}")
            
            # Delete metadata
            if METADATA_FILE.exists():
                METADATA_FILE.unlink()
                deleted_files.append(str(METADATA_FILE))
                logger.info(f"✅ Deleted: {METADATA_FILE}")
            
            # Delete tracker (clears all PDF hashes)
            if TRACKER_FILE.exists():
                TRACKER_FILE.unlink()
                deleted_files.append(str(TRACKER_FILE))
                logger.info(f"✅ Deleted: {TRACKER_FILE}")
            
            # Clear in-memory data
            self.tracker.clear()
            self.tracker.save()
            self.index = None
            self.chunks = []
            self.metadata = {}
            
            logger.info(f"🔄 KB reset complete - deleted {len(deleted_files)} files")
            
            return {
                'success': True,
                'message': f'KB reset complete - {len(deleted_files)} files deleted',
                'files_deleted': deleted_files
            }
            
        except Exception as e:
            logger.error(f"❌ Reset failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Reset failed: {str(e)}'
            }



    
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
                texts=chunk_texts
               
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
# ✅ ADD CREWAI TOOL WRAPPERS HERE
# ============================================================================

# Singleton instance for tools
_kb_instance = None

def _get_kb_instance() -> KBTools:
    """Get or create KB Tools singleton instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KBTools()
        # Auto-load index if it exists
        if FAISS_INDEX_FILE.exists():
            _kb_instance.load_index()
    return _kb_instance


@tool("Index Document")
def index_document_tool(pdf_path: str) -> dict:
    """
    Index a single PDF document into the knowledge base.
    
    Args:
        pdf_path: Path to PDF file to index
        
    Returns:
        dict: {success, chunks_indexed, kb_version, error}
    """
    kb = _get_kb_instance()
    
    # For single document, we use build_index with specific folder
    # Or you can add a dedicated add_document method to KBTools
    
    try:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            return {
                'success': False,
                'error': f'File not found: {pdf_path}'
            }
        
        # Build index from parent directory (will handle deduplication)
        result = kb.build_index(
            input_folder=str(pdf_path_obj.parent),
            force=False,
            recursive=False
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error indexing document: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@tool("Index Directory")
def index_directory_tool(directory_path: str, force: bool = False) -> dict:
    """
    Index all PDF documents in a directory.
    
    Args:
        directory_path: Path to directory containing PDFs
        force: Force rebuild of existing index
        
    Returns:
        dict: {success, pdfs_processed, chunks_indexed, kb_version, errors}
    """
    kb = _get_kb_instance()
    
    try:
        result = kb.build_index(
            input_folder=directory_path,
            force=force,
            recursive=True
        )
        return result
        
    except Exception as e:
        logger.error(f"Error indexing directory: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@tool("Update Document")
def update_document_tool(pdf_path: str) -> dict:
    """
    Update an existing document in the knowledge base.
    Re-indexes if modified, skips if unchanged.
    
    Args:
        pdf_path: Path to PDF file to update
        
    Returns:
        dict: {success, status, chunks_updated, kb_version}
    """
    kb = _get_kb_instance()
    
    try:
        # Check PDF status first
        status_info = kb.check_pdf_status(pdf_path)
        
        if status_info['status'] == 'unchanged':
            return {
                'success': True,
                'status': 'unchanged',
                'message': 'Document unchanged, no update needed'
            }
        
        # If modified or new, reindex
        result = kb.build_index(
            input_folder=str(Path(pdf_path).parent),
            force=False,
            recursive=False
        )
        
        return {
            'success': result['success'],
            'status': status_info['status'],
            'chunks_updated': result.get('chunks_indexed', 0),
            'kb_version': result.get('kb_version')
        }
        
    except Exception as e:
        logger.error(f"Error updating document: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@tool("Delete Document")
def delete_document_tool(filename: str) -> dict:
    """
    Delete a document from the knowledge base.
    Removes chunks and rebuilds index.
    
    Args:
        filename: Name of PDF file to delete
        
    Returns:
        dict: {success, chunks_removed, kb_version}
    """
    kb = _get_kb_instance()
    
    try:
        # Find chunks from this document
        chunks_to_remove = [
            chunk for chunk in kb.chunks 
            if chunk.get('source') == filename
        ]
        
        if not chunks_to_remove:
            return {
                'success': False,
                'error': f'No chunks found for document: {filename}'
            }
        
        # Remove chunks
        kb.chunks = [
            chunk for chunk in kb.chunks 
            if chunk.get('source') != filename
        ]
        
        # Rebuild index
        if kb.chunks:
            chunk_texts = [chunk['text'] for chunk in kb.chunks]
            embeddings = generate_embeddings_batch(chunk_texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            kb.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
            kb.index.add(embeddings_array)
            
            # Save updated index
            faiss.write_index(kb.index, str(FAISS_INDEX_FILE))
            write_json(kb.chunks, CHUNKS_FILE)
            
            # Update metadata
            kb.metadata['total_chunks'] = len(kb.chunks)
            kb.metadata['kb_version'] = kb._calculate_kb_version()
            write_json(kb.metadata, METADATA_FILE)
        else:
            # No chunks left, remove index files
            FAISS_INDEX_FILE.unlink(missing_ok=True)
            CHUNKS_FILE.unlink(missing_ok=True)
        
        # Update tracker
        kb.tracker.remove_pdf(filename)
        kb.tracker.save()
        
        return {
            'success': True,
            'chunks_removed': len(chunks_to_remove),
            'kb_version': kb.metadata.get('kb_version')
        }
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@tool("Get KB Statistics")
def get_kb_stats_tool() -> dict:
    """
    Get knowledge base statistics and metadata.
    
    Returns:
        dict: {total_documents, total_chunks, kb_version, tracker_stats, metadata}
    """
    kb = _get_kb_instance()
    
    try:
        stats = kb.get_kb_stats()
        return {
            'success': True,
            **stats
        }
        
    except Exception as e:
        logger.error(f"Error getting KB stats: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@tool("Rebuild Index")
def rebuild_index_tool(force: bool = True) -> dict:
    """
    Rebuild the entire knowledge base index from scratch.
    
    Args:
        force: Force rebuild even if index exists
        
    Returns:
        dict: {success, pdfs_processed, chunks_indexed, kb_version}
    """
    kb = _get_kb_instance()
    
    try:
        result = kb.build_index(
            input_folder=None,  # Uses default INPUT_FOLDER
            force=force,
            recursive=True
        )
        return result
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# Convenience Functions (keep existing)
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
