"""
Debug script to test core/deduplication.py

Tests:
1. PDF Tracker operations
2. PDF status checking
3. Content duplication detection
4. Chunk deduplication
5. Batch operations
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_deduplication():
    """Test deduplication system"""
    
    print("🧪 Testing core/deduplication.py...\n")
    
    try:
        from core.deduplication import (
            PDFTracker,
            check_pdf_status,
            check_content_duplication,
            calculate_chunk_signature,
            deduplicate_chunks,
            get_existing_chunk_signatures,
            scan_directory_for_changes,
            generate_deduplication_report,
            create_pdf_entry,
            mark_as_duplicate
        )
        
        print("✅ Import successful!\n")
        
        # Create temporary directory for tests
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tracker_file = temp_path / "test_tracker.json"
            
            # Test 1: PDF Tracker creation and operations
            print("Test 1: PDF Tracker Operations")
            print("-" * 40)
            
            tracker = PDFTracker(tracker_file=tracker_file)
            assert tracker.tracker_data == {}, "Tracker should start empty"
            print("✅ Tracker initialized")
            
            # Add PDF entry
            success = tracker.add_or_update_pdf(
                filename="test1.pdf",
                file_hash="abc123",
                content_signature="xyz789",
                path="/path/to/test1.pdf",
                total_pages=10,
                total_chunks=50
            )
            assert success, "Failed to add PDF"
            print("✅ PDF entry added")
            
            # Retrieve PDF info
            info = tracker.get_pdf_info("test1.pdf")
            assert info is not None, "Failed to retrieve PDF info"
            assert info['file_hash'] == "abc123", "Incorrect file hash"
            assert info['total_chunks'] == 50, "Incorrect chunk count"
            print(f"✅ PDF info retrieved: {info['filename'] if 'filename' in info else 'test1.pdf'}")
            
            # Get stats
            stats = tracker.get_stats()
            assert stats['total_pdfs_tracked'] == 1, "Incorrect PDF count"
            assert stats['canonical_pdfs'] == 1, "Incorrect canonical count"
            print(f"✅ Tracker stats: {stats['total_pdfs_tracked']} PDFs tracked\n")
            
            # Test 2: PDF status checking
            print("Test 2: PDF Status Checking")
            print("-" * 40)
            
            # Create test PDF file
            test_pdf = temp_path / "new_file.pdf"
            test_pdf.write_text("This is a test PDF content")
            
            # Check status (should be 'new')
            status_info = check_pdf_status(test_pdf, tracker)
            assert status_info['status'] == 'new', "Status should be 'new'"
            print(f"✅ New file detected: {status_info['status']}")
            
            # Add to tracker
            from utils.file_utils import calculate_file_hash
            file_hash = calculate_file_hash(test_pdf)
            tracker.add_or_update_pdf(
                filename="new_file.pdf",
                file_hash=file_hash,
                content_signature="content123",
                path=str(test_pdf)
            )
            
            # Check again (should be 'unchanged')
            status_info = check_pdf_status(test_pdf, tracker)
            assert status_info['status'] == 'unchanged', "Status should be 'unchanged'"
            print(f"✅ Unchanged file detected: {status_info['status']}")
            
            # Modify file
            test_pdf.write_text("Modified content")
            
            # Check again (should be 'modified')
            status_info = check_pdf_status(test_pdf, tracker)
            assert status_info['status'] == 'modified', "Status should be 'modified'"
            print(f"✅ Modified file detected: {status_info['status']}\n")
            
            # Test 3: Content duplication detection
            print("Test 3: Content Duplication Detection")
            print("-" * 40)
            
            # Check unique content
            dup_check = check_content_duplication("unique_signature", "test2.pdf", tracker)
            assert not dup_check['is_duplicate'], "Should not be duplicate"
            print("✅ Unique content detected")
            
            # Add another PDF with same content signature as test1.pdf
            tracker.add_or_update_pdf(
                filename="test_duplicate.pdf",
                file_hash="different_hash",
                content_signature="xyz789",  # Same as test1.pdf
                path="/path/to/test_duplicate.pdf",
                is_canonical=True
            )
            
            # Check duplicate
            dup_check = check_content_duplication("xyz789", "test3.pdf", tracker)
            assert dup_check['is_duplicate'], "Should be duplicate"
            assert dup_check['canonical_filename'] == "test1.pdf", "Wrong canonical file"
            print(f"✅ Duplicate content detected (canonical: {dup_check['canonical_filename']})\n")
            
            # Test 4: Chunk signature calculation
            print("Test 4: Chunk Signature Calculation")
            print("-" * 40)
            
            text1 = "Patient diagnosed with diabetes"
            text2 = "PATIENT DIAGNOSED WITH DIABETES"  # Different case
            
            sig1 = calculate_chunk_signature(text1)
            sig2 = calculate_chunk_signature(text2)
            
            assert sig1 == sig2, "Signatures should match (case-insensitive)"
            assert len(sig1) == 64, "SHA256 should be 64 chars"
            print(f"✅ Chunk signature calculated: {sig1[:16]}...")
            print(f"✅ Case-insensitive matching works\n")
            
            # Test 5: Chunk deduplication
            print("Test 5: Chunk Deduplication")
            print("-" * 40)
            
            # Create test chunks with duplicates
            chunks = [
                {'text': 'Patient presents with chest pain'},
                {'text': 'Blood pressure: 140/90'},
                {'text': 'Patient presents with chest pain'},  # Duplicate
                {'text': 'Heart rate: 88 bpm'},
                {'text': 'BLOOD PRESSURE: 140/90'},  # Duplicate (different case)
            ]
            
            unique_chunks, signatures, dup_count = deduplicate_chunks(chunks)
            
            assert len(unique_chunks) == 3, f"Should have 3 unique chunks, got {len(unique_chunks)}"
            assert dup_count == 2, f"Should remove 2 duplicates, got {dup_count}"
            print(f"✅ Deduplication: 5 chunks → {len(unique_chunks)} unique chunks")
            print(f"✅ Duplicates removed: {dup_count}")
            print(f"✅ New signatures generated: {len(signatures)}\n")
            
            # Test 6: Cross-document deduplication
            print("Test 6: Cross-Document Deduplication")
            print("-" * 40)
            
            # Extract signatures from first batch
            existing_sigs = get_existing_chunk_signatures(unique_chunks)
            print(f"✅ Extracted {len(existing_sigs)} existing signatures")
            
            # New chunks with some overlap
            new_chunks = [
                {'text': 'Patient presents with chest pain'},  # Exists in KB
                {'text': 'New symptom: shortness of breath'},  # New
            ]
            
            unique_new, new_sigs, dup_count = deduplicate_chunks(new_chunks, existing_sigs)
            
            assert len(unique_new) == 1, "Should have 1 unique chunk (1 duplicate)"
            assert dup_count == 1, "Should remove 1 duplicate"
            print(f"✅ Cross-document deduplication: {len(new_chunks)} → {len(unique_new)} unique")
            print(f"✅ Duplicates removed: {dup_count}\n")
            
            # Test 7: Batch directory scanning
            print("Test 7: Batch Directory Scanning")
            print("-" * 40)
            
            # Create test PDFs in directory
            pdf_dir = temp_path / "pdfs"
            pdf_dir.mkdir()
            
            (pdf_dir / "new1.pdf").write_text("New PDF 1")
            (pdf_dir / "new2.pdf").write_text("New PDF 2")
            
            categorized = scan_directory_for_changes(pdf_dir, tracker)
            
            assert len(categorized['new']) == 2, "Should detect 2 new PDFs"
            print(f"✅ Directory scan: {len(categorized['new'])} new PDFs")
            print(f"   • New: {len(categorized['new'])}")
            print(f"   • Modified: {len(categorized['modified'])}")
            print(f"   • Unchanged: {len(categorized['unchanged'])}\n")
            
            # Test 8: Deduplication report
            print("Test 8: Deduplication Report")
            print("-" * 40)
            
            report = generate_deduplication_report(
                tracker=tracker,
                chunks_deduplicated=5,
                pdfs_processed=3,
                pdfs_skipped_duplicate=1
            )
            
            assert 'pdf_deduplication' in report, "Report missing PDF section"
            assert 'chunk_deduplication' in report, "Report missing chunk section"
            print("✅ Deduplication report generated")
            print(f"   • PDFs tracked: {report['pdf_deduplication']['total_pdfs_tracked']}")
            print(f"   • Chunks deduplicated: {report['chunk_deduplication']['chunks_deduplicated']}\n")
            
            # Test 9: Helper functions
            print("Test 9: Helper Functions")
            print("-" * 40)
            
            test_pdf2 = temp_path / "helper_test.pdf"
            test_pdf2.write_text("Helper test content")
            
            entry = create_pdf_entry(
                pdf_path=test_pdf2,
                full_text="Helper test content",
                total_pages=1,
                total_chunks=5
            )
            
            assert 'filename' in entry, "Entry missing filename"
            assert 'content_signature' in entry, "Entry missing content signature"
            assert entry['total_chunks'] == 5, "Incorrect chunk count"
            print("✅ PDF entry created via helper")
            
            # Test mark as duplicate
            tracker.add_or_update_pdf(**entry)
            success = mark_as_duplicate(
                tracker=tracker,
                duplicate_filename="duplicate_test.pdf",
                canonical_filename=entry['filename'],
                pdf_path=test_pdf2
            )
            assert success, "Failed to mark as duplicate"
            
            dup_info = tracker.get_pdf_info("duplicate_test.pdf")
            assert dup_info['is_canonical'] == False, "Should not be canonical"
            assert dup_info['duplicate_of'] == entry['filename'], "Wrong canonical reference"
            print("✅ Duplicate marking works\n")
        
        print("🎉 ALL TESTS PASSED! 🎉\n")
        
        print("=" * 80)
        print("DEDUPLICATION SYSTEM VERIFIED")
        print("=" * 80)
        print("✅ PDF Tracker (load/save/query)")
        print("✅ PDF status checking (new/modified/unchanged)")
        print("✅ Content duplication detection")
        print("✅ Chunk signature calculation")
        print("✅ Within-document deduplication")
        print("✅ Cross-document deduplication")
        print("✅ Batch directory scanning")
        print("✅ Deduplication reporting")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_deduplication()
    sys.exit(0 if success else 1)
