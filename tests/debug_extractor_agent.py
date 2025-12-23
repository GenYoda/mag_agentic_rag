"""
================================================================================
DEBUG SCRIPT: Extractor Agent Test
================================================================================
Tests the Extractor Agent in isolation

Test Scenarios:
1. Extract text from a simple PDF
2. Process PDF with chunking (full pipeline)
3. Extract metadata from document
4. Validate extraction quality
5. Handle corrupted/problematic PDF

Prerequisites:
- Extraction tools must be implemented
- Sample PDFs in data/documents/
- PyMuPDF (fitz) installed: pip install PyMuPDF
- Azure OpenAI configured (if using LLM for validation)

Run:
    python tests/debug/debug_extractor_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.extractor_agent import create_extractor_agent


def test_extractor_agent():
    """Test extractor agent with various document scenarios."""
    
    print("\n" + "="*80)
    print("EXTRACTOR AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Extractor Agent
    # ========================================================================
    print("📝 Step 1: Creating Extractor Agent...")
    try:
        extractor_agent = create_extractor_agent(verbose=True)
        print(f"✅ Agent created successfully")
        print(f"   Role: {extractor_agent.role}")
        print(f"   Tools: {len(extractor_agent.tools)} tools loaded")
        print(f"   Max Iterations: {extractor_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Extract Text from Simple PDF
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Extract Text from PDF")
    print("-"*80)
    print("PDF: data/documents/sample_medical_record.pdf")
    print("Expected: Raw text extracted page-by-page\n")
    
    task1 = Task(
        description=(
            "Extract text from this PDF:\n\n"
            "Path: 'data/documents/sample_medical_record.pdf'\n\n"
            "Use extract_pdf_tool to:\n"
            "1. Open the PDF file\n"
            "2. Extract text from each page\n"
            "3. Combine into full document text\n"
            "4. Return extracted text with page count\n\n"
            "This is the basic extraction without chunking."
        ),
        agent=extractor_agent,
        expected_output=(
            "Extraction result:\n"
            "- Total pages: N\n"
            "- Total characters: M\n"
            "- Text preview (first 500 chars)\n"
            "- Extraction method: PyMuPDF\n"
            "- Success: True"
        )
    )
    
    crew1 = Crew(
        agents=[extractor_agent],
        tasks=[task1],
        verbose=True
    )
    
    try:
        result1 = crew1.kickoff()
        print(f"\n✅ Test 1 Complete")
        print(f"Result: {result1}")
    except Exception as e:
        print(f"\n❌ Test 1 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Test 2: Process PDF with Chunking (Full Pipeline)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Process PDF with Chunking")
    print("-"*80)
    print("PDF: Extract + Chunk in one operation")
    print("Expected: Chunks ready for indexing\n")
    
    task2 = Task(
        description=(
            "Process this PDF with full extraction and chunking pipeline:\n\n"
            "Path: 'data/documents/sample_medical_record.pdf'\n"
            "Chunk size: 500 tokens\n"
            "Chunk overlap: 100 tokens\n\n"
            "Use process_pdf_with_chunking_tool to:\n"
            "1. Extract all text from PDF\n"
            "2. Split into semantic chunks\n"
            "3. Add metadata to each chunk (source, page numbers, chunk ID)\n"
            "4. Generate content signatures for deduplication\n"
            "5. Return list of chunks ready for embedding\n\n"
            "This is the recommended method for KB ingestion."
        ),
        agent=extractor_agent,
        expected_output=(
            "Processing result:\n"
            "- Total pages: N\n"
            "- Total chunks: M\n"
            "- Average chunk size: ~500 tokens\n"
            "- Chunks with metadata: Yes\n"
            "- Content signature: hash...\n"
            "- File hash: hash...\n"
            "- Ready for indexing: True"
        )
    )
    
    crew2 = Crew(
        agents=[extractor_agent],
        tasks=[task2],
        verbose=True
    )
    
    try:
        result2 = crew2.kickoff()
        print(f"\n✅ Test 2 Complete")
        print(f"Result: {result2}")
    except Exception as e:
        print(f"\n❌ Test 2 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Test 3: Extract Metadata from Document
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Extract Document Metadata")
    print("-"*80)
    print("Get file and document metadata\n")
    
    task3 = Task(
        description=(
            "Extract comprehensive metadata from PDF:\n\n"
            "Path: 'data/documents/sample_medical_record.pdf'\n\n"
            "Use extract_metadata_tool to get:\n"
            "- File metadata: name, size, modification date\n"
            "- PDF metadata: page count, author, title, creation date\n"
            "- Content hash (SHA-256)\n"
            "- Content signature for deduplication\n"
            "- PDF properties (encrypted, scanned, etc.)\n\n"
            "This metadata is used by KB Agent for tracking."
        ),
        agent=extractor_agent,
        expected_output=(
            "Metadata:\n"
            "{\n"
            "  'filename': 'sample_medical_record.pdf',\n"
            "  'file_size': X bytes,\n"
            "  'modified_date': 'timestamp',\n"
            "  'total_pages': N,\n"
            "  'author': 'name or unknown',\n"
            "  'creation_date': 'date',\n"
            "  'file_hash': 'sha256...',\n"
            "  'content_signature': 'hash...',\n"
            "  'is_scanned': False,\n"
            "  'is_encrypted': False\n"
            "}"
        )
    )
    
    crew3 = Crew(
        agents=[extractor_agent],
        tasks=[task3],
        verbose=True
    )
    
    try:
        result3 = crew3.kickoff()
        print(f"\n✅ Test 3 Complete")
        print(f"Result: {result3}")
    except Exception as e:
        print(f"\n❌ Test 3 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Test 4: Validate Extraction Quality
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Validate Extraction Quality")
    print("-"*80)
    print("Check if extraction was successful and high-quality\n")
    
    task4 = Task(
        description=(
            "Validate extraction quality for processed document:\n\n"
            "Path: 'data/documents/sample_medical_record.pdf'\n\n"
            "Use validate_extraction_tool to check:\n"
            "1. Minimum text length (should have substantial content)\n"
            "2. Reasonable chunk count (not too few/many)\n"
            "3. No encoding errors or garbled text\n"
            "4. Medical content structure (sections, terminology)\n"
            "5. Completeness (all pages processed)\n\n"
            "Return validation report with quality score."
        ),
        agent=extractor_agent,
        expected_output=(
            "Validation result:\n"
            "- Quality score: 0.0-1.0\n"
            "- Text length: Sufficient/Insufficient\n"
            "- Chunk count: Reasonable/Too few/Too many\n"
            "- Encoding: OK/Issues detected\n"
            "- Medical content: Valid/Invalid\n"
            "- Warnings: [list of issues]\n"
            "- Overall: Pass/Fail"
        )
    )
    
    crew4 = Crew(
        agents=[extractor_agent],
        tasks=[task4],
        verbose=True
    )
    
    try:
        result4 = crew4.kickoff()
        print(f"\n✅ Test 4 Complete")
        print(f"Result: {result4}")
    except Exception as e:
        print(f"\n❌ Test 4 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Test 5: Handle Problematic PDF
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Handle Problematic/Corrupted PDF")
    print("-"*80)
    print("Test error handling and recovery\n")
    
    task5 = Task(
        description=(
            "Attempt to extract from a problematic PDF:\n\n"
            "Scenarios to test:\n"
            "1. Corrupted PDF file\n"
            "2. Encrypted PDF without password\n"
            "3. Scanned PDF requiring OCR\n"
            "4. Empty PDF (no content)\n\n"
            "Expected behavior:\n"
            "- Graceful error handling\n"
            "- Detailed error messages\n"
            "- Suggestions for resolution\n"
            "- Partial results if possible\n\n"
            "Test with: 'data/documents/corrupted.pdf' (if available)\n"
            "If file doesn't exist, that's also a valid test case."
        ),
        agent=extractor_agent,
        expected_output=(
            "Error handling result:\n"
            "- Success: False (expected)\n"
            "- Error type: FileNotFound/Corrupted/Encrypted/etc.\n"
            "- Error message: Detailed explanation\n"
            "- Suggested action: How to fix\n"
            "- Partial result: Any data extracted before error\n"
            "- Log: Full error traceback for debugging"
        )
    )
    
    crew5 = Crew(
        agents=[extractor_agent],
        tasks=[task5],
        verbose=True
    )
    
    try:
        result5 = crew5.kickoff()
        print(f"\n✅ Test 5 Complete")
        print(f"Result: {result5}")
    except Exception as e:
        print(f"\n❌ Test 5 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("EXTRACTOR AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Raw text extraction from PDF")
    print("  ✓ Full processing pipeline with chunking")
    print("  ✓ Metadata extraction")
    print("  ✓ Extraction quality validation")
    print("  ✓ Error handling for problematic files")
    print("\n💡 Next Steps:")
    print("  1. Verify PyMuPDF installed: pip install PyMuPDF")
    print("  2. Place sample PDFs in data/documents/")
    print("  3. Test with real medical documents")
    print("  4. Tune chunk size/overlap in settings")
    print("  5. Integrate with KB Agent for full indexing pipeline")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_extractor_agent()
