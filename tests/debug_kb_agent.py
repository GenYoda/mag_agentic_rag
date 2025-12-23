"""
================================================================================
DEBUG SCRIPT: KB Agent Test
================================================================================
Tests the KB Agent in isolation

Test Scenarios:
1. Get current KB statistics
2. Index a single document
3. Index a directory (batch)
4. Update an existing document
5. Delete a document and rebuild index

Prerequisites:
- KB tools must be implemented
- Sample documents in data/documents/
- Write access to data/kb/ directory
- Azure OpenAI configured for embeddings

Run:
    python tests/debug/debug_kb_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.kb_agent import create_kb_agent


def test_kb_agent():
    """Test KB agent with various management operations."""
    
    print("\n" + "="*80)
    print("KB AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create KB Agent
    # ========================================================================
    print("📝 Step 1: Creating KB Agent...")
    try:
        kb_agent = create_kb_agent(verbose=True)
        print(f"✅ Agent created successfully")
        print(f"   Role: {kb_agent.role}")
        print(f"   Tools: {len(kb_agent.tools)} tools loaded")
        print(f"   Max Iterations: {kb_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Get KB Statistics
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Get Knowledge Base Statistics")
    print("-"*80)
    print("Expected: Current KB metrics (documents, chunks, version)\n")
    
    task1 = Task(
        description=(
            "Get comprehensive knowledge base statistics:\n\n"
            "Use get_kb_stats_tool to retrieve:\n"
            "- Total indexed documents\n"
            "- Total chunks in FAISS index\n"
            "- Embedding dimension (should be 3072)\n"
            "- Index type (FAISS IndexFlatL2)\n"
            "- KB version hash\n"
            "- Last update timestamp\n"
            "- Index file size"
        ),
        agent=kb_agent,
        expected_output=(
            "KB statistics report:\n"
            "{\n"
            "  'total_documents': X,\n"
            "  'total_chunks': Y,\n"
            "  'embedding_dimension': 3072,\n"
            "  'index_type': 'FAISS IndexFlatL2',\n"
            "  'kb_version': 'hash...',\n"
            "  'last_updated': 'timestamp',\n"
            "  'index_size_mb': Z\n"
            "}"
        )
    )
    
    crew1 = Crew(
        agents=[kb_agent],
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
    # Test 2: Index Single Document
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Index Single Document")
    print("-"*80)
    print("Document: data/documents/sample_medical_record.pdf")
    print("Expected: Document indexed with chunk count\n")
    
    task2 = Task(
        description=(
            "Index a single medical document:\n\n"
            "Document path: 'data/documents/sample_medical_record.pdf'\n\n"
            "Use index_document_tool to:\n"
            "1. Extract text from PDF\n"
            "2. Chunk text into semantic units (~500 tokens)\n"
            "3. Generate embeddings for each chunk\n"
            "4. Add to FAISS index\n"
            "5. Store metadata (filename, page numbers, timestamps)\n"
            "6. Update KB version hash\n\n"
            "Return: Number of chunks indexed and KB version"
        ),
        agent=kb_agent,
        expected_output=(
            "Indexing result:\n"
            "- Document: sample_medical_record.pdf\n"
            "- Chunks indexed: N\n"
            "- Embedding dimension: 3072\n"
            "- KB version updated: new_hash\n"
            "- Status: Success"
        )
    )
    
    crew2 = Crew(
        agents=[kb_agent],
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
    # Test 3: Index Directory (Batch)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Index Directory (Batch Operation)")
    print("-"*80)
    print("Directory: data/documents/")
    print("Expected: All documents indexed with progress tracking\n")
    
    task3 = Task(
        description=(
            "Index all documents in a directory:\n\n"
            "Directory: 'data/documents/'\n\n"
            "Use index_directory_tool to:\n"
            "1. Scan directory for supported files (.pdf, .docx, .txt)\n"
            "2. Process each document sequentially\n"
            "3. Track progress (X of Y documents)\n"
            "4. Handle errors gracefully (skip corrupted files)\n"
            "5. Provide summary of successful/failed indexing\n"
            "6. Update KB version after completion"
        ),
        agent=kb_agent,
        expected_output=(
            "Batch indexing result:\n"
            "- Total files found: N\n"
            "- Successfully indexed: M\n"
            "- Failed: P (with error details)\n"
            "- Total chunks added: Q\n"
            "- KB version: new_hash\n"
            "- Duration: X seconds"
        )
    )
    
    crew3 = Crew(
        agents=[kb_agent],
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
    # Test 4: Update Existing Document
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Update Existing Document")
    print("-"*80)
    print("Update document that changed and reindex\n")
    
    task4 = Task(
        description=(
            "Update an existing document in the KB:\n\n"
            "Document: 'data/documents/sample_medical_record.pdf'\n"
            "(Assume this file was modified after initial indexing)\n\n"
            "Use update_document_tool to:\n"
            "1. Find existing chunks for this document\n"
            "2. Remove old chunks from index\n"
            "3. Reindex updated document\n"
            "4. Update metadata with new timestamp\n"
            "5. Increment KB version\n\n"
            "This ensures KB stays current with document changes."
        ),
        agent=kb_agent,
        expected_output=(
            "Update result:\n"
            "- Document: sample_medical_record.pdf\n"
            "- Old chunks removed: X\n"
            "- New chunks added: Y\n"
            "- Net change: +/- Z chunks\n"
            "- KB version: updated_hash\n"
            "- Status: Success"
        )
    )
    
    crew4 = Crew(
        agents=[kb_agent],
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
    # Test 5: Delete Document and Rebuild Index
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Delete Document from KB")
    print("-"*80)
    print("Remove document and rebuild index\n")
    
    task5 = Task(
        description=(
            "Delete a document from the knowledge base:\n\n"
            "Document to delete: 'data/documents/old_record.pdf'\n\n"
            "Use delete_document_tool to:\n"
            "1. Identify all chunks from this document\n"
            "2. Remove chunks from FAISS index\n"
            "3. Update metadata to remove document entry\n"
            "4. Rebuild index to optimize structure\n"
            "5. Update KB version\n\n"
            "Since FAISS doesn't support efficient deletion, we rebuild the index."
        ),
        agent=kb_agent,
        expected_output=(
            "Deletion result:\n"
            "- Document deleted: old_record.pdf\n"
            "- Chunks removed: N\n"
            "- Index rebuilt: Yes\n"
            "- New total chunks: M\n"
            "- KB version: new_hash\n"
            "- Status: Success"
        )
    )
    
    crew5 = Crew(
        agents=[kb_agent],
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
    print("KB AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ KB statistics retrieval")
    print("  ✓ Single document indexing")
    print("  ✓ Batch directory indexing")
    print("  ✓ Document update and reindexing")
    print("  ✓ Document deletion with index rebuild")
    print("\n💡 Next Steps:")
    print("  1. Verify FAISS index files in data/kb/")
    print("  2. Check metadata.json for document tracking")
    print("  3. Test with real medical documents")
    print("  4. Monitor index size and performance")
    print("  5. Set up automated reindexing pipeline")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_kb_agent()
