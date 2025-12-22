"""
================================================================================
DEBUG SCRIPT: Retrieval Agent Test
================================================================================
Tests the Retrieval Agent in isolation with mock/real queries

Test Scenarios:
1. Single query search (basic FAISS retrieval)
2. Multi-query search with fusion
3. Distance threshold filtering
4. Get retrieval statistics
5. Test with HyDE-enhanced query

Prerequisites:
- FAISS index must be built (run indexing pipeline first)
- Knowledge base documents must be indexed
- Retrieval tools must be implemented
- Embeddings configured (Azure OpenAI)

Run:
    python tests/debug/debug_retrieval_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.retrieval_agent import create_retrieval_agent


def test_retrieval_agent():
    """Test retrieval agent with various search scenarios."""
    
    print("\n" + "="*80)
    print("RETRIEVAL AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Retrieval Agent
    # ========================================================================
    print("📝 Step 1: Creating Retrieval Agent...")
    try:
        retrieval_agent = create_retrieval_agent(verbose=True)
        print(f"✅ Agent created successfully")
        print(f"   Role: {retrieval_agent.role}")
        print(f"   Tools: {len(retrieval_agent.tools)} tools loaded")
        print(f"   Max Iterations: {retrieval_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Single Query Search
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Single Query Search")
    print("-"*80)
    print("Query: 'patient diagnosis hypertension'")
    print("Expected: Top 5 relevant chunks with similarity scores\n")
    
    task1 = Task(
        description=(
            "Search the knowledge base for this query:\n\n"
            "'patient diagnosis hypertension'\n\n"
            "Use the search_tool to perform a single query semantic search. "
            "Return the top 5 most relevant document chunks with:\n"
            "- Chunk text (first 200 characters)\n"
            "- Similarity score (0.0 to 1.0)\n"
            "- Source document\n"
            "- Chunk ID/index"
        ),
        agent=retrieval_agent,
        expected_output=(
            "Search results containing:\n"
            "- Number of chunks retrieved (target: 5)\n"
            "- Each chunk with similarity score, text preview, source\n"
            "- Highest similarity score should be > 0.7 for relevant results\n"
            "- Results ranked by similarity (highest first)"
        )
    )
    
    crew1 = Crew(
        agents=[retrieval_agent],
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
    # Test 2: Multi-Query Search with Fusion
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Multi-Query Search with Fusion")
    print("-"*80)
    print("Queries: 3 variations of the same question")
    print("Expected: Fused results using RRF or max-score\n")
    
    task2 = Task(
        description=(
            "Perform multi-query search with these query variations:\n\n"
            "1. 'What is the patient diagnosis?'\n"
            "2. 'What was the patient diagnosed with?'\n"
            "3. 'What medical condition does the patient have?'\n\n"
            "Use search_multiple_queries_tool to:\n"
            "- Search each query independently\n"
            "- Fuse results using Reciprocal Rank Fusion (RRF)\n"
            "- Return top 5 chunks after fusion\n"
            "- Show which queries each chunk matched"
        ),
        agent=retrieval_agent,
        expected_output=(
            "Fused search results with:\n"
            "- Top 5 chunks ranked by fusion score\n"
            "- Fusion method used (RRF or max-score)\n"
            "- How many queries each chunk matched\n"
            "- Combined similarity scores"
        )
    )
    
    crew2 = Crew(
        agents=[retrieval_agent],
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
    # Test 3: Search with Distance Threshold
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Search with Distance Threshold Filtering")
    print("-"*80)
    print("Query: Generic query that might return irrelevant results")
    print("Expected: Only chunks above similarity threshold\n")
    
    task3 = Task(
        description=(
            "Search for 'treatment plan' with a strict distance threshold:\n\n"
            "Query: 'treatment plan'\n"
            "Parameters:\n"
            "- top_k: 10 (retrieve more candidates)\n"
            "- distance_threshold: 0.75 (filter out low-similarity chunks)\n\n"
            "This tests that the retrieval system filters out irrelevant results "
            "even when requesting many chunks."
        ),
        agent=retrieval_agent,
        expected_output=(
            "Filtered search results:\n"
            "- Only chunks with similarity > 0.75\n"
            "- May return fewer than 10 chunks if threshold filters some out\n"
            "- All results should be highly relevant to 'treatment plan'"
        )
    )
    
    crew3 = Crew(
        agents=[retrieval_agent],
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
    # Test 4: Get Retrieval Statistics
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Get Retrieval Statistics")
    print("-"*80)
    print("Expected: Index stats, total chunks, embedding dimension\n")
    
    task4 = Task(
        description=(
            "Get retrieval system statistics:\n\n"
            "Use get_retrieval_stats_tool to retrieve:\n"
            "- Total indexed chunks\n"
            "- Embedding dimension (should be 3072 for text-embedding-3-large)\n"
            "- Index type (FAISS IndexFlatL2 or similar)\n"
            "- Knowledge base version/timestamp\n"
            "- Total source documents"
        ),
        agent=retrieval_agent,
        expected_output=(
            "Statistics summary:\n"
            "- Indexed chunks: [number]\n"
            "- Embedding dimension: 3072\n"
            "- Index type: FAISS [type]\n"
            "- Last updated: [timestamp]\n"
            "- Source documents: [number]"
        )
    )
    
    crew4 = Crew(
        agents=[retrieval_agent],
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
    # Test 5: HyDE-Enhanced Query Search
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Search with HyDE-Enhanced Query")
    print("-"*80)
    print("Using hypothetical answer for better semantic matching\n")
    
    task5 = Task(
        description=(
            "Search using a HyDE (Hypothetical Document Embeddings) query:\n\n"
            "Instead of searching for 'What treatment was provided for hypertension?', "
            "search for this hypothetical answer:\n\n"
            "'The patient was treated for essential hypertension with Lisinopril 10mg "
            "once daily. Treatment plan included lifestyle modifications with low-sodium "
            "diet and regular exercise. Blood pressure monitored weekly.'\n\n"
            "HyDE improves retrieval by searching for text similar to expected answers "
            "rather than question text."
        ),
        agent=retrieval_agent,
        expected_output=(
            "HyDE search results:\n"
            "- Chunks describing actual hypertension treatments\n"
            "- Higher quality matches than question-based search\n"
            "- Results should contain medication names, dosages, lifestyle advice"
        )
    )
    
    crew5 = Crew(
        agents=[retrieval_agent],
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
    print("RETRIEVAL AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Single query semantic search")
    print("  ✓ Multi-query fusion retrieval")
    print("  ✓ Distance threshold filtering")
    print("  ✓ Retrieval statistics")
    print("  ✓ HyDE-enhanced search")
    print("\n💡 Next Steps:")
    print("  1. Verify FAISS index exists in data/kb/")
    print("  2. Check embedding dimension matches (3072)")
    print("  3. Test with real medical documents from your KB")
    print("  4. Tune distance threshold based on results")
    print("  5. Integrate with Reranking Agent for refinement")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_retrieval_agent()
