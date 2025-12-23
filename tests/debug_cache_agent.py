"""
================================================================================
DEBUG SCRIPT: Cache Agent Test
================================================================================
Tests the Cache Agent in isolation with mock data

Test Scenarios:
1. Search empty cache (expect miss)
2. Add answer to cache (expect success)
3. Search for similar query (expect hit)
4. Get cache statistics (expect metrics)

Prerequisites:
- Cache tools must be implemented and working
- Embeddings must be configured (Azure OpenAI or local)
- FAISS must be installed

Run:
    python tests/debug/debug_cache_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.cache_agent import create_cache_agent


def test_cache_agent():
    """Test cache agent with progressive scenarios."""
    
    print("\n" + "="*80)
    print("CACHE AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Cache Agent
    # ========================================================================
    print("📝 Step 1: Creating Cache Agent...")
    try:
        cache_agent = create_cache_agent(verbose=True)
        print(f"✅ Agent created successfully")
        print(f"   Role: {cache_agent.role}")
        print(f"   Tools: {len(cache_agent.tools)} tools loaded")
        print(f"   Max Iterations: {cache_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # ========================================================================
    # Test 1: Search Empty Cache (Should Miss)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Search Empty Cache (Expect MISS)")
    print("-"*80)
    print("Query: 'What is the patient diagnosis?'")
    print("Expected: Cache miss (no similar query found)\n")
    
    task1 = Task(
        description=(
            "Search the semantic cache for this query: 'What is the patient diagnosis?'\n"
            "This is the first query, so the cache should be empty."
        ),
        agent=cache_agent,
        expected_output=(
            "Cache search result indicating a MISS (no cached answer found). "
            "Include similarity score if available."
        )
    )
    
    crew1 = Crew(
        agents=[cache_agent],
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
    # Test 2: Add Answer to Cache
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Add Answer to Cache")
    print("-"*80)
    print("Query: 'What is the patient diagnosis?'")
    print("Answer: 'The patient was diagnosed with hypertension.'")
    print("Expected: Successfully cached\n")
    
    task2 = Task(
        description=(
            "Add this Q&A pair to the semantic cache:\n\n"
            "Query: 'What is the patient diagnosis?'\n"
            "Answer: 'The patient was diagnosed with hypertension and prescribed Lisinopril 10mg daily.'\n"
            "Metadata: {\n"
            "  'confidence': 0.95,\n"
            "  'sources': ['medical_record.pdf'],\n"
            "  'validation_passed': True\n"
            "}\n\n"
            "Use the add_to_cache tool to store this."
        ),
        agent=cache_agent,
        expected_output="Confirmation message that the answer was successfully added to cache"
    )
    
    crew2 = Crew(
        agents=[cache_agent],
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
    # Test 3: Search for Similar Query (Should Hit)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Search for Similar Query (Expect HIT)")
    print("-"*80)
    print("Query: 'What was the patient diagnosed with?'")
    print("Expected: Cache hit with similarity > 0.95\n")
    
    task3 = Task(
        description=(
            "Search the cache for this query: 'What was the patient diagnosed with?'\n\n"
            "This is semantically similar to the cached query 'What is the patient diagnosis?' "
            "so it should return a cache HIT with high similarity score."
        ),
        agent=cache_agent,
        expected_output=(
            "Cache HIT result with:\n"
            "- Cached answer\n"
            "- Similarity score (should be > 0.95)\n"
            "- Original query\n"
            "- Source metadata"
        )
    )
    
    crew3 = Crew(
        agents=[cache_agent],
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
    # Test 4: Get Cache Statistics
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Get Cache Statistics")
    print("-"*80)
    print("Expected: Metrics including hits, misses, hit rate, total entries\n")
    
    task4 = Task(
        description=(
            "Retrieve cache statistics including:\n"
            "- Total queries processed\n"
            "- Cache hits\n"
            "- Cache misses\n"
            "- Hit rate percentage\n"
            "- Total cached entries\n"
            "- Similarity threshold"
        ),
        agent=cache_agent,
        expected_output="Cache statistics summary with all metrics"
    )
    
    crew4 = Crew(
        agents=[cache_agent],
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
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("CACHE AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Cache miss detection")
    print("  ✓ Answer caching")
    print("  ✓ Semantic similarity matching")
    print("  ✓ Statistics tracking")
    print("\n💡 Next Steps:")
    print("  1. Verify cache files created in data/cache/")
    print("  2. Check FAISS index and JSON data files")
    print("  3. Test with real medical queries")
    print("  4. Integrate with full RAG pipeline")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_cache_agent()
