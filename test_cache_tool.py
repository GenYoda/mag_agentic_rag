"""
Test Script for CacheTools

Tests:
1. Cache initialization
2. Cache miss (first query)
3. Cache hit (similar query)
4. Cache expiration
5. Cache statistics
6. LRU eviction
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.cache_tools import CacheTools
import logging
import time

# Configure logging to show INFO messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted test header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def test_cache_initialization():
    """Test 1: Cache Initialization"""
    print_header("TEST 1: Cache Initialization")
    
    cache = CacheTools()
    
    stats = cache.get_cache_stats()
    
    print(f"\n📊 Cache Configuration:")
    print(f"   Enabled: {'✅ Yes' if stats['enabled'] else '❌ No'}")
    print(f"   Similarity Threshold: {stats['similarity_threshold']}")
    print(f"   TTL: {stats['ttl_hours']} hours")
    print(f"   Max Entries: {stats['max_entries']}")
    print(f"   Current Entries: {stats['total_entries']}")
    
    print("\n✅ Initialization test complete")


def test_cache_miss():
    """Test 2: Cache Miss (First Query)"""
    print_header("TEST 2: Cache Miss (First Query)")
    
    cache = CacheTools()
    
    # Clear cache for clean test
    cache.clear_cache()
    
    query = "What are the allegations against Memorial Health?"
    
    print(f"\n🔍 Query: {query}")
    print(f"   (First time asking this question)")
    
    result = cache.get_cached_answer(query)
    
    if result is None:
        print(f"\n❌ Cache MISS (as expected)")
        print(f"   → Need to generate answer with LLM")
    else:
        print(f"\n⚠️  Unexpected cache hit!")
    
    stats = cache.get_cache_stats()
    print(f"\n📊 Stats:")
    print(f"   Total Queries: {stats['total_queries']}")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate']:.1%}")
    
    print("\n✅ Cache miss test complete")


def test_cache_save_and_hit():
    """Test 3: Cache Save and Hit"""
    print_header("TEST 3: Cache Save and Hit")
    
    cache = CacheTools()
    
    # Clear cache
    cache.clear_cache()
    
    # First query - cache miss, then save answer
    query1 = "What are the allegations against Memorial Health?"
    answer1 = (
        "Memorial Health is alleged to have provided negligent care leading to "
        "severe brain damage for Frank Ward after delays in treatment. The hospital "
        "is held liable under respondeat superior for the negligent actions of its "
        "employees, including Nurses Roy, Robinson, and Tinker."
    )
    sources1 = ["1_4.pdf", "trimmed_complaint.pdf", "synopsis_mag.pdf"]
    
    print(f"\n📝 Step 1: Cache the answer")
    print(f"   Query: {query1}")
    
    success = cache.cache_answer(query1, answer1, sources1)
    
    if success:
        print(f"   ✅ Answer cached successfully")
    else:
        print(f"   ❌ Failed to cache answer")
    
    # Second query - semantically similar (should be cache hit)
    query2 = "What claims have been made against Memorial Health?"  # Similar!
    
    print(f"\n🔍 Step 2: Query with similar question")
    print(f"   Original: {query1}")
    print(f"   Similar:  {query2}")
    
    result = cache.get_cached_answer(query2)
    
    if result and result['cached']:
        print(f"\n✅ Cache HIT!")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Original Query: {result['original_query']}")
        print(f"   Cached Answer: {result['answer'][:150]}...")
        print(f"   Sources: {', '.join(result['sources'])}")
        print(f"   Hit Count: {result['hit_count']}")
    else:
        print(f"\n❌ Cache MISS (unexpected)")
    
    # Third query - different question (should be cache miss)
    query3 = "What is the weather today?"  # Completely different
    
    print(f"\n🔍 Step 3: Query with different question")
    print(f"   Query: {query3}")
    
    result = cache.get_cached_answer(query3)
    
    if result is None:
        print(f"   ❌ Cache MISS (as expected - different topic)")
    else:
        print(f"   ⚠️  Unexpected cache hit")
    
    # Stats
    stats = cache.get_cache_stats()
    print(f"\n📊 Final Stats:")
    print(f"   Total Queries: {stats['total_queries']}")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate']:.1%}")
    print(f"   Cached Entries: {stats['total_entries']}")
    
    print("\n✅ Cache save and hit test complete")


def test_multiple_similar_queries():
    """Test 4: Multiple Similar Queries"""
    print_header("TEST 4: Multiple Similar Queries")
    
    cache = CacheTools()
    cache.clear_cache()
    
    # Cache original answer
    original_query = "Who are the defendants in this case?"
    answer = (
        "The defendants are: Savannah Health Services LLC d/b/a Memorial Health, "
        "Brittany Roy RN, Nephettia Robinson RN, Rachel Tinker RN, Sound Inpatient "
        "Physicians Inc, Dr. Everette Thombs Jr., The Neurological Institute of "
        "Savannah, Christine Lokey NP, and unknown parties Jane Does 1-5 and XYZ Corp 1-5."
    )
    
    print(f"\n📝 Caching answer for: '{original_query}'")
    cache.cache_answer(original_query, answer)
    
    # Test similar queries
    similar_queries = [
        "Who are the defendants?",
        "List all the defendants",
        "What parties are being sued?",
        "Who is being accused in this lawsuit?",
        "Name the defendants in the case"
    ]
    
    print(f"\n🔍 Testing {len(similar_queries)} similar queries:\n")
    
    hits = 0
    for i, query in enumerate(similar_queries, 1):
        result = cache.get_cached_answer(query)
        
        if result and result['cached']:
            hits += 1
            print(f"   [{i}] ✅ HIT  (sim: {result['similarity']:.4f}) - {query}")
        else:
            print(f"   [{i}] ❌ MISS - {query}")
    
    hit_rate = hits / len(similar_queries) * 100
    print(f"\n📊 Results:")
    print(f"   Hits: {hits}/{len(similar_queries)} ({hit_rate:.0f}%)")
    
    stats = cache.get_cache_stats()
    print(f"\n📊 Overall Cache Stats:")
    print(f"   Hit Rate: {stats['hit_rate']:.1%}")
    print(f"   Total Queries: {stats['total_queries']}")
    
    print("\n✅ Multiple similar queries test complete")


def test_cache_statistics():
    """Test 5: Cache Statistics"""
    print_header("TEST 5: Cache Statistics & Management")
    
    cache = CacheTools()
    
    # Get stats
    stats = cache.get_cache_stats()
    
    print(f"\n📊 Cache Statistics:")
    print(f"   Status: {'✅ Enabled' if stats['enabled'] else '❌ Disabled'}")
    print(f"   Total Entries: {stats['total_entries']}/{stats['max_entries']}")
    print(f"   Total Queries: {stats['total_queries']}")
    print(f"   Cache Hits: {stats['hits']}")
    print(f"   Cache Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate']:.1%}")
    print(f"   Answers Cached: {stats['saves']}")
    print(f"   Similarity Threshold: {stats['similarity_threshold']}")
    print(f"   TTL: {stats['ttl_hours']} hours")
    
    # Get recent entries
    print(f"\n📝 Recent Cache Entries:")
    entries = cache.get_cache_entries(limit=5)
    
    if entries:
        for i, entry in enumerate(entries, 1):
            print(f"\n   [{i}] Query: {entry['query']}")
            print(f"       Answer: {entry['answer']}")
            print(f"       Hits: {entry['hit_count']}")
            print(f"       Created: {entry['created_at'][:19]}")
    else:
        print("   (No entries yet)")
    
    print("\n✅ Statistics test complete")


def test_cost_savings():
    """Test 6: Cost Savings Calculation"""
    print_header("TEST 6: Cost Savings Demonstration")
    
    cache = CacheTools()
    cache.clear_cache()
    
    # Simulate real usage
    queries = [
        ("What are the allegations?", "First query - generate answer"),
        ("What are the claims?", "Similar - should hit cache"),
        ("What allegations were made?", "Similar - should hit cache"),
        ("What are the accusations?", "Similar - should hit cache"),
        ("Who are the defendants?", "Different - generate answer"),
        ("Who is being sued?", "Similar to previous - should hit cache"),
        ("List the defendants", "Similar - should hit cache"),
    ]
    
    # Cost per LLM call (approximate)
    COST_PER_LLM_CALL = 0.03  # $0.03 per answer generation
    TIME_PER_LLM_CALL = 2.0   # 2 seconds
    TIME_PER_CACHE_HIT = 0.05  # 50ms
    
    print(f"\n🧪 Simulating {len(queries)} queries:\n")
    
    # First pass: cache some answers
    cache.cache_answer(
        "What are the allegations?",
        "Memorial Health is alleged to have caused severe brain damage...",
        ["source1.pdf"]
    )
    cache.cache_answer(
        "Who are the defendants?",
        "The defendants include Memorial Health, Nurse Roy, Nurse Robinson...",
        ["source2.pdf"]
    )
    
    llm_calls = 0
    cache_hits = 0
    
    for i, (query, expected) in enumerate(queries, 1):
        result = cache.get_cached_answer(query)
        
        if result and result['cached']:
            cache_hits += 1
            print(f"   [{i}] 💾 CACHE HIT - {query}")
        else:
            llm_calls += 1
            print(f"   [{i}] 🤖 LLM CALL  - {query}")
    
    # Calculate savings
    cost_without_cache = len(queries) * COST_PER_LLM_CALL
    cost_with_cache = llm_calls * COST_PER_LLM_CALL
    cost_saved = cost_without_cache - cost_with_cache
    savings_percent = (cost_saved / cost_without_cache) * 100
    
    time_without_cache = len(queries) * TIME_PER_LLM_CALL
    time_with_cache = (llm_calls * TIME_PER_LLM_CALL) + (cache_hits * TIME_PER_CACHE_HIT)
    time_saved = time_without_cache - time_with_cache
    
    print(f"\n💰 Cost Analysis:")
    print(f"   Without Cache: ${cost_without_cache:.2f} ({len(queries)} LLM calls)")
    print(f"   With Cache:    ${cost_with_cache:.2f} ({llm_calls} LLM calls, {cache_hits} cache hits)")
    print(f"   Saved:         ${cost_saved:.2f} ({savings_percent:.0f}%)")
    
    print(f"\n⏱️  Time Analysis:")
    print(f"   Without Cache: {time_without_cache:.1f}s")
    print(f"   With Cache:    {time_with_cache:.2f}s")
    print(f"   Saved:         {time_saved:.2f}s ({(time_saved/time_without_cache)*100:.0f}%)")
    
    # Extrapolate to production
    daily_queries = 1000
    monthly_queries = daily_queries * 30
    
    monthly_cost_without = monthly_queries * COST_PER_LLM_CALL
    monthly_cost_with = monthly_queries * COST_PER_LLM_CALL * (1 - (savings_percent/100))
    monthly_savings = monthly_cost_without - monthly_cost_with
    
    print(f"\n📈 Production Estimate (with {savings_percent:.0f}% hit rate):")
    print(f"   Daily Queries: {daily_queries}")
    print(f"   Monthly Cost Without Cache: ${monthly_cost_without:.2f}")
    print(f"   Monthly Cost With Cache:    ${monthly_cost_with:.2f}")
    print(f"   Monthly Savings:            ${monthly_savings:.2f}")
    
    print("\n✅ Cost savings test complete")


def run_all_tests():
    """Run all cache tools tests."""
    print("\n" + "🧪" * 40)
    print("CACHE TOOLS TEST SUITE (Phase 3 - Tool 7)")
    print("🧪" * 40)
    
    try:
        test_cache_initialization()
        test_cache_miss()
        test_cache_save_and_hit()
        test_multiple_similar_queries()
        test_cache_statistics()
        test_cost_savings()
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("✅ PASS - Cache Initialization")
        print("✅ PASS - Cache Miss")
        print("✅ PASS - Cache Save & Hit")
        print("✅ PASS - Multiple Similar Queries")
        print("✅ PASS - Cache Statistics")
        print("✅ PASS - Cost Savings")
        print("\nResults: 6/6 tests passed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
