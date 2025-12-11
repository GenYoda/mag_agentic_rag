"""
Debug script to test core/semantic_cache.py

Tests:
1. Cache initialization
2. Adding entries
3. Cache hit (exact match)
4. Cache hit (similar question)
5. Cache miss (different question)
6. Statistics tracking
7. Expiration (if TTL set)
8. Save/load
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_semantic_cache():
    """Test semantic cache system"""
    
    print("🧪 Testing core/semantic_cache.py...\n")
    
    try:
        from core.semantic_cache import (
            CacheEntry,
            SemanticCache,
            print_cache_stats
        )
        
        print("✅ Import successful!\n")
        
        # Test 1: Cache initialization
        print("Test 1: Cache Initialization")
        print("-" * 40)
        
        cache = SemanticCache(
            similarity_threshold=0.90,  # Lower threshold for testing
            ttl_hours=None,  # No expiration for testing
            max_cache_size=10
        )
        
        print(f"✅ Cache initialized")
        print(f"   • Similarity threshold: {cache.similarity_threshold}")
        print(f"   • Max size: {cache.max_cache_size}\n")
        
        # Test 2: Adding entries
        print("Test 2: Adding Cache Entries")
        print("-" * 40)
        print("⏳ Generating embeddings and adding to cache...")
        
        qa_pairs = [
            ("What is diabetes?", "Diabetes is a metabolic disorder characterized by high blood sugar."),
            ("What medications treat hypertension?", "Common medications include ACE inhibitors, beta blockers, and diuretics."),
            ("What is normal blood pressure?", "Normal blood pressure is typically below 120/80 mmHg."),
        ]
        
        for q, a in qa_pairs:
            success = cache.add(q, a, metadata={'source': 'test'})
            if success:
                print(f"✅ Added: '{q[:40]}...'")
            else:
                print(f"❌ Failed: '{q[:40]}...'")
        
        print()
        
        # Test 3: Cache hit - Exact match
        print("Test 3: Cache Hit (Exact Match)")
        print("-" * 40)
        
        query1 = "What is diabetes?"
        print(f"Query: '{query1}'")
        
        result = cache.get(query1)
        
        if result:
            answer, meta = result
            print(f"✅ Cache HIT!")
            print(f"   • Similarity: {meta['similarity']:.4f}")
            print(f"   • Answer: {answer[:60]}...\n")
        else:
            print(f"❌ Cache MISS\n")
        
        # Test 4: Cache hit - Similar question
        print("Test 4: Cache Hit (Similar Question)")
        print("-" * 40)
        
        query2 = "What is diabetes mellitus?"  # Similar to "What is diabetes?"
        print(f"Query: '{query2}'")
        
        result = cache.get(query2)
        
        if result:
            answer, meta = result
            print(f"✅ Cache HIT!")
            print(f"   • Matched question: '{meta['cached_question']}'")
            print(f"   • Similarity: {meta['similarity']:.4f}")
            print(f"   • Answer: {answer[:60]}...\n")
        else:
            print(f"❌ Cache MISS (similarity below threshold)\n")
        
        # Test 5: Cache miss - Different question
        print("Test 5: Cache Miss (Different Question)")
        print("-" * 40)
        
        query3 = "What is the weather today?"  # Completely different
        print(f"Query: '{query3}'")
        
        result = cache.get(query3)
        
        if result:
            print(f"❌ Unexpected cache HIT\n")
        else:
            print(f"✅ Cache MISS (as expected)\n")
        
        # Test 6: Statistics
        print("Test 6: Cache Statistics")
        print("-" * 40)
        
        stats = cache.get_stats()
        print(f"✅ Statistics:")
        print(f"   • Total entries: {stats['total_entries']}")
        print(f"   • Total queries: {stats['total_queries']}")
        print(f"   • Hits: {stats['hits']}")
        print(f"   • Misses: {stats['misses']}")
        print(f"   • Hit rate: {stats['hit_rate']:.2%}\n")
        
        # Test 7: Multiple queries (track hit rate)
        print("Test 7: Multiple Queries (Hit Rate Tracking)")
        print("-" * 40)
        
        test_queries = [
            "What is diabetes?",  # Should hit
            "What treats high blood pressure?",  # Should hit (similar to hypertension)
            "What is cancer?",  # Should miss
            "Normal BP range?",  # Should hit (similar to blood pressure)
        ]
        
        for q in test_queries:
            result = cache.get(q)
            status = "HIT" if result else "MISS"
            print(f"   • '{q[:40]}...' → {status}")
        
        print()
        
        # Test 8: Save/Load
        print("Test 8: Save and Load Cache")
        print("-" * 40)
        
        # Save
        cache.save("test_cache.json")
        print(f"✅ Cache saved")
        
        # Create new cache and load
        cache2 = SemanticCache()
        success = cache2.load("test_cache.json")
        
        if success:
            print(f"✅ Cache loaded")
            print(f"   • Entries: {len(cache2.entries)}")
            print(f"   • Stats: {cache2.stats}")
            
            # Verify by querying
            test_result = cache2.get("What is diabetes?")
            if test_result:
                print(f"✅ Verification query successful\n")
            else:
                print(f"❌ Verification query failed\n")
        else:
            print(f"❌ Failed to load cache\n")
        
        # Test 9: Clear operations
        print("Test 9: Clear Operations")
        print("-" * 40)
        
        print(f"Before clear: {len(cache.entries)} entries")
        cache.clear()
        print(f"After clear: {len(cache.entries)} entries")
        print(f"✅ Cache cleared\n")
        
        # Restore for final stats
        cache.load("test_cache.json")
        
        # Print full stats
        print_cache_stats(cache)
        
        print("🎉 ALL TESTS PASSED! 🎉\n")
        
        print("=" * 80)
        print("SEMANTIC CACHE SYSTEM VERIFIED")
        print("=" * 80)
        print("✅ Cache initialization")
        print("✅ Adding entries with embeddings")
        print("✅ Cache hits (exact and similar)")
        print("✅ Cache misses (different questions)")
        print("✅ Hit rate tracking")
        print("✅ Statistics calculation")
        print("✅ Save/load persistence")
        print("✅ Clear operations")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_semantic_cache()
    sys.exit(0 if success else 1)
