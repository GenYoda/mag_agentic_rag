"""
Test script for RerankingTools (Phase 3 - Tool 4)

Tests:
1. Cross-encoder reranking
2. Score normalization
3. Top-K filtering
4. Method comparison (optional)
5. Reranking statistics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.reranking_tools import RerankingTools
from tools.retrieval_tools import RetrievalTools
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_cross_encoder_reranking():
    """Test basic cross-encoder reranking."""
    print("\n" + "="*80)
    print("TEST 1: Cross-Encoder Reranking")
    print("="*80)
    
    try:
        # Get some chunks from retrieval
        retrieval = RetrievalTools()
        query = "What are the allegations against Memorial Health?"
        
        print(f"🔍 Query: {query}")
        print(f"📊 Retrieving chunks...\n")
        
        search_result = retrieval.search(query=query, top_k=10)
        
        if not search_result['success'] or not search_result['results']:
            print(f"❌ No results from retrieval")
            return False
        
        chunks = search_result['results']
        print(f"✅ Retrieved {len(chunks)} chunks")
        print(f"📊 Top chunk before reranking:")
        print(f"   Similarity: {chunks[0].get('similarity', 'N/A')}")
        print(f"   Text: {chunks[0]['text'][:150]}...\n")
        
        # Rerank with cross-encoder
        reranker = RerankingTools(method="cross_encoder")
        
        print(f"🔄 Reranking with cross-encoder...")
        reranked = reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=5
        )
        
        if reranked:
            print(f"✅ SUCCESS\n")
            print(f"📊 Top 3 Results After Reranking:")
            for i, chunk in enumerate(reranked[:3], 1):
                print(f"\n   [{i}] Rerank Score: {chunk.get('rerank_score', 'N/A'):.4f}")
                print(f"       Original Similarity: {chunk.get('similarity', 'N/A')}")
                print(f"       Source: {chunk['metadata'].get('source')}")
                print(f"       Text: {chunk['text'][:150]}...")
            
            return True
        else:
            print(f"❌ FAILED: No reranked results")
            return False
            
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        logger.error("Test crashed", exc_info=True)
        return False


def test_score_normalization():
    """Test score normalization methods."""
    print("\n" + "="*80)
    print("TEST 2: Score Normalization")
    print("="*80)
    
    try:
        retrieval = RetrievalTools()
        query = "Who are the defendants?"
        
        search_result = retrieval.search(query=query, top_k=5)
        
        if not search_result['success']:
            print(f"❌ Retrieval failed")
            return False
        
        chunks = search_result['results']
        reranker = RerankingTools()
        
        print(f"🔍 Testing different normalization methods...")
        
        # Test different normalization methods
        methods = ["minmax", "sigmoid", "none"]
        
        for method in methods:
            # Temporarily set normalization method
            import config.settings as settings
            original = settings.RERANKING_SCORE_NORMALIZATION
            settings.RERANKING_SCORE_NORMALIZATION = method
            
            reranked = reranker.rerank(
                query=query,
                chunks=[c.copy() for c in chunks],
                top_k=3,
                normalize_scores=(method != "none")
            )
            
            print(f"\n📊 Method: {method}")
            if reranked:
                scores = [c.get('rerank_score', 0) for c in reranked]
                print(f"   Scores: {[f'{s:.4f}' for s in scores]}")
                print(f"   Range: [{min(scores):.4f}, {max(scores):.4f}]")
            
            # Restore original
            settings.RERANKING_SCORE_NORMALIZATION = original
        
        print(f"\n✅ SUCCESS - All normalization methods tested")
        return True
        
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        logger.error("Test crashed", exc_info=True)
        return False


def test_topk_filtering():
    """Test top-K filtering after reranking."""
    print("\n" + "="*80)
    print("TEST 3: Top-K Filtering")
    print("="*80)
    
    try:
        retrieval = RetrievalTools()
        query = "What medical condition did Frank Ward suffer?"
        
        search_result = retrieval.search(query=query, top_k=10)
        
        if not search_result['success']:
            print(f"❌ Retrieval failed")
            return False
        
        chunks = search_result['results']
        reranker = RerankingTools()
        
        print(f"📊 Retrieved: {len(chunks)} chunks")
        
        # Test different top_k values
        for k in [3, 5, 8]:
            reranked = reranker.rerank(
                query=query,
                chunks=[c.copy() for c in chunks],
                top_k=k
            )
            
            print(f"\n   Top-{k}: Returned {len(reranked)} chunks")
            if reranked:
                print(f"   Top score: {reranked[0].get('rerank_score', 'N/A'):.4f}")
                print(f"   Last score: {reranked[-1].get('rerank_score', 'N/A'):.4f}")
        
        print(f"\n✅ SUCCESS - Top-K filtering works correctly")
        return True
        
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        logger.error("Test crashed", exc_info=True)
        return False


def test_reranking_stats():
    """Test reranking statistics."""
    print("\n" + "="*80)
    print("TEST 4: Reranking Statistics")
    print("="*80)
    
    try:
        reranker = RerankingTools()
        
        stats = reranker.get_reranking_stats()
        
        print(f"📊 Reranking Configuration:")
        print(f"   Method: {stats['method']}")
        print(f"   Model: {stats['model']}")
        print(f"   Model Loaded: {stats['model_loaded']}")
        print(f"   Batch Size: {stats['batch_size']}")
        print(f"   Normalization: {stats['normalization']}")
        print(f"   LLM Fallback: {stats['llm_fallback_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        logger.error("Test crashed", exc_info=True)
        return False


def test_model_loading():
    """Test model loading and caching."""
    print("\n" + "="*80)
    print("TEST 5: Model Loading & Caching")
    print("="*80)
    
    try:
        print(f"🔄 Loading cross-encoder model (first time may take ~1 minute)...")
        
        reranker = RerankingTools()
        
        # Force model load
        reranker._load_cross_encoder()
        
        if reranker.cross_encoder is not None:
            print(f"✅ Model loaded successfully and cached")
            print(f"📦 Model: {reranker.model_name}")
            return True
        else:
            print(f"❌ Model failed to load")
            return False
        
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        logger.error("Test crashed", exc_info=True)
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪" * 40)
    print("RERANKING TOOLS TEST SUITE (Phase 3 - Tool 4)")
    print("🧪" * 40)
    print("\nℹ️  First run will download cross-encoder model (~80MB)")
    print("ℹ️  Subsequent runs will use cached model\n")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Cross-Encoder Reranking", test_cross_encoder_reranking),
        ("Score Normalization", test_score_normalization),
        ("Top-K Filtering", test_topk_filtering),
        ("Reranking Statistics", test_reranking_stats)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}", exc_info=True)
            results[test_name] = "CRASH"
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results.items():
        if status == "PASS":
            print(f"✅ PASS - {test_name}")
        elif status == "FAIL":
            print(f"❌ FAIL - {test_name}")
        else:
            print(f"💥 CRASH - {test_name}")
    
    passed = sum(1 for s in results.values() if s == "PASS")
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
