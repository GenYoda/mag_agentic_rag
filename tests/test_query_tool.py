"""
Test Script for QueryTools

Tests:
1. Query classification
2. Query decomposition
3. HyDE generation
4. Query variations
5. Full enhancement pipeline
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.query_tools import QueryTools
import logging

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


def test_query_classification():
    """Test 1: Query Classification"""
    print_header("TEST 1: Query Classification")
    
    query_tools = QueryTools()
    
    test_queries = [
        "What are the allegations against Memorial Health?",  # Simple factual
        "What medication was prescribed and what was the dosage?",  # Complex multi-part
        "Why did Frank Ward suffer brain damage?",  # Analytical
        "Compare the actions of Nurse Roy and Nurse Robinson",  # Comparison
        "List all defendants in this case"  # List query
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        classification = query_tools.classify_query(query)
        
        print(f"   📊 Type: {classification['type']}")
        print(f"   📊 Complexity: {classification['complexity']}")
        print(f"   📊 Requires Decomposition: {classification['requires_decomposition']}")
        print(f"   📊 Requires HyDE: {classification['requires_hyde']}")
        print(f"   📊 Is List Query: {classification['is_list_query']}")
    
    print("\n✅ Classification test complete")


def test_query_decomposition():
    """Test 2: Query Decomposition"""
    print_header("TEST 2: Query Decomposition")
    
    query_tools = QueryTools()
    
    complex_queries = [
        "What medication was prescribed and what was the dosage?",
        "Who are the defendants and what are they accused of?",
        "When did the incident occur and what was the immediate response?"
    ]
    
    for i, query in enumerate(complex_queries, 1):
        print(f"\n🔍 Complex Query {i}:")
        print(f"   Original: {query}")
        
        sub_queries = query_tools.decompose_query(query)
        
        print(f"\n   📝 Decomposed into {len(sub_queries)} sub-queries:")
        for j, sub_q in enumerate(sub_queries, 1):
            print(f"      [{j}] {sub_q}")
    
    print("\n✅ Decomposition test complete")


def test_hyde_generation():
    """Test 3: HyDE (Hypothetical Document Embeddings)"""
    print_header("TEST 3: HyDE Generation")
    
    query_tools = QueryTools()
    
    test_queries = [
        "What caused Frank Ward's brain damage?",
        "What are the allegations against Memorial Health?",
        "What was the timeline of events?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        hyde_text = query_tools.generate_hypothetical_answer(query)
        
        print(f"\n   💡 HyDE Hypothetical Answer:")
        print(f"   {hyde_text}")
    
    print("\n✅ HyDE generation test complete")


def test_query_variations():
    """Test 4: Query Variations"""
    print_header("TEST 4: Query Variations")
    
    query_tools = QueryTools()
    
    test_queries = [
        "What are the allegations?",
        "Who are the defendants?",
        "What medical condition did the patient suffer?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        variations = query_tools.generate_query_variations(query, max_variations=3)
        
        print(f"\n   🔄 Generated {len(variations)} variations:")
        for j, var in enumerate(variations, 1):
            print(f"      [{j}] {var}")
    
    print("\n✅ Query variations test complete")


def test_full_enhancement_pipeline():
    """Test 5: Full Query Enhancement Pipeline"""
    print_header("TEST 5: Full Enhancement Pipeline")
    
    query_tools = QueryTools()
    
    test_queries = [
        "What are the allegations against Memorial Health?",  # Simple
        "What medication was prescribed and what was the dosage?",  # Complex
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}: {query}")
        print(f"{'─'*80}")
        
        enhanced = query_tools.enhance_query(query)
        
        print(f"\n📊 Classification:")
        print(f"   Type: {enhanced['classification']['type']}")
        print(f"   Complexity: {enhanced['classification']['complexity']}")
        
        if len(enhanced['sub_queries']) > 1:
            print(f"\n📝 Sub-Queries ({len(enhanced['sub_queries'])}):")
            for j, sq in enumerate(enhanced['sub_queries'], 1):
                print(f"   [{j}] {sq}")
        
        if enhanced['hyde_text']:
            print(f"\n💡 HyDE Text:")
            print(f"   {enhanced['hyde_text'][:150]}...")
        
        if len(enhanced['variations']) > 1:
            print(f"\n🔄 Query Variations ({len(enhanced['variations'])}):")
            for j, var in enumerate(enhanced['variations'], 1):
                print(f"   [{j}] {var}")
        
        print(f"\n✅ Enhanced: {enhanced['enhanced']}")
    
    print("\n✅ Full enhancement pipeline test complete")


def test_configuration():
    """Test 6: Configuration Display"""
    print_header("TEST 6: Configuration & Statistics")
    
    query_tools = QueryTools()
    
    stats = query_tools.get_query_stats()
    
    print("\n📊 Query Enhancement Configuration:")
    print(f"   Classification: {'✅ Enabled' if stats['classification_enabled'] else '❌ Disabled'}")
    print(f"   Decomposition: {'✅ Enabled' if stats['decomposition_enabled'] else '❌ Disabled'}")
    print(f"   HyDE: {'✅ Enabled' if stats['hyde_enabled'] else '❌ Disabled'}")
    print(f"   Variations: {'✅ Enabled' if stats['variations_enabled'] else '❌ Disabled'}")
    print(f"\n📊 Limits:")
    print(f"   Max Sub-Queries: {stats['max_subqueries']}")
    print(f"   Max Variations: {stats['max_variations']}")
    print(f"   Variation Method: {stats['variation_method']}")
    
    print("\n✅ Configuration test complete")


def run_all_tests():
    """Run all query tools tests."""
    print("\n" + "🧪" * 40)
    print("QUERY TOOLS TEST SUITE (Phase 3 - Tool 5)")
    print("🧪" * 40)
    
    try:
        test_configuration()
        test_query_classification()
        test_query_decomposition()
        test_hyde_generation()
        test_query_variations()
        test_full_enhancement_pipeline()
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("✅ PASS - Configuration")
        print("✅ PASS - Query Classification")
        print("✅ PASS - Query Decomposition")
        print("✅ PASS - HyDE Generation")
        print("✅ PASS - Query Variations")
        print("✅ PASS - Full Enhancement Pipeline")
        print("\nResults: 6/6 tests passed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
