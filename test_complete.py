"""
Test script for complete RAG pipeline
Run: python tests/test_complete_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from orchestration.complete_rag_pipeline import run_rag_query, create_rag_session

def main():
    print("\n" + "="*80)
    print("TESTING COMPLETE RAG PIPELINE")
    print("="*80 + "\n")
    
    # Test 1: Simple query
    print("Test 1: Simple Query")
    result = run_rag_query(
        query="who is frank ward ? ",
        verbose=True
    )
    
    print(f"\nAnswer: {result.get('answer', 'N/A')}")
    print(f"Validation Score: {result.get('validation', {}).get('quality_score', 'N/A')}")
    print(f"Healing Applied: {result.get('healing_applied', False)}")
    
    print("\n✅ Test Complete!\n")

if __name__ == "__main__":
    main()
