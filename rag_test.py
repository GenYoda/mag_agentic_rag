"""
test_rag_pipeline_with_reranking.py

End-to-end test showing:
1. Query → Retrieval → Answer (without reranking)
2. Query → Retrieval → Reranking → Answer (with reranking)
3. Side-by-side quality comparison
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.retrieval_tools import RetrievalTools
from tools.reranking_tools import RerankingTools
from tools.answer_generation_tools import AnswerGenerationTools
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_rag_with_and_without_reranking():
    """Compare RAG quality with and without reranking."""
    
    print("\n" + "="*80)
    print("END-TO-END RAG PIPELINE TEST: WITH vs WITHOUT RERANKING")
    print("="*80)
    
    # Test queries
    queries = [
        "What are the allegations against Memorial Health?",
        "What medical condition did Frank Ward suffer?",
        "Who are the defendants in this case?"
    ]
    
    retrieval = RetrievalTools()
    reranker = RerankingTools(method="cross_encoder")
    answer_gen = AnswerGenerationTools()
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve chunks
        print(f"\n📊 Step 1: Retrieval (top 10 chunks)")
        search_result = retrieval.search(query=query, top_k=10)
        
        if not search_result['success']:
            print(f"❌ Retrieval failed")
            continue
        
        chunks = search_result['results']
        print(f"   ✅ Retrieved {len(chunks)} chunks")
        print(f"   📈 Top similarity: {chunks[0]['similarity']:.4f}")
        print(f"   📉 Bottom similarity: {chunks[-1]['similarity']:.4f}")
        
        # Step 2A: Answer WITHOUT reranking (using top 5 by similarity)
        print(f"\n🤖 Step 2A: Generate Answer WITHOUT Reranking")
        
        # Use generate_answer_with_custom_context (from your AnswerGenerationTools)
        answer_without = answer_gen.generate_answer_with_custom_context(
            query=query,
            context_chunks=chunks[:5]  # Top 5 by similarity
        )
        
        if answer_without['success']:
            print(f"   Answer (no rerank):")
            print(f"   {answer_without['answer'][:250]}...")
        else:
            print(f"   ❌ Failed: {answer_without.get('error', 'Unknown error')}")
        
        # Step 2B: Rerank chunks
        print(f"\n🔄 Step 2B: Rerank Chunks")
        reranked = reranker.rerank(
            query=query,
            chunks=chunks.copy(),
            top_k=5
        )
        
        print(f"   ✅ Reranked to top 5")
        if reranked:
            print(f"   📈 Top rerank score: {reranked[0]['rerank_score']:.4f}")
            print(f"   📉 Bottom rerank score: {reranked[-1]['rerank_score']:.4f}")
        
        # Step 2C: Answer WITH reranking
        print(f"\n🤖 Step 2C: Generate Answer WITH Reranking")
        answer_with = answer_gen.generate_answer_with_custom_context(
            query=query,
            context_chunks=reranked
        )
        
        if answer_with['success']:
            print(f"   Answer (reranked):")
            print(f"   {answer_with['answer'][:250]}...")
        else:
            print(f"   ❌ Failed: {answer_with.get('error', 'Unknown error')}")
        
        # Step 3: Compare
        print(f"\n📊 COMPARISON:")
        print(f"   {'─'*76}")
        print(f"   WITHOUT Reranking:")
        print(f"   {answer_without.get('answer', 'N/A')}")
        print(f"   {'─'*76}")
        print(f"   WITH Reranking:")
        print(f"   {answer_with.get('answer', 'N/A')}")
        print(f"   {'─'*76}")
        
        # Show which chunks were used
        print(f"\n📄 Top 3 Chunks Used:")
        print(f"\n   WITHOUT Reranking (by similarity):")
        for j, chunk in enumerate(chunks[:3], 1):
            print(f"      [{j}] Sim: {chunk['similarity']:.4f} | {chunk['metadata']['source']} (Page {chunk['metadata'].get('page_numbers', ['?'])[0]})")
            print(f"          {chunk['text'][:100]}...")
        
        print(f"\n   WITH Reranking (by relevance):")
        for j, chunk in enumerate(reranked[:3], 1):
            print(f"      [{j}] Score: {chunk['rerank_score']:.4f} | {chunk['metadata']['source']} (Page {chunk['metadata'].get('page_numbers', ['?'])[0]})")
            print(f"          {chunk['text'][:100]}...")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_rag_with_and_without_reranking()
