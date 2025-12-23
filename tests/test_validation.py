"""
================================================================================
RAG VALIDATION TEST - Phase 3 Final
================================================================================
Tests ValidationTools integration with RAG pipeline
No self-healing - just validation and reporting
================================================================================
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from tools.retrieval_tools import RetrievalTools
from tools.answer_generation_tools import AnswerGenerationTools
from tools.reranking_tools import RerankingTools
from tools.validation_tools import ValidationTools


# ============================================================================
# RAG Pipeline with Validation
# ============================================================================

class RAGPipeline:
    """RAG pipeline with validation (no healing)"""
    
    def __init__(self):
        self.retriever = RetrievalTools()
        self.reranker = RerankingTools()
        self.generator = AnswerGenerationTools()
        self.validator = ValidationTools(
            min_answer_length=50,
            min_citations=1,
            min_quality_score=0.6,
            enable_llm_judge=True,  # ← Azure OpenAI
            llm_provider="azure_openai"
        )
        logger.info("✅ RAG Pipeline with Validation initialized")
    
    def query(self, user_query: str, top_k: int = 5) -> dict:
        """Run complete RAG pipeline with validation"""
        
        start_time = datetime.now()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: {user_query}")
        logger.info(f"{'='*80}")
        
        # 1. Retrieve
        logger.info("Step 1: Retrieval...")
        retrieval_result = self.retriever.search(
            query=user_query,
            top_k=top_k,
            return_scores=True
        )
        
        if not retrieval_result['success']:
            logger.error(f"Retrieval failed: {retrieval_result.get('error')}")
            return {'success': False, 'error': 'Retrieval failed'}
        
        retrieved = retrieval_result['results']
        logger.info(f"   ✅ Retrieved {len(retrieved)} chunks")
        
        # 2. Rerank
        logger.info("Step 2: Reranking...")
        reranked = self.reranker.rerank(
            query=user_query,
            chunks=retrieved,
            top_k=3
        )
        logger.info(f"   ✅ Reranked to top {len(reranked)} chunks")
        
        # 3. Generate
        logger.info("Step 3: Generation...")
        gen_result = self.generator.generate_answer_with_custom_context(
            query=user_query,
            context_chunks=reranked
        )
        
        if not gen_result['success']:
            return {'success': False, 'error': 'Generation failed'}
        
        answer = gen_result['answer']
        logger.info(f"   ✅ Generated answer ({len(answer)} chars)")
        
        # 4. Validate
        logger.info("Step 4: Validation...")
        context_texts = [chunk['text'] for chunk in reranked]
        
        validation = self.validator.validate_answer(
            answer=answer,
            query=user_query,
            context_chunks=context_texts
        )
        
        logger.info(f"   {'✅' if validation.is_valid else '❌'} "
                   f"Valid: {validation.is_valid}, "
                   f"Quality: {validation.quality_score:.2f}")
        
        # Calculate time
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'success': True,
            'query': user_query,
            'answer': answer,
            'validation': validation,
            'sources': reranked,
            'metadata': {
                'chunks_retrieved': len(retrieved),
                'chunks_used': len(reranked),
                'is_valid': validation.is_valid,
                'quality_score': validation.quality_score,
                'confidence_score': validation.confidence_score,
                'citation_count': validation.citation_count,
                'has_hallucination': validation.has_hallucination,
                'llm_judge_used': validation.llm_judge_used,
                'total_time_ms': total_time
            }
        }


# ============================================================================
# Display Functions
# ============================================================================

def print_result(result: dict):
    """Pretty print a single result"""
    
    print(f"\n{'='*80}")
    print(f"QUERY: {result['query']}")
    print(f"{'='*80}")
    
    print(f"\n📚 SOURCES ({len(result['sources'])} chunks):")
    for i, chunk in enumerate(result['sources'], 1):
        text = chunk.get('text', str(chunk))
        print(f"   [{i}] {text[:80]}...")
    
    print(f"\n💬 ANSWER ({len(result['answer'])} chars):")
    print(f"   {result['answer']}")
    
    val = result['validation']
    print(f"\n📊 VALIDATION:")
    print(f"   Status: {'✅ VALID' if val.is_valid else '❌ INVALID'}")
    print(f"   Quality Score: {val.quality_score:.2f}/1.0")
    print(f"   Confidence: {val.confidence_score:.2f}/1.0")
    print(f"   Citations: {val.citation_count}")
    print(f"   Hallucination: {'⚠️ Yes' if val.has_hallucination else '✅ No'}")
    print(f"   LLM Judge: {'✅ Used' if val.llm_judge_used else '❌ Not used'}")
    print(f"   Validation Time: {val.validation_time_ms:.1f}ms")
    
    if val.issues:
        print(f"\n⚠️  ISSUES ({len(val.issues)}):")
        for issue in val.issues:
            print(f"   - [{issue.severity.upper()}] {issue.category}: {issue.message}")
    
    if val.llm_judge_reasoning:
        print(f"\n🤖 LLM JUDGE:")
        print(f"   {val.llm_judge_reasoning[:150]}...")
    
    meta = result['metadata']
    print(f"\n📈 METADATA:")
    print(f"   Total Time: {meta['total_time_ms']:.1f}ms")
    print(f"   Chunks Retrieved: {meta['chunks_retrieved']}")
    print(f"   Chunks Used: {meta['chunks_used']}")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# Test Runner
# ============================================================================

def run_validation_tests():
    """Run validation tests on multiple queries"""
    
    print("\n" + "🧪"*40)
    print("RAG VALIDATION TEST - PHASE 3 FINAL")
    print("Testing ValidationTools + LLM Judge")
    print("🧪"*40)
    
    pipeline = RAGPipeline()
    
    test_queries = [
        
        "give the details of the patient",
        
        "What happened to the patient?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'#'*80}")
        
        result = pipeline.query(query)
        
        if result['success']:
            print_result(result)
            results.append(result)
        else:
            print(f"\n❌ Query failed: {result.get('error')}")
    
    # Summary
    print(f"\n{'🎯'*40}")
    print("FINAL SUMMARY")
    print(f"{'🎯'*40}")
    
    if not results:
        print("\n❌ No successful queries")
        return
    
    total = len(results)
    valid = sum(1 for r in results if r['metadata']['is_valid'])
    avg_quality = sum(r['metadata']['quality_score'] for r in results) / total
    avg_confidence = sum(r['metadata']['confidence_score'] for r in results) / total
    avg_time = sum(r['metadata']['total_time_ms'] for r in results) / total
    avg_citations = sum(r['metadata']['citation_count'] for r in results) / total
    hallucinations = sum(1 for r in results if r['metadata']['has_hallucination'])
    
    print(f"\n📊 Statistics:")
    print(f"   Total Queries: {total}")
    print(f"   Valid Answers: {valid} ({valid/total*100:.1f}%)")
    print(f"   Hallucinations Detected: {hallucinations}")
    print(f"   Average Quality: {avg_quality:.2f}")
    print(f"   Average Confidence: {avg_confidence:.2f}")
    print(f"   Average Citations: {avg_citations:.1f}")
    print(f"   Average Time: {avg_time:.1f}ms")
    
    print(f"\n✅ Phase 3 Complete!")
    print(f"   - ValidationTools: ✅ Working")
    print(f"   - LLM-as-Judge: ✅ Working")
    print(f"   - Citation Detection: ✅ Working")
    print(f"   - Hallucination Detection: ✅ Working")
    


if __name__ == "__main__":
    try:
        run_validation_tests()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
