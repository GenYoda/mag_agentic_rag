"""
================================================================================
DEBUG SCRIPT: Reranking Agent Test
================================================================================
Tests the Reranking Agent in isolation with mock chunks

Test Scenarios:
1. Cross-encoder reranking (default fast method)
2. LLM-based reranking (high-quality fallback)
3. Smart fallback strategy (automatic selection)
4. Compare before/after rankings
5. Edge case: all irrelevant chunks

Prerequisites:
- Reranking tools must be implemented
- Cross-encoder model installed (sentence-transformers)
- Azure OpenAI configured for LLM reranking
- Mock chunks prepared for testing

Run:
    python tests/debug/debug_reranking_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.reranking_agent import create_reranking_agent


def test_reranking_agent():
    """Test reranking agent with various scenarios."""
    
    print("\n" + "="*80)
    print("RERANKING AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Reranking Agent
    # ========================================================================
    print("📝 Step 1: Creating Reranking Agent...")
    try:
        reranking_agent = create_reranking_agent(verbose=True)
        print(f"✅ Agent created successfully")
        print(f"   Role: {reranking_agent.role}")
        print(f"   Tools: {len(reranking_agent.tools)} tools loaded")
        print(f"   Max Iterations: {reranking_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Cross-Encoder Reranking (Default Fast Method)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Cross-Encoder Reranking")
    print("-"*80)
    print("Query: 'patient hypertension treatment'")
    print("Chunks: 5 medical text chunks (varying relevance)")
    print("Expected: Reranked by cross-encoder scores\n")
    
    task1 = Task(
        description=(
            "Rerank these chunks for query: 'patient hypertension treatment'\n\n"
            "Initial chunks (from retrieval, may be poorly ranked):\n\n"
            "1. 'Patient was prescribed medication for blood pressure management.'\n"
            "2. 'The patient history includes diabetes and high cholesterol.'\n"
            "3. 'Hypertension was managed with Lisinopril 10mg daily.'\n"
            "4. 'Patient reported no allergies to medications.'\n"
            "5. 'Blood pressure readings: 140/90, 135/88, 130/85 over 3 weeks.'\n\n"
            "Use rerank_crossencoder_tool to rerank these chunks. Return top 3."
        ),
        agent=reranking_agent,
        expected_output=(
            "Reranked chunks (top 3) with cross-encoder scores:\n"
            "1. Chunk 3: 'Hypertension was managed...' - Score: ~0.85+\n"
            "2. Chunk 1: 'prescribed medication...' - Score: ~0.70+\n"
            "3. Chunk 5: 'Blood pressure readings...' - Score: ~0.60+\n"
            "(Chunks 2 and 4 should rank lower as less relevant)"
        )
    )
    
    crew1 = Crew(
        agents=[reranking_agent],
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
    # Test 2: LLM-Based Reranking (High-Quality Method)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: LLM-Based Reranking")
    print("-"*80)
    print("Query: Complex medical query requiring reasoning")
    print("Expected: LLM judges relevance with reasoning\n")
    
    task2 = Task(
        description=(
            "Rerank these chunks using LLM-based scoring for query:\n\n"
            "'What medications were prescribed for the patient and what were the dosages?'\n\n"
            "Chunks:\n"
            "1. 'Dr. Smith prescribed Lisinopril 10mg once daily for hypertension.'\n"
            "2. 'Patient takes medication regularly with meals.'\n"
            "3. 'Metformin 500mg twice daily was prescribed for diabetes management.'\n"
            "4. 'Patient scheduled follow-up appointment in 3 months.'\n"
            "5. 'No adverse reactions reported to current medications.'\n\n"
            "Use rerank_llm_tool which asks GPT-4 to judge relevance."
        ),
        agent=reranking_agent,
        expected_output=(
            "LLM reranked results:\n"
            "1. Chunk 1: Lisinopril 10mg (directly answers medication + dosage)\n"
            "2. Chunk 3: Metformin 500mg (directly answers medication + dosage)\n"
            "3. Chunk 5: No adverse reactions (somewhat relevant)\n"
            "4-5. Other chunks (less relevant to specific question)\n"
            "Each with LLM reasoning explanation"
        )
    )
    
    crew2 = Crew(
        agents=[reranking_agent],
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
    # Test 3: Smart Fallback Strategy
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Smart Fallback Strategy")
    print("-"*80)
    print("Automatically choose cross-encoder or LLM based on quality")
    print("Expected: Start with cross-encoder, fallback if needed\n")
    
    task3 = Task(
        description=(
            "Rerank using smart fallback strategy for query:\n\n"
            "'What are the contraindications for the prescribed medication?'\n\n"
            "Chunks (potentially low-quality matches):\n"
            "1. 'Patient has no known drug allergies.'\n"
            "2. 'Medication should be taken with food.'\n"
            "3. 'Contraindications include renal impairment and pregnancy.'\n"
            "4. 'Patient education materials provided.'\n"
            "5. 'Side effects may include dizziness and fatigue.'\n\n"
            "Use rerank_smart_fallback_tool:\n"
            "- Tries cross-encoder first\n"
            "- If max score < 0.5, falls back to LLM\n"
            "- Returns reranked results + method used"
        ),
        agent=reranking_agent,
        expected_output=(
            "Smart fallback result:\n"
            "- Method used: cross-encoder or llm (with explanation)\n"
            "- Top chunk: Chunk 3 (directly mentions contraindications)\n"
            "- Scores and reasoning\n"
            "- Fallback triggered: Yes/No"
        )
    )
    
    crew3 = Crew(
        agents=[reranking_agent],
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
    # Test 4: Compare Before/After Rankings
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Compare Before/After Rankings")
    print("-"*80)
    print("Show improvement from reranking\n")
    
    task4 = Task(
        description=(
            "Demonstrate reranking improvement for query: 'patient current medications'\n\n"
            "Initial ranking (by embedding similarity - may have errors):\n"
            "1. 'Patient discontinued previous medications.' (Score: 0.88)\n"
            "2. 'Current medication list: Lisinopril 10mg, Metformin 500mg.' (Score: 0.85)\n"
            "3. 'Patient adheres to medication schedule.' (Score: 0.82)\n"
            "4. 'Medication refills available at pharmacy.' (Score: 0.80)\n\n"
            "Notice Chunk 1 ranks highest but says 'discontinued' not 'current'!\n\n"
            "Rerank using cross-encoder to fix this ordering."
        ),
        agent=reranking_agent,
        expected_output=(
            "Comparison showing:\n"
            "BEFORE: Chunk 1 (discontinued meds) ranked #1 - WRONG!\n"
            "AFTER: Chunk 2 (current med list) ranked #1 - CORRECT!\n"
            "Demonstrates reranking catches semantic nuances"
        )
    )
    
    crew4 = Crew(
        agents=[reranking_agent],
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
    # Test 5: Edge Case - All Irrelevant Chunks
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Edge Case - All Irrelevant Chunks")
    print("-"*80)
    print("Test behavior when no chunks are relevant\n")
    
    task5 = Task(
        description=(
            "Rerank for query: 'patient surgical history'\n\n"
            "Chunks (all about medications, not surgery):\n"
            "1. 'Patient takes Lisinopril for blood pressure.'\n"
            "2. 'Metformin prescribed for diabetes management.'\n"
            "3. 'No known drug allergies reported.'\n"
            "4. 'Medication adherence is excellent.'\n"
            "5. 'Patient uses weekly pill organizer.'\n\n"
            "None of these mention surgery! Rerank should give all low scores."
        ),
        agent=reranking_agent,
        expected_output=(
            "Reranking result showing:\n"
            "- All chunks with low relevance scores (< 0.3)\n"
            "- Warning that no chunks are relevant to query\n"
            "- Recommendation to expand search or rephrase query\n"
            "This helps detect when retrieval failed to find relevant context"
        )
    )
    
    crew5 = Crew(
        agents=[reranking_agent],
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
    print("RERANKING AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Cross-encoder reranking (fast, accurate)")
    print("  ✓ LLM-based reranking (high-quality, slower)")
    print("  ✓ Smart fallback strategy (automatic)")
    print("  ✓ Before/after comparison (improvement demo)")
    print("  ✓ Edge case handling (all irrelevant)")
    print("\n💡 Next Steps:")
    print("  1. Install cross-encoder model: pip install sentence-transformers")
    print("  2. Tune fallback threshold in config/settings.py")
    print("  3. Test with real retrieval results")
    print("  4. Measure speed: cross-encoder vs LLM")
    print("  5. Integrate with Answer Generation Agent")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_reranking_agent()
