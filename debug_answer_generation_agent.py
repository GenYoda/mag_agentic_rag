"""
================================================================================
DEBUG SCRIPT: Answer Generation Agent Test
================================================================================
Tests the Answer Generation Agent in isolation

Test Scenarios:
1. Generate answer from query (full RAG pipeline)
2. Generate answer from pre-retrieved context
3. Test citation formatting
4. Handle missing information gracefully
5. Multi-turn conversation with context

Prerequisites:
- Answer generation tools must be implemented
- Azure OpenAI configured and working
- Knowledge base indexed (for full RAG test)
- Context formatting utilities working

Run:
    python tests/debug/debug_answer_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.answer_agent import create_answer_agent  # ✅ Corrected import


def test_answer_agent():
    """Test answer generation with various scenarios."""
    
    print("\n" + "="*80)
    print("ANSWER GENERATION AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Answer Generation Agent
    # ========================================================================
    print("📝 Step 1: Creating Answer Generation Agent...")
    try:
        answer_agent = create_answer_agent(verbose=True)  # ✅ Corrected function name
        print(f"✅ Agent created successfully")
        print(f"   Role: {answer_agent.role}")
        print(f"   Tools: {len(answer_agent.tools)} tools loaded")
        print(f"   Max Iterations: {answer_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Generate Answer from Query (Full RAG Pipeline)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Generate Answer from Query (Full RAG)")
    print("-"*80)
    print("Query: 'What treatment was provided for hypertension?'")
    print("Expected: Answer with citations from knowledge base\n")
    
    task1 = Task(
        description=(
            "Generate an answer for this query using the full RAG pipeline:\n\n"
            "'What treatment was provided for the patient's hypertension?'\n\n"
            "Use generate_answer_tool which will:\n"
            "1. Retrieve relevant context from the knowledge base\n"
            "2. Format context for the LLM\n"
            "3. Generate a well-cited answer\n"
            "4. Return answer with source citations\n\n"
            "The answer MUST include [doc:X] citations for all factual claims."
        ),
        agent=answer_agent,
        expected_output=(
            "Medical answer with proper structure:\n"
            "- Direct answer to the question\n"
            "- Specific treatment details (medications, dosages)\n"
            "- Citations in [doc:X] format\n"
            "- Source documents listed\n"
            "Example: 'The patient was treated for hypertension with Lisinopril 10mg "
            "daily [doc:1]. Lifestyle modifications including low-sodium diet were "
            "recommended [doc:1].'"
        )
    )
    
    crew1 = Crew(
        agents=[answer_agent],
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
    # Test 2: Generate Answer from Pre-Retrieved Context
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Generate Answer from Pre-Retrieved Context")
    print("-"*80)
    print("Query: 'What medications were prescribed?'")
    print("Context: Pre-provided medical record chunks\n")
    
    task2 = Task(
        description=(
            "Generate an answer using pre-retrieved context:\n\n"
            "Query: 'What medications were prescribed to the patient?'\n\n"
            "Context chunks (pre-retrieved and reranked):\n"
            "[doc:1] 'Dr. Sarah Johnson prescribed Lisinopril 10mg once daily for "
            "hypertension management on March 15, 2024.'\n"
            "[doc:2] 'Metformin 500mg twice daily was prescribed for type 2 diabetes "
            "control, to be taken with meals.'\n"
            "[doc:3] 'Patient was instructed to take both medications consistently "
            "and report any side effects.'\n\n"
            "Use generate_answer_with_context_tool to synthesize an answer from these chunks."
        ),
        agent=answer_agent,
        expected_output=(
            "Synthesized answer citing both medications:\n"
            "'The patient was prescribed two medications: Lisinopril 10mg once daily "
            "for hypertension [doc:1] and Metformin 500mg twice daily for diabetes "
            "management [doc:2]. Both medications should be taken consistently [doc:3].'"
        )
    )
    
    crew2 = Crew(
        agents=[answer_agent],
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
    # Test 3: Test Citation Formatting
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Verify Proper Citation Formatting")
    print("-"*80)
    print("Ensure all claims have [doc:X] citations\n")
    
    task3 = Task(
        description=(
            "Generate an answer and verify citation quality:\n\n"
            "Query: 'When was the patient diagnosed and by whom?'\n\n"
            "Context:\n"
            "[doc:1] 'Patient John Smith was diagnosed with essential hypertension "
            "on January 10, 2024.'\n"
            "[doc:2] 'The diagnosis was made by Dr. Michael Williams, MD, Cardiology, "
            "at Memorial Medical Center.'\n"
            "[doc:3] 'Follow-up appointments were scheduled every 3 months to monitor "
            "blood pressure control.'\n\n"
            "Generate answer ensuring:\n"
            "- Each fact has a citation\n"
            "- Citations are in [doc:X] format\n"
            "- No claims without citations"
        ),
        agent=answer_agent,
        expected_output=(
            "Well-cited answer:\n"
            "'John Smith was diagnosed with essential hypertension on January 10, 2024 [doc:1]. "
            "The diagnosis was made by Dr. Michael Williams, MD, a cardiologist at Memorial "
            "Medical Center [doc:2]. Follow-up appointments were scheduled every 3 months [doc:3].'\n\n"
            "Every sentence should have at least one citation."
        )
    )
    
    crew3 = Crew(
        agents=[answer_agent],
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
    # Test 4: Handle Missing Information Gracefully
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Handle Missing Information")
    print("-"*80)
    print("Test behavior when context doesn't contain the answer\n")
    
    task4 = Task(
        description=(
            "Generate answer when information is not available:\n\n"
            "Query: 'What surgical procedures did the patient undergo?'\n\n"
            "Context (only mentions medications, no surgery):\n"
            "[doc:1] 'Patient was prescribed Lisinopril 10mg for hypertension.'\n"
            "[doc:2] 'Metformin 500mg was prescribed for diabetes management.'\n"
            "[doc:3] 'Patient reports good medication adherence.'\n\n"
            "The context contains NO information about surgery. Generate an appropriate "
            "response that acknowledges this without making assumptions."
        ),
        agent=answer_agent,
        expected_output=(
            "Honest response acknowledging missing information:\n"
            "'Based on the available documents, there is no information about surgical "
            "procedures. The documents only contain information about prescribed medications "
            "[doc:1][doc:2] and medication adherence [doc:3].'\n\n"
            "Should NOT make up surgery information or use general medical knowledge."
        )
    )
    
    crew4 = Crew(
        agents=[answer_agent],
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
    # Test 5: Multi-Turn Conversation with Context
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Multi-Turn Conversation")
    print("-"*80)
    print("Generate answer using conversation history\n")
    
    task5 = Task(
        description=(
            "Generate answer for a follow-up question using conversation history:\n\n"
            "Conversation History:\n"
            "Turn 1:\n"
            "  Q: 'Who is the treating physician?'\n"
            "  A: 'Dr. Sarah Johnson is the treating physician [doc:1].'\n\n"
            "Turn 2 (current):\n"
            "  Q: 'What did she prescribe?'\n\n"
            "Context:\n"
            "[doc:1] 'Dr. Sarah Johnson, MD, is the primary care physician.'\n"
            "[doc:2] 'Dr. Johnson prescribed Lisinopril 10mg once daily.'\n"
            "[doc:3] 'Additional prescription: Metformin 500mg twice daily.'\n\n"
            "Generate answer that:\n"
            "- Understands 'she' refers to Dr. Sarah Johnson\n"
            "- Lists prescribed medications\n"
            "- Includes proper citations"
        ),
        agent=answer_agent,
        expected_output=(
            "Context-aware answer:\n"
            "'Dr. Sarah Johnson prescribed Lisinopril 10mg once daily [doc:2] and "
            "Metformin 500mg twice daily [doc:3] to the patient.'\n\n"
            "Should correctly resolve 'she' → Dr. Sarah Johnson from conversation history."
        )
    )
    
    crew5 = Crew(
        agents=[answer_agent],
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
    print("ANSWER GENERATION AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Full RAG pipeline (query → answer)")
    print("  ✓ Answer generation from pre-retrieved context")
    print("  ✓ Proper citation formatting [doc:X]")
    print("  ✓ Graceful handling of missing information")
    print("  ✓ Multi-turn conversation awareness")
    print("\n💡 Next Steps:")
    print("  1. Verify all answers have proper citations")
    print("  2. Check for hallucinations (claims not in context)")
    print("  3. Test with real medical documents")
    print("  4. Tune temperature if answers too creative/deterministic")
    print("  5. Integrate with Validation Agent for quality checks")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_answer_agent()
