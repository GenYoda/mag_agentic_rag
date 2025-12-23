"""
================================================================================
DEBUG SCRIPT: Query Enhancement Agent Test
================================================================================
Tests the Query Enhancement Agent with various query types

Test Scenarios:
1. Classify simple factual query
2. Decompose complex multi-part query
3. Generate hypothetical answer (HyDE)
4. Generate query variations
5. Run full enhancement pipeline

Prerequisites:
- Query tools must be implemented and working
- Azure OpenAI credentials configured in settings
- LLM access for query classification and generation

Run:
    python tests/debug/debug_query_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.query_agent import create_query_agent  # ✅ Corrected import


def test_query_agent():
    """Test query enhancement with various query types."""
    
    print("\n" + "="*80)
    print("QUERY ENHANCEMENT AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Query Enhancement Agent
    # ========================================================================
    print("📝 Step 1: Creating Query Enhancement Agent...")
    try:
        query_agent = create_query_agent(verbose=True)  # ✅ Corrected function name
        print(f"✅ Agent created successfully")
        print(f"   Role: {query_agent.role}")
        print(f"   Tools: {len(query_agent.tools)} tools loaded")
        print(f"   Max Iterations: {query_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Classify Simple Factual Query
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Classify Simple Factual Query")
    print("-"*80)
    print("Query: 'What is the patient's diagnosis?'")
    print("Expected: Type=factual, Complexity=simple, recommendations\n")
    
    task1 = Task(
        description=(
            "Classify this query: 'What is the patient's diagnosis?'\n\n"
            "Use the classify_query tool to determine:\n"
            "- Query type (factual, analytical, comparison, list)\n"
            "- Complexity (simple or complex)\n"
            "- Whether it requires decomposition\n"
            "- Whether HyDE would help\n"
            "- Whether query variations are needed"
        ),
        agent=query_agent,
        expected_output=(
            "Query classification result with:\n"
            "- Type: factual (asking for specific information)\n"
            "- Complexity: simple (single question)\n"
            "- Requires decomposition: No\n"
            "- Requires HyDE: Recommended for better retrieval\n"
            "- Requires variations: Yes (for broader coverage)"
        )
    )
    
    crew1 = Crew(
        agents=[query_agent],
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
    # Test 2: Decompose Complex Multi-Part Query
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Decompose Complex Multi-Part Query")
    print("-"*80)
    print("Query: Complex question with 3 parts")
    print("Expected: 3 simple sub-queries\n")
    
    task2 = Task(
        description=(
            "Decompose this complex query into simple sub-queries:\n\n"
            "'What was the patient diagnosed with, who was the treating physician, "
            "and what medications were prescribed?'\n\n"
            "Use the decompose_query tool to break this into atomic sub-questions. "
            "Each sub-question should focus on ONE aspect and be answerable independently."
        ),
        agent=query_agent,
        expected_output=(
            "List of 3 simple sub-queries:\n"
            "1. What was the patient diagnosed with?\n"
            "2. Who was the treating physician?\n"
            "3. What medications were prescribed?"
        )
    )
    
    crew2 = Crew(
        agents=[query_agent],
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
    # Test 3: Generate Hypothetical Answer (HyDE)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Generate Hypothetical Answer (HyDE)")
    print("-"*80)
    print("Query: 'What treatment was provided for the patient's hypertension?'")
    print("Expected: Hypothetical medical document excerpt\n")
    
    task3 = Task(
        description=(
            "Generate a hypothetical answer for this query:\n\n"
            "'What treatment was provided for the patient's hypertension?'\n\n"
            "Use the generate_hypothetical_answer tool (HyDE technique) to create a "
            "detailed, document-like answer that would typically appear in a medical record. "
            "This helps retrieval by providing text similar to what we're looking for.\n\n"
            "The hypothetical answer should:\n"
            "- Use formal medical language\n"
            "- Include specific details (medications, dosages, lifestyle changes)\n"
            "- Be 2-3 sentences long\n"
            "- Read like an actual medical record excerpt"
        ),
        agent=query_agent,
        expected_output=(
            "Hypothetical answer example:\n"
            "'The patient was treated for essential hypertension with Lisinopril 10mg "
            "once daily. Treatment plan also included lifestyle modifications including "
            "a low-sodium diet (2000mg/day), regular aerobic exercise 30 minutes daily, "
            "and weight management. Blood pressure was monitored weekly with target "
            "of <130/80 mmHg.'"
        )
    )
    
    crew3 = Crew(
        agents=[query_agent],
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
    # Test 4: Generate Query Variations
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Generate Query Variations")
    print("-"*80)
    print("Query: 'What medications did the doctor prescribe?'")
    print("Expected: 3 alternative phrasings\n")
    
    task4 = Task(
        description=(
            "Generate 3 alternative phrasings for this query:\n\n"
            "'What medications did the doctor prescribe?'\n\n"
            "Use the generate_query_variations tool to create semantically equivalent "
            "variations with different wording. This helps retrieve documents that use "
            "different terminology.\n\n"
            "Variations should:\n"
            "- Mean the same thing\n"
            "- Use synonyms (e.g., 'medications' → 'drugs', 'pharmaceuticals')\n"
            "- Use different sentence structures\n"
            "- Be concise and focused"
        ),
        agent=query_agent,
        expected_output=(
            "List of 3+ query variations:\n"
            "1. What medications did the doctor prescribe? (original)\n"
            "2. Which drugs were prescribed by the physician?\n"
            "3. What pharmaceutical treatment was ordered?\n"
            "4. What prescription medications were given?"
        )
    )
    
    crew4 = Crew(
        agents=[query_agent],
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
    # Test 5: Full Enhancement Pipeline
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Full Query Enhancement Pipeline")
    print("-"*80)
    print("Query: Complex comparison query")
    print("Expected: Complete enhancement with all techniques\n")
    
    task5 = Task(
        description=(
            "Run the FULL enhancement pipeline on this complex query:\n\n"
            "'Compare the patient's current blood pressure readings to previous readings "
            "and explain any changes in the treatment plan.'\n\n"
            "Use the enhance_query tool which runs the complete pipeline:\n"
            "1. Classify the query (type: comparison, complexity: complex)\n"
            "2. Decompose into sub-queries if needed\n"
            "3. Generate HyDE (hypothetical answer)\n"
            "4. Create query variations\n\n"
            "This is the main tool that orchestrates all enhancement techniques."
        ),
        agent=query_agent,
        expected_output=(
            "Complete enhancement result with:\n"
            "- Classification: {type: comparison, complexity: complex}\n"
            "- Sub-queries: ['current BP readings?', 'previous BP readings?', 'treatment changes?']\n"
            "- HyDE text: Hypothetical medical note about BP comparison\n"
            "- Query variations: Alternative phrasings\n"
            "- Enhanced: True"
        )
    )
    
    crew5 = Crew(
        agents=[query_agent],
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
    print("QUERY ENHANCEMENT AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Query classification (type, complexity)")
    print("  ✓ Complex query decomposition")
    print("  ✓ HyDE hypothetical answer generation")
    print("  ✓ Query variation creation")
    print("  ✓ Full enhancement pipeline")
    print("\n💡 Next Steps:")
    print("  1. Test with real medical queries from your domain")
    print("  2. Tune HyDE temperature if needed (in settings)")
    print("  3. Verify query variations are semantically equivalent")
    print("  4. Integrate with Retrieval Agent for end-to-end testing")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_query_agent()
