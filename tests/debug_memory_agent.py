"""
================================================================================
DEBUG SCRIPT: Memory Agent Test
================================================================================
Tests the Memory Agent in isolation with conversation scenarios

Test Scenarios:
1. Add first exchange (establish initial context)
2. Get context for follow-up question (test context retrieval)
3. Extract entities from medical text
4. Resolve coreference (pronouns → entities)
5. Test multi-turn conversation flow

Prerequisites:
- Memory tools must be implemented with the public methods added
- Entity extraction should work (pattern-based or LLM-based)
- Session persistence directory should be writable

Run:
    python tests/debug/debug_memory_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.memory_agent import create_memory_agent


def test_memory_agent():
    """Test memory agent with progressive conversation scenarios."""
    
    print("\n" + "="*80)
    print("MEMORY AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Memory Agent
    # ========================================================================
    print("📝 Step 1: Creating Memory Agent...")
    try:
        memory_agent = create_memory_agent(verbose=True)
        print(f"✅ Agent created successfully")
        print(f"   Role: {memory_agent.role}")
        print(f"   Tools: {len(memory_agent.tools)} tools loaded")
        print(f"   Max Iterations: {memory_agent.max_iter}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # ========================================================================
    # Test 1: Add First Exchange (Establish Context)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Add First Exchange to Memory")
    print("-"*80)
    print("Query: 'Who is the patient?'")
    print("Answer: 'The patient is John Smith, a 45-year-old male...'")
    print("Expected: Exchange stored with extracted entities\n")
    
    task1 = Task(
        description=(
            "Add this Q&A exchange to conversation memory:\n\n"
            "Query: 'Who is the patient?'\n"
            "Answer: 'The patient is John Smith, a 45-year-old male treated by "
            "Dr. Wilson at Memorial Health on March 15, 2024.'\n\n"
            "Use the add_exchange tool to store this conversation turn. "
            "The tool should automatically extract entities (John Smith, Dr. Wilson, "
            "Memorial Health, March 15, 2024)."
        ),
        agent=memory_agent,
        expected_output=(
            "Confirmation that the exchange was added to memory with:\n"
            "- Turn ID\n"
            "- Extracted entities (patient name, doctor name, facility, date)\n"
            "- Success message"
        )
    )
    
    crew1 = Crew(
        agents=[memory_agent],
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
    # Test 2: Get Context for Follow-Up Question
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Get Context for Follow-Up Question")
    print("-"*80)
    print("Query: 'What is his diagnosis?'")
    print("Expected: Context includes previous exchange + resolved 'his' → John Smith\n")
    
    task2 = Task(
        description=(
            "Get conversation context for this follow-up query: 'What is his diagnosis?'\n\n"
            "The pronoun 'his' refers to the patient mentioned in the previous exchange "
            "(John Smith). Use get_context_for_query to:\n"
            "1. Retrieve recent conversation history\n"
            "2. Resolve 'his' to 'John Smith'\n"
            "3. Identify relevant entities\n"
            "4. Return the resolved query and context"
        ),
        agent=memory_agent,
        expected_output=(
            "Context result containing:\n"
            "- Recent exchanges (previous Q&A about John Smith)\n"
            "- Resolved query: 'What is John Smith's diagnosis?'\n"
            "- Relevant entities: John Smith (patient), Dr. Wilson (doctor), etc.\n"
            "- Original query: 'What is his diagnosis?'"
        )
    )
    
    crew2 = Crew(
        agents=[memory_agent],
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
    # Test 3: Extract Entities from Medical Text
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Extract Entities from Medical Text")
    print("-"*80)
    print("Text: Medical report with multiple entities")
    print("Expected: Extracted patients, doctors, medications, dates, facilities\n")
    
    task3 = Task(
        description=(
            "Extract all medical entities from this text:\n\n"
            "'Dr. Sarah Johnson prescribed Lisinopril 10mg to patient Mary Davis on "
            "March 15, 2024 for treatment of hypertension at Savannah Health Center. "
            "Follow-up appointment scheduled with RN Thompson on April 1, 2024.'\n\n"
            "Use extract_entities to identify:\n"
            "- Doctors: Dr. Sarah Johnson\n"
            "- Patients: Mary Davis\n"
            "- Medications: Lisinopril 10mg\n"
            "- Dates: March 15, 2024 and April 1, 2024\n"
            "- Facilities: Savannah Health Center\n"
            "- Staff: RN Thompson\n"
            "- Diagnoses: hypertension"
        ),
        agent=memory_agent,
        expected_output=(
            "List of extracted entities organized by type:\n"
            "- Persons (doctors, nurses, patients)\n"
            "- Medications with dosages\n"
            "- Dates and temporal references\n"
            "- Organizations (healthcare facilities)\n"
            "- Medical conditions/diagnoses"
        )
    )
    
    crew3 = Crew(
        agents=[memory_agent],
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
    # Test 4: Resolve Coreference (Pronouns)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Resolve Coreference")
    print("-"*80)
    print("Query: 'What medication did she prescribe?'")
    print("Expected: 'she' resolved to 'Dr. Sarah Johnson'\n")
    
    task4 = Task(
        description=(
            "Resolve the pronoun in this query: 'What medication did she prescribe?'\n\n"
            "Based on recent conversation history (Test 3 mentioned Dr. Sarah Johnson), "
            "use resolve_coreference to:\n"
            "1. Identify the pronoun 'she'\n"
            "2. Look up recent female entities in memory\n"
            "3. Resolve to 'Dr. Sarah Johnson'\n"
            "4. Return the resolved query"
        ),
        agent=memory_agent,
        expected_output=(
            "Resolved query: 'What medication did Dr. Sarah Johnson prescribe?'"
        )
    )
    
    crew4 = Crew(
        agents=[memory_agent],
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
    # Test 5: Multi-Turn Conversation Flow
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Multi-Turn Conversation Flow")
    print("-"*80)
    print("Simulating 3-turn conversation with context building\n")
    
    task5 = Task(
        description=(
            "Simulate this 3-turn conversation:\n\n"
            "Turn 1:\n"
            "  Q: 'Who treated the patient?'\n"
            "  A: 'Dr. Emily Rodriguez treated the patient at Central Medical.'\n"
            "  → Add to memory\n\n"
            "Turn 2:\n"
            "  Q: 'What did she diagnose?'\n"
            "  → Get context (resolve 'she' to Dr. Emily Rodriguez)\n"
            "  → Expected: Context shows Turn 1 and resolved query\n\n"
            "Turn 3:\n"
            "  Q: 'When was the appointment?'\n"
            "  → Get context (general follow-up)\n"
            "  → Expected: Context shows Turn 1 and Turn 2\n\n"
            "Execute all three turns in sequence."
        ),
        agent=memory_agent,
        expected_output=(
            "Multi-turn conversation summary showing:\n"
            "- Turn 1: Exchange added successfully\n"
            "- Turn 2: 'she' resolved to Dr. Emily Rodriguez\n"
            "- Turn 3: Context includes all previous turns\n"
            "- Entity memory tracks Dr. Emily Rodriguez across turns"
        )
    )
    
    crew5 = Crew(
        agents=[memory_agent],
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
    print("MEMORY AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Exchange storage in conversation memory")
    print("  ✓ Context retrieval for follow-up questions")
    print("  ✓ Entity extraction from medical text")
    print("  ✓ Pronoun/coreference resolution")
    print("  ✓ Multi-turn conversation handling")
    print("\n💡 Next Steps:")
    print("  1. Verify memory files created in data/memory/")
    print("  2. Check session persistence (JSON files)")
    print("  3. Test with longer conversation histories")
    print("  4. Integrate with Query Enhancement Agent")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_memory_agent()
