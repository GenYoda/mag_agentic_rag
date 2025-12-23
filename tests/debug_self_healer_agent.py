"""
================================================================================
DEBUG SCRIPT: Self-Healer Agent Test
================================================================================
Tests the Self-Healer Agent in isolation with various validation scenarios

Test Scenarios:
1. Missing citations (should add citations)
2. Hallucination detection (should regenerate with stricter grounding)
3. Incomplete answer (should retrieve more context)
4. Citation mismatch (should fix source mapping)
5. Full retry workflow (validation → healing → re-validation)

Prerequisites:
- Self-Healer Agent implemented
- Validation Agent working
- Answer generation tools available
- Retrieval and query tools available
- settings.py configured with self-healing parameters

Run:
    python tests/debug/debug_self_healer_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.self_healer_agent import create_self_healer_agent
from agents.validation_agent import create_validation_agent
from agents.answer_agent import create_answer_agent
from config.settings import SELF_HEAL_MAX_RETRIES, SELF_HEAL_MIN_QUALITY_SCORE


def test_self_healer_agent():
    """Test self-healer agent with various validation failure scenarios."""
    
    print("\n" + "="*80)
    print("SELF-HEALER AGENT DEBUG TEST")
    print("="*80 + "\n")
    print(f"Configuration:")
    print(f"  • Max Retries: {SELF_HEAL_MAX_RETRIES}")
    print(f"  • Min Quality Score: {SELF_HEAL_MIN_QUALITY_SCORE}")
    print()
    
    # ========================================================================
    # Setup: Create Agents
    # ========================================================================
    print("📝 Step 1: Creating Agents...")
    try:
        self_healer_agent = create_self_healer_agent(verbose=True)
        validation_agent = create_validation_agent(verbose=True)
        answer_agent = create_answer_agent(verbose=True)
        
        print(f"✅ Self-Healer Agent created")
        print(f"   Role: {self_healer_agent.role}")
        print(f"   Tools: {len(self_healer_agent.tools)} tools available")
        print(f"   Max Iterations: {self_healer_agent.max_iter}")
        
        print(f"✅ Validation Agent created")
        print(f"✅ Answer Agent created")
    except Exception as e:
        print(f"❌ Failed to create agents: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Missing Citations (Self-Healing)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Self-Heal Missing Citations")
    print("-"*80)
    print("Scenario: Answer has no citations")
    print("Expected: Self-healer adds citations\n")
    
    task1 = Task(
        description=(
            "Self-healing workflow for missing citations:\n\n"
            "VALIDATION RESULT (from Validation Agent):\n"
            "{\n"
            "  'is_valid': False,\n"
            "  'quality_score': 0.65,\n"
            "  'confidence_score': 0.70,\n"
            "  'has_hallucination': False,\n"
            "  'has_citations': False,\n"
            "  'citation_count': 0,\n"
            "  'issues': [\n"
            "    {\n"
            "      'severity': 'warning',\n"
            "      'category': 'citations',\n"
            "      'message': 'Insufficient citations (0, need 1)',\n"
            "      'auto_fixable': True\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "ORIGINAL ANSWER:\n"
            "'The patient was diagnosed with essential hypertension and prescribed "
            "Lisinopril 10mg once daily.'\n\n"
            "CONTEXT CHUNKS:\n"
            "[doc:1] 'Patient diagnosed with essential hypertension on Jan 10, 2024.'\n"
            "[doc:2] 'Treatment: Lisinopril 10mg once daily for blood pressure control.'\n\n"
            "QUERY:\n"
            "'What is the patient diagnosis and treatment?'\n\n"
            "YOUR TASK:\n"
            "1. Analyze the validation result\n"
            "2. Identify issue: Missing citations\n"
            "3. Decision: Regenerate with citation emphasis\n"
            "4. Use generate_answer_with_context_tool with instruction:\n"
            "   'CRITICAL: Cite every factual claim with [doc:X]'\n"
            "5. Return improved answer with citations"
        ),
        agent=self_healer_agent,
        expected_output=(
            "Improved answer with citations:\n"
            "'The patient was diagnosed with essential hypertension [doc:1] and "
            "prescribed Lisinopril 10mg once daily [doc:2].'\n\n"
            "Healing action taken: Regenerated with citation emphasis\n"
            "Expected validation: PASS (quality_score > 0.80)"
        )
    )
    
    crew1 = Crew(
        agents=[self_healer_agent],
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
    # Test 2: Hallucination Detection (Self-Healing)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Self-Heal Hallucination")
    print("-"*80)
    print("Scenario: Answer contains invented information")
    print("Expected: Self-healer regenerates with stricter grounding\n")
    
    task2 = Task(
        description=(
            "Self-healing workflow for hallucination:\n\n"
            "VALIDATION RESULT:\n"
            "{\n"
            "  'is_valid': False,\n"
            "  'quality_score': 0.55,\n"
            "  'confidence_score': 0.60,\n"
            "  'has_hallucination': True,\n"
            "  'issues': [\n"
            "    {\n"
            "      'severity': 'critical',\n"
            "      'category': 'hallucination',\n"
            "      'message': 'LLM detected potential hallucination',\n"
            "      'details': {\n"
            "        'claims': ['dosage 10mg not stated', 'date March 15 not in context'],\n"
            "        'reasoning': 'Answer includes specific details not in source'\n"
            "      }\n"
            "    }\n"
            "  ],\n"
            "  'llm_judge_reasoning': 'Answer contains dosage and date not present in context'\n"
            "}\n\n"
            "ORIGINAL ANSWER (with hallucinations):\n"
            "'Patient diagnosed on March 15, 2024 and prescribed Lisinopril 10mg.'\n\n"
            "CONTEXT CHUNKS (no date or dosage):\n"
            "[doc:1] 'Patient diagnosed with hypertension.'\n"
            "[doc:2] 'Treatment includes Lisinopril.'\n\n"
            "QUERY:\n"
            "'When was patient diagnosed and what medication?'\n\n"
            "YOUR TASK:\n"
            "1. Identify issue: Hallucination (invented date and dosage)\n"
            "2. Decision: Regenerate with stricter grounding\n"
            "3. Use generate_answer_with_context_tool with instruction:\n"
            "   'ONLY use information EXPLICITLY stated in context. Do NOT add dates, "
            "dosages, or other details not present.'\n"
            "4. Return answer WITHOUT hallucinated details"
        ),
        agent=self_healer_agent,
        expected_output=(
            "Improved answer without hallucinations:\n"
            "'Patient was diagnosed with hypertension [doc:1]. Treatment includes "
            "Lisinopril [doc:2]. Specific date and dosage not mentioned in records.'\n\n"
            "Healing action: Regenerated with strict grounding instruction\n"
            "Removed hallucinated claims: date, dosage"
        )
    )
    
    crew2 = Crew(
        agents=[self_healer_agent],
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
    # Test 3: Incomplete Answer (Retrieve More Context)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Self-Heal Incomplete Answer")
    print("-"*80)
    print("Scenario: Answer missing part of multi-part question")
    print("Expected: Self-healer retrieves more context or decomposes query\n")
    
    task3 = Task(
        description=(
            "Self-healing workflow for incomplete answer:\n\n"
            "VALIDATION RESULT:\n"
            "{\n"
            "  'is_valid': False,\n"
            "  'quality_score': 0.60,\n"
            "  'issues': [\n"
            "    {\n"
            "      'severity': 'warning',\n"
            "      'category': 'completeness',\n"
            "      'message': 'Answer addresses only part of the question',\n"
            "      'details': {\n"
            "        'missing': 'Did not answer: what are the side effects?'\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "ORIGINAL ANSWER (incomplete):\n"
            "'Patient prescribed Lisinopril 10mg for hypertension [doc:1].'\n\n"
            "QUERY (multi-part):\n"
            "'What medication was prescribed and what are the potential side effects?'\n\n"
            "CONTEXT CHUNKS (limited):\n"
            "[doc:1] 'Prescribed Lisinopril 10mg for hypertension control.'\n"
            "(Missing: side effects information)\n\n"
            "YOUR TASK:\n"
            "1. Identify issue: Incomplete answer (missing side effects)\n"
            "2. Decision: Retrieve more context about Lisinopril side effects\n"
            "3. Strategy A: Use search_tool to get additional chunks about side effects\n"
            "4. Strategy B: Use decompose_query_tool to separate questions\n"
            "5. Regenerate complete answer covering both parts"
        ),
        agent=self_healer_agent,
        expected_output=(
            "Complete answer addressing both parts:\n"
            "'Patient was prescribed Lisinopril 10mg for hypertension control [doc:1]. "
            "Potential side effects of Lisinopril include dizziness, dry cough, and "
            "fatigue [doc:3][doc:4].'\n\n"
            "Healing actions:\n"
            "1. Retrieved additional context about side effects\n"
            "2. Regenerated with complete information\n"
            "Both questions answered"
        )
    )
    
    crew3 = Crew(
        agents=[self_healer_agent],
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
    # Test 4: Question Type Format (Multiple Choice)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Self-Heal Question Format (Multiple Choice)")
    print("-"*80)
    print("Scenario: Answer doesn't match multiple choice format")
    print("Expected: Self-healer reformats as multiple choice\n")
    
    task4 = Task(
        description=(
            "Self-healing workflow for format mismatch:\n\n"
            "VALIDATION RESULT:\n"
            "{\n"
            "  'is_valid': False,\n"
            "  'quality_score': 0.50,\n"
            "  'issues': [\n"
            "    {\n"
            "      'severity': 'warning',\n"
            "      'category': 'quality',\n"
            "      'message': 'Answer format does not match question type',\n"
            "      'details': {\n"
            "        'expected': 'Multiple choice format (A/B/C/D)',\n"
            "        'got': 'Paragraph format'\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "ORIGINAL ANSWER (wrong format):\n"
            "'Hypertension is managed with medication and lifestyle changes [doc:1].'\n\n"
            "QUERY (multiple choice):\n"
            "'Select the correct treatment for hypertension:\\n"
            "A) Antibiotics only\\n"
            "B) Medication and lifestyle changes\\n"
            "C) Surgery only\\n"
            "D) No treatment needed'\n\n"
            "CONTEXT:\n"
            "[doc:1] 'Hypertension managed with medication and lifestyle modifications.'\n\n"
            "YOUR TASK:\n"
            "1. Identify issue: Wrong format (paragraph instead of A/B/C/D)\n"
            "2. Decision: Regenerate in multiple choice format\n"
            "3. Use generate_answer_with_context_tool with instruction:\n"
            "   'Provide answer as single letter (A/B/C/D) with brief reasoning'\n"
            "4. Return properly formatted multiple choice answer"
        ),
        agent=self_healer_agent,
        expected_output=(
            "Properly formatted multiple choice answer:\n"
            "'The correct answer is B) Medication and lifestyle changes [doc:1].\\n\\n"
            "Reasoning: The patient records indicate hypertension is managed through "
            "both medication and lifestyle modifications.'\n\n"
            "Healing action: Reformatted to match question type"
        )
    )
    
    crew4 = Crew(
        agents=[self_healer_agent],
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
    # Test 5: Full Retry Workflow (Multi-Agent)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Full Self-Healing Workflow (Multi-Agent)")
    print("-"*80)
    print("Scenario: Complete workflow with retry loop")
    print("Expected: Answer → Validation → Healing → Re-validation\n")
    
    task5_answer = Task(
        description=(
            "Generate answer for query:\n"
            "'What medications were prescribed to the patient?'\n\n"
            "Context:\n"
            "[doc:1] 'Patient prescribed Lisinopril for blood pressure.'\n"
            "[doc:2] 'Metformin prescribed for diabetes management.'\n\n"
            "Generate initial answer (may have issues)."
        ),
        agent=answer_agent,
        expected_output="Answer to the query"
    )
    
    task5_validate = Task(
        description=(
            "Validate the generated answer.\n"
            "Check for: citations, hallucinations, completeness, quality.\n"
            "Return detailed validation result."
        ),
        agent=validation_agent,
        expected_output="Validation result with is_valid, quality_score, issues"
    )
    
    task5_heal = Task(
        description=(
            "IF validation failed (quality_score < 0.80):\n"
            "1. Analyze validation issues\n"
            "2. Select appropriate healing strategy\n"
            "3. Execute corrective action\n"
            "4. Return improved answer\n"
            "5. (Answer will be re-validated)\n\n"
            f"Max retries: {SELF_HEAL_MAX_RETRIES}\n"
            "Track best attempt if all retries fail."
        ),
        agent=self_healer_agent,
        expected_output="Healed answer with metadata (retry_count, improvement)"
    )
    
    crew5 = Crew(
        agents=[answer_agent, validation_agent, self_healer_agent],
        tasks=[task5_answer, task5_validate, task5_heal],
        verbose=True
    )
    
    try:
        result5 = crew5.kickoff()
        print(f"\n✅ Test 5 Complete - Full Workflow")
        print(f"Result: {result5}")
    except Exception as e:
        print(f"\n❌ Test 5 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SELF-HEALER AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ Missing citations healing")
    print("  ✓ Hallucination detection and removal")
    print("  ✓ Incomplete answer completion")
    print("  ✓ Question type format correction")
    print("  ✓ Full retry workflow with validation loop")
    print("\n💡 Next Steps:")
    print("  1. Test with real medical Q&A")
    print(f"  2. Tune SELF_HEAL_MIN_QUALITY_SCORE (current: {SELF_HEAL_MIN_QUALITY_SCORE})")
    print(f"  3. Adjust SELF_HEAL_MAX_RETRIES if needed (current: {SELF_HEAL_MAX_RETRIES})")
    print("  4. Monitor healing success rate")
    print("  5. Integrate into production RAG pipeline")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_self_healer_agent()
