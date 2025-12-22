"""
================================================================================
DEBUG SCRIPT: Validation Agent Test
================================================================================
Tests the Validation Agent in isolation

Test Scenarios:
1. Validate high-quality answer (should pass)
2. Detect missing citations (should fail)
3. Detect hallucinations (LLM-as-Judge)
4. Validate citation accuracy
5. Test self-healing workflow

Prerequisites:
- Validation tools must be implemented
- Azure OpenAI configured (for LLM-as-Judge)
- Answer generation tools working (for regeneration)

Run:
    python tests/debug/debug_validation_agent.py
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crewai import Task, Crew
from agents.validation_agent import create_validation_agent


def test_validation_agent():
    """Test validation agent with various answer quality scenarios."""
    
    print("\n" + "="*80)
    print("VALIDATION AGENT DEBUG TEST")
    print("="*80 + "\n")
    
    # ========================================================================
    # Setup: Create Validation Agent
    # ========================================================================
    print("📝 Step 1: Creating Validation Agent...")
    try:
        validation_agent = create_validation_agent(verbose=True)
        print(f"✅ Agent created successfully")
        print(f"   Role: {validation_agent.role}")
        print(f"   Tools: {len(validation_agent.tools)} tools loaded")
        print(f"   Max Iterations: {validation_agent.max_iter}")
        print(f"   Can Delegate: {validation_agent.allow_delegation}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # Test 1: Validate High-Quality Answer (Should PASS)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 1: Validate High-Quality Answer")
    print("-"*80)
    print("Expected: PASS with quality score > 0.90\n")
    
    task1 = Task(
        description=(
            "Validate this high-quality answer:\n\n"
            "Query: 'What medications were prescribed to the patient?'\n\n"
            "Answer: 'The patient was prescribed Lisinopril 10mg once daily for "
            "hypertension [doc:1] and Metformin 500mg twice daily for type 2 diabetes "
            "management [doc:2]. Both medications should be taken consistently with "
            "regular monitoring [doc:3].'\n\n"
            "Context chunks:\n"
            "[doc:1] 'Dr. Johnson prescribed Lisinopril 10mg once daily for hypertension.'\n"
            "[doc:2] 'Metformin 500mg twice daily prescribed for type 2 diabetes management.'\n"
            "[doc:3] 'Patient instructed to take medications consistently and report for "
            "regular blood work monitoring.'\n\n"
            "Use validate_answer_tool to check:\n"
            "- All claims have citations\n"
            "- Citations match context\n"
            "- No hallucinations\n"
            "- Good structure and clarity"
        ),
        agent=validation_agent,
        expected_output=(
            "Validation result:\n"
            "- is_valid: True\n"
            "- quality_score: ≥ 0.90\n"
            "- has_hallucination: False\n"
            "- has_citations: True\n"
            "- citation_count: 3\n"
            "- issues: [] (no issues)\n"
            "- recommendation: 'pass'"
        )
    )
    
    crew1 = Crew(
        agents=[validation_agent],
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
    # Test 2: Detect Missing Citations (Should FAIL)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Detect Missing Citations")
    print("-"*80)
    print("Expected: FAIL - missing citations on factual claims\n")
    
    task2 = Task(
        description=(
            "Validate this answer with MISSING citations:\n\n"
            "Query: 'What is the patient diagnosis?'\n\n"
            "Answer: 'The patient was diagnosed with essential hypertension and type 2 diabetes. "
            "Treatment includes Lisinopril and Metformin. Blood pressure is monitored weekly.'\n\n"
            "Context chunks:\n"
            "[doc:1] 'Patient diagnosed with essential hypertension and type 2 diabetes.'\n"
            "[doc:2] 'Treatment: Lisinopril 10mg, Metformin 500mg.'\n"
            "[doc:3] 'Blood pressure monitored weekly.'\n\n"
            "Problem: Answer has NO citations!\n\n"
            "Use validate_answer_tool to detect this issue."
        ),
        agent=validation_agent,
        expected_output=(
            "Validation result:\n"
            "- is_valid: False\n"
            "- quality_score: < 0.70\n"
            "- has_citations: False\n"
            "- issues: [\n"
            "    {'severity': 'error', 'category': 'citations', 'message': 'Missing citations'}\n"
            "  ]\n"
            "- recommendation: 'regenerate'\n"
            "- suggested_fix: 'Add [doc:X] citations to all factual claims'"
        )
    )
    
    crew2 = Crew(
        agents=[validation_agent],
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
    # Test 3: Detect Hallucinations (LLM-as-Judge)
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Detect Hallucinations")
    print("-"*80)
    print("Expected: FAIL - hallucinated information not in context\n")
    
    task3 = Task(
        description=(
            "Validate this answer with HALLUCINATIONS:\n\n"
            "Query: 'What is the patient diagnosis?'\n\n"
            "Answer: 'The patient was diagnosed with essential hypertension [doc:1] "
            "on March 15, 2024 at 2:30 PM by Dr. Sarah Johnson, a board-certified "
            "cardiologist with 15 years of experience.'\n\n"
            "Context chunks:\n"
            "[doc:1] 'Patient diagnosed with essential hypertension.'\n\n"
            "Hallucinations:\n"
            "- Date/time (March 15, 2024, 2:30 PM) - NOT in context\n"
            "- Doctor name (Dr. Sarah Johnson) - NOT in context\n"
            "- Doctor credentials (board-certified, 15 years) - NOT in context\n\n"
            "Use validate_answer_tool with LLM-as-Judge to detect these."
        ),
        agent=validation_agent,
        expected_output=(
            "Validation result:\n"
            "- is_valid: False\n"
            "- quality_score: < 0.50\n"
            "- has_hallucination: True\n"
            "- llm_judge_used: True\n"
            "- llm_judge_reasoning: 'Answer contains information not present in context'\n"
            "- issues: [\n"
            "    {'severity': 'critical', 'category': 'hallucination', "
            "'message': 'Invented facts not in source'}\n"
            "  ]\n"
            "- recommendation: 'regenerate'"
        )
    )
    
    crew3 = Crew(
        agents=[validation_agent],
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
    # Test 4: Validate Citation Accuracy
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Validate Citation Accuracy")
    print("-"*80)
    print("Expected: FAIL - citation mismatch (wrong source cited)\n")
    
    task4 = Task(
        description=(
            "Validate this answer with INCORRECT citations:\n\n"
            "Query: 'What medications were prescribed?'\n\n"
            "Answer: 'Patient prescribed Lisinopril 10mg [doc:2] and "
            "Metformin 500mg [doc:1].'\n\n"
            "Context chunks:\n"
            "[doc:1] 'Patient prescribed Lisinopril 10mg for hypertension.'\n"
            "[doc:2] 'Patient prescribed Metformin 500mg for diabetes.'\n\n"
            "Problem: Citations are SWAPPED!\n"
            "- Lisinopril is cited as [doc:2] but it's in [doc:1]\n"
            "- Metformin is cited as [doc:1] but it's in [doc:2]\n\n"
            "Detect this citation mismatch."
        ),
        agent=validation_agent,
        expected_output=(
            "Validation result:\n"
            "- is_valid: False\n"
            "- quality_score: < 0.70\n"
            "- issues: [\n"
            "    {'severity': 'error', 'category': 'citation_accuracy', "
            "'message': 'Citation mismatch - claim does not match cited source'}\n"
            "  ]\n"
            "- recommendation: 'regenerate'\n"
            "- suggested_fix: 'Correct citation IDs to match actual sources'"
        )
    )
    
    crew4 = Crew(
        agents=[validation_agent],
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
    # Test 5: Get Validation Summary
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Get Human-Readable Validation Summary")
    print("-"*80)
    print("Convert validation dict to readable summary\n")
    
    task5 = Task(
        description=(
            "Generate a human-readable validation summary from this validation result:\n\n"
            "{\n"
            "  'is_valid': False,\n"
            "  'quality_score': 0.65,\n"
            "  'confidence_score': 0.70,\n"
            "  'has_hallucination': True,\n"
            "  'has_citations': True,\n"
            "  'citation_count': 2,\n"
            "  'issues': [\n"
            "    {'severity': 'critical', 'category': 'hallucination', "
            "'message': 'Invented date not in context'},\n"
            "    {'severity': 'warning', 'category': 'quality', "
            "'message': 'Answer could be more concise'}\n"
            "  ],\n"
            "  'warnings': ['Minor formatting issue'],\n"
            "  'llm_judge_used': True,\n"
            "  'llm_judge_reasoning': 'Answer includes specific date not mentioned in source'\n"
            "}\n\n"
            "Use get_validation_summary_tool to format this nicely."
        ),
        agent=validation_agent,
        expected_output=(
            "Human-readable summary:\n\n"
            "VALIDATION FAILED ❌\n"
            "Quality Score: 65% (Fair)\n"
            "Confidence: 70%\n\n"
            "Issues Found:\n"
            "- ❌ CRITICAL: Invented date not in context (hallucination)\n"
            "- ⚠️  WARNING: Answer could be more concise (quality)\n\n"
            "LLM Judge: Answer includes specific date not mentioned in source\n\n"
            "Recommendation: Regenerate answer to fix hallucination"
        )
    )
    
    crew5 = Crew(
        agents=[validation_agent],
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
    print("VALIDATION AGENT TESTING COMPLETE")
    print("="*80)
    print("\n📊 Summary:")
    print("  ✓ High-quality answer validation (pass)")
    print("  ✓ Missing citation detection (fail)")
    print("  ✓ Hallucination detection with LLM-as-Judge")
    print("  ✓ Citation accuracy verification")
    print("  ✓ Human-readable validation summary")
    print("\n💡 Next Steps:")
    print("  1. Test self-healing workflow (regeneration)")
    print("  2. Tune quality score thresholds in settings")
    print("  3. Test with real medical Q&A pairs")
    print("  4. Integrate into full RAG pipeline")
    print("  5. Monitor validation metrics over time")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_validation_agent()
