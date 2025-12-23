"""
================================================================================
PHASE 3: INTELLIGENT SELF-HEALING - OFFLINE TEST SUITE
================================================================================
Purpose: Test Phase 3 components without requiring network/Azure connection

Tests:
1. Diagnostic Engine - Issue analysis and technique recommendation
2. Healing Techniques - Each technique's output format
3. Validation Enhancement - Enhanced validation output
4. Integration - Full diagnostic → technique flow

Can be run offline with mock data.
================================================================================
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import json
from typing import Dict, List, Any

# Phase 3 imports
from tools.diagnostic_engine import (
    diagnose_validation_failure,
    classify_issue_severity,
    recommend_healing_technique,
    DiagnosisResult
)

from tools.healing_techniques import (
    add_citation_emphasis,
    strict_grounding,
    expand_context,
    enforce_format,
    rephrase_query,
    simplify_answer,
    fix_citation_format,
    regenerate_with_emphasis,
    get_technique_for_issue
)

from tools.validation_tools import (
    enhance_validation_output,
    get_issue_severity_summary,
    should_retry_with_healing
)

# ============================================================================
# Mock Data
# ============================================================================

MOCK_QUERY = "What medications were prescribed to the patient?"

MOCK_CHUNKS = [
    {
        'text': 'Patient was prescribed Lisinopril 10mg daily for hypertension.',
        'source': 'medical_record.pdf',
        'page': 1
    },
    {
        'text': 'Metformin 500mg twice daily was added for diabetes management.',
        'source': 'medical_record.pdf',
        'page': 2
    },
    {
        'text': 'Follow-up appointment scheduled for 3 months.',
        'source': 'medical_record.pdf',
        'page': 3
    }
]

MOCK_ANSWER_MISSING_CITATIONS = """
The patient was prescribed Lisinopril 10mg daily for blood pressure and 
Metformin 500mg twice daily for diabetes.
"""

MOCK_ANSWER_WITH_CITATIONS = """
The patient was prescribed Lisinopril 10mg daily for hypertension [doc:0] and 
Metformin 500mg twice daily for diabetes management [doc:1].
"""

MOCK_ANSWER_HALLUCINATED = """
The patient was prescribed Lisinopril 10mg daily [doc:0], Metformin 500mg [doc:1],
and Atorvastatin 20mg for cholesterol.
"""

MOCK_ANSWER_TOO_VERBOSE = """
The patient, who was experiencing elevated blood pressure readings consistent with
a diagnosis of hypertension, was subsequently prescribed Lisinopril at a dosage of
10 milligrams to be taken once daily. Additionally, the patient, having been diagnosed
with type 2 diabetes mellitus, was started on Metformin at a dosage of 500 milligrams
to be administered twice per day for glycemic control.
"""

# ============================================================================
# Test Suite
# ============================================================================

def print_header(title: str):
    """Print test section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_test(name: str):
    """Print test name."""
    print(f"\n→ Test: {name}")
    print("-" * 80)

def print_result(label: str, value: Any, indent: int = 2):
    """Print test result."""
    indent_str = " " * indent
    if isinstance(value, (dict, list)):
        print(f"{indent_str}{label}:")
        print(f"{indent_str}  {json.dumps(value, indent=2)}")
    else:
        print(f"{indent_str}{label}: {value}")

def print_pass(message: str = "PASS"):
    """Print pass indicator."""
    print(f"  ✅ {message}")

def print_fail(message: str = "FAIL"):
    """Print fail indicator."""
    print(f"  ❌ {message}")

# ============================================================================
# Test 1: Diagnostic Engine
# ============================================================================

def test_diagnostic_engine():
    """Test diagnostic engine with various validation failures."""
    print_header("TEST 1: DIAGNOSTIC ENGINE")
    
    # Test Case 1.1: Missing Citations
    print_test("1.1: Diagnose missing citations")
    
    validation_missing_citations = {
        'quick_pass': False,
        'issues': [
            {
                'type': 'missing_citations',
                'severity': 'high',
                'description': 'Answer lacks source citations'
            }
        ],
        'quality_score': 0.6,
        'confidence': 'low'
    }
    
    diagnosis = diagnose_validation_failure(
        validation_missing_citations,
        MOCK_QUERY,
        MOCK_CHUNKS,
        retry_count=0
    )
    
    print_result("Root Cause", diagnosis.root_cause)
    print_result("Primary Issue", diagnosis.primary_issue_type)
    print_result("Recommended Techniques", diagnosis.recommended_techniques)
    print_result("Escalation Level", diagnosis.escalation_level)
    print_result("Confidence", f"{diagnosis.confidence:.2f}")
    
    assert diagnosis.root_cause == 'llm_instruction_following', "Wrong root cause"
    assert 'add_citation_emphasis' in diagnosis.recommended_techniques, "Wrong technique"
    assert diagnosis.escalation_level == 1, "Wrong escalation level"
    print_pass("Missing citations diagnosed correctly")
    
    # Test Case 1.2: Hallucination
    print_test("1.2: Diagnose hallucination")
    
    validation_hallucination = {
        'quick_pass': False,
        'issues': [
            {
                'type': 'hallucination',
                'severity': 'critical',
                'description': 'Answer contains ungrounded claims'
            }
        ],
        'quality_score': 0.4,
        'confidence': 'low'
    }
    
    diagnosis = diagnose_validation_failure(
        validation_hallucination,
        MOCK_QUERY,
        MOCK_CHUNKS,
        retry_count=0
    )
    
    print_result("Root Cause", diagnosis.root_cause)
    print_result("Recommended Techniques", diagnosis.recommended_techniques)
    
    assert diagnosis.root_cause == 'context_grounding_failure', "Wrong root cause"
    assert 'strict_grounding' in diagnosis.recommended_techniques, "Wrong technique"
    print_pass("Hallucination diagnosed correctly")
    
    # Test Case 1.3: Escalation on retry
    print_test("1.3: Escalation level increases with retries")
    
    diagnosis_retry0 = diagnose_validation_failure(
        validation_missing_citations, MOCK_QUERY, MOCK_CHUNKS, retry_count=0
    )
    diagnosis_retry1 = diagnose_validation_failure(
        validation_missing_citations, MOCK_QUERY, MOCK_CHUNKS, retry_count=1
    )
    diagnosis_retry2 = diagnose_validation_failure(
        validation_missing_citations, MOCK_QUERY, MOCK_CHUNKS, retry_count=2
    )
    
    print_result("Retry 0 Escalation", diagnosis_retry0.escalation_level)
    print_result("Retry 1 Escalation", diagnosis_retry1.escalation_level)
    print_result("Retry 2 Escalation", diagnosis_retry2.escalation_level)
    
    assert diagnosis_retry0.escalation_level == 1, "Retry 0 should be level 1"
    assert diagnosis_retry1.escalation_level == 2, "Retry 1 should be level 2"
    assert diagnosis_retry2.escalation_level == 3, "Retry 2 should be level 3"
    print_pass("Escalation levels correct")
    
    # Test Case 1.4: Multi-issue prioritization
    print_test("1.4: Multi-issue prioritization")
    
    validation_multi_issue = {
        'quick_pass': False,
        'issues': [
            {'type': 'too_short', 'severity': 'medium'},
            {'type': 'missing_citations', 'severity': 'high'},
            {'type': 'suspiciously_verbose', 'severity': 'low'}
        ],
        'quality_score': 0.5
    }
    
    diagnosis = diagnose_validation_failure(
        validation_multi_issue, MOCK_QUERY, MOCK_CHUNKS, retry_count=0
    )
    
    print_result("Primary Issue", diagnosis.primary_issue_type)
    print_result("Multi-Issue Flag", diagnosis.multi_issue)
    
    assert diagnosis.primary_issue_type == 'missing_citations', "Should prioritize high severity"
    assert diagnosis.multi_issue == True, "Should detect multi-issue"
    print_pass("Multi-issue prioritization correct")

# ============================================================================
# Test 2: Healing Techniques
# ============================================================================

def test_healing_techniques():
    """Test each healing technique's output format."""
    print_header("TEST 2: HEALING TECHNIQUES")
    
    # Test Case 2.1: Add Citation Emphasis
    print_test("2.1: add_citation_emphasis")
    
    result = add_citation_emphasis(
        MOCK_QUERY,
        MOCK_CHUNKS,
        MOCK_ANSWER_MISSING_CITATIONS
    )
    
    print_result("Technique", result['technique'])
    print_result("Has Enhanced Prompt", 'enhanced_prompt' in result)
    print_result("Temperature", result['temperature'])
    print_result("Emphasis Type", result['emphasis'])
    
    assert result['technique'] == 'add_citation_emphasis', "Wrong technique name"
    assert 'enhanced_prompt' in result, "Missing enhanced prompt"
    assert 'CRITICAL' in result['enhanced_prompt'], "Prompt not emphatic enough"
    assert result['temperature'] == 0.1, "Temperature should be low"
    print_pass("Citation emphasis technique correct")
    
    # Test Case 2.2: Strict Grounding
    print_test("2.2: strict_grounding")
    
    result = strict_grounding(
        MOCK_QUERY,
        MOCK_CHUNKS,
        MOCK_ANSWER_HALLUCINATED
    )
    
    print_result("Technique", result['technique'])
    print_result("Temperature", result['temperature'])
    
    assert result['technique'] == 'strict_grounding', "Wrong technique"
    assert 'ONLY' in result['enhanced_prompt'], "Not strict enough"
    assert result['temperature'] <= 0.1, "Temperature too high"
    print_pass("Strict grounding technique correct")
    
    # Test Case 2.3: Expand Context
    print_test("2.3: expand_context")
    
    result = expand_context(MOCK_QUERY, MOCK_CHUNKS, current_top_k=5)
    
    print_result("Technique", result['technique'])
    print_result("Action", result['action'])
    print_result("New Top-K", result['new_top_k'])
    print_result("Rerank", result['rerank'])
    
    assert result['action'] == 'retrieve_more', "Wrong action"
    assert result['new_top_k'] > 5, "Should increase top_k"
    assert result['rerank'] == True, "Should rerank"
    print_pass("Expand context technique correct")
    
    # Test Case 2.4: Enforce Format
    print_test("2.4: enforce_format - auto-detect formats")
    
    # Multiple choice question
    mc_query = "Select the correct option: A) Lisinopril B) Metformin C) Both"
    result_mc = enforce_format(mc_query, MOCK_CHUNKS)
    print_result("Multiple Choice Format", result_mc['expected_format'])
    assert result_mc['expected_format'] == 'multiple_choice', "Should detect MC"
    
    # Bullet list question
    list_query = "List all medications prescribed"
    result_list = enforce_format(list_query, MOCK_CHUNKS)
    print_result("Bullet List Format", result_list['expected_format'])
    assert result_list['expected_format'] == 'bullet_list', "Should detect list"
    
    # Yes/No question
    bool_query = "Is the patient on Lisinopril?"
    result_bool = enforce_format(bool_query, MOCK_CHUNKS)
    print_result("Boolean Format", result_bool['expected_format'])
    assert result_bool['expected_format'] == 'boolean', "Should detect boolean"
    
    print_pass("Format detection works correctly")
    
    # Test Case 2.5: Rephrase Query
    print_test("2.5: rephrase_query")
    
    result = rephrase_query(MOCK_QUERY, MOCK_CHUNKS)
    
    print_result("Technique", result['technique'])
    print_result("Variations Count", len(result['query_variations']))
    print_result("Variations", result['query_variations'])
    
    assert result['action'] == 'retrieve_with_variations', "Wrong action"
    assert len(result['query_variations']) > 1, "Should generate variations"
    print_pass("Query rephrasing generates variations")
    
    # Test Case 2.6: Simplify Answer
    print_test("2.6: simplify_answer")
    
    result = simplify_answer(
        MOCK_QUERY,
        MOCK_CHUNKS,
        MOCK_ANSWER_TOO_VERBOSE
    )
    
    print_result("Technique", result['technique'])
    print_result("Max Tokens", result['max_tokens'])
    print_result("Original Length (words)", result['metadata']['original_length'])
    
    assert result['max_tokens'] == 200, "Should limit tokens"
    assert 'simple' in result['enhanced_prompt'].lower(), "Should emphasize simplicity"
    print_pass("Simplify technique limits verbosity")

# ============================================================================
# Test 3: Technique Selector
# ============================================================================

def test_technique_selector():
    """Test get_technique_for_issue with escalation."""
    print_header("TEST 3: TECHNIQUE SELECTOR")
    
    # Test Case 3.1: Issue → Technique mapping
    print_test("3.1: Issue type to technique mapping")
    
    result = get_technique_for_issue(
        'missing_citations',
        escalation_level=1,
        query=MOCK_QUERY,
        chunks=MOCK_CHUNKS,
        original_answer=MOCK_ANSWER_MISSING_CITATIONS
    )
    
    print_result("Issue Type", 'missing_citations')
    print_result("Escalation Level", 1)
    print_result("Technique Applied", result['technique'])
    
    assert result['technique'] == 'add_citation_emphasis', "Wrong technique for L1"
    print_pass("Level 1 technique correct")
    
    # Test Case 3.2: Escalation changes technique
    print_test("3.2: Escalation changes technique selection")
    
    result_l1 = get_technique_for_issue(
        'hallucination',
        escalation_level=1,
        query=MOCK_QUERY,
        chunks=MOCK_CHUNKS,
        original_answer=MOCK_ANSWER_HALLUCINATED
    )
    
    result_l2 = get_technique_for_issue(
        'hallucination',
        escalation_level=2,
        query=MOCK_QUERY,
        chunks=MOCK_CHUNKS,
        original_answer=MOCK_ANSWER_HALLUCINATED
    )
    
    print_result("L1 Technique", result_l1['technique'])
    print_result("L2 Technique", result_l2['technique'])
    
    assert result_l1['technique'] != result_l2['technique'], "Should use different techniques"
    print_pass("Escalation changes technique")

# ============================================================================
# Test 4: Validation Enhancement
# ============================================================================

def test_validation_enhancement():
    """Test validation output enhancement."""
    print_header("TEST 4: VALIDATION ENHANCEMENT")
    
    # Test Case 4.1: Enhance validation output
    print_test("4.1: Enhance validation with diagnosis")
    
    basic_validation = {
        'quick_pass': False,
        'issues': [
            {'type': 'missing_citations', 'description': 'No citations'}
        ],
        'quality_score': 0.6
    }
    
    enhanced = enhance_validation_output(
        basic_validation,
        MOCK_QUERY,
        MOCK_CHUNKS
    )
    
    print_result("Has Diagnosis", 'diagnosis' in enhanced)
    print_result("Diagnosis Keys", list(enhanced.get('diagnosis', {}).keys()))
    print_result("Issue Has Severity", 'severity' in enhanced['issues'][0])
    print_result("Issue Has Technique", 'recommended_technique' in enhanced['issues'][0])
    
    assert 'diagnosis' in enhanced, "Should add diagnosis"
    assert 'severity' in enhanced['issues'][0], "Should add severity"
    assert 'recommended_technique' in enhanced['issues'][0], "Should add technique"
    print_pass("Validation enhancement adds diagnostic info")
    
    # Test Case 4.2: Severity summary
    print_test("4.2: Issue severity summary")
    
    issues = [
        {'type': 'hallucination', 'severity': 'critical'},
        {'type': 'missing_citations', 'severity': 'high'},
        {'type': 'too_short', 'severity': 'medium'},
        {'type': 'suspiciously_verbose', 'severity': 'low'}
    ]
    
    summary = get_issue_severity_summary(issues)
    
    print_result("Severity Summary", summary)
    
    assert summary['critical'] == 1, "Should count 1 critical"
    assert summary['high'] == 1, "Should count 1 high"
    assert summary['medium'] == 1, "Should count 1 medium"
    assert summary['low'] == 1, "Should count 1 low"
    print_pass("Severity summary correct")
    
    # Test Case 4.3: Retry decision logic
    print_test("4.3: Should retry with healing decision")
    
    # Case: Low quality score
    validation_low_quality = {'quality_score': 0.5, 'issues': [], 'is_valid': True}
    should_retry_1 = should_retry_with_healing(validation_low_quality, quality_threshold=0.8)
    print_result("Low quality → Retry", should_retry_1)
    assert should_retry_1 == True, "Should retry on low quality"
    
    # Case: Critical issues
    validation_critical = {
        'quality_score': 0.9,
        'issues': [{'type': 'hallucination', 'severity': 'critical'}],
        'is_valid': False
    }
    should_retry_2 = should_retry_with_healing(validation_critical)
    print_result("Critical issue → Retry", should_retry_2)
    assert should_retry_2 == True, "Should retry on critical issues"
    
    # Case: High quality, no issues
    validation_good = {'quality_score': 0.95, 'issues': [], 'is_valid': True}
    should_retry_3 = should_retry_with_healing(validation_good)
    print_result("Good quality → No retry", should_retry_3)
    assert should_retry_3 == False, "Should not retry when good"
    
    print_pass("Retry decision logic correct")

# ============================================================================
# Test 5: Integration Test
# ============================================================================

def test_integration():
    """Test full diagnostic → technique → enhancement flow."""
    print_header("TEST 5: INTEGRATION TEST")
    
    print_test("5.1: Full Phase 3 workflow")
    
    # Step 1: Validation fails
    validation_result = {
        'quick_pass': False,
        'issues': [
            {'type': 'missing_citations', 'description': 'No citations found'}
        ],
        'quality_score': 0.55,
        'is_valid': False
    }
    
    print_result("Step 1: Validation", "FAILED (quality: 0.55)")
    
    # Step 2: Enhance with diagnostic info
    enhanced_validation = enhance_validation_output(
        validation_result,
        MOCK_QUERY,
        MOCK_CHUNKS
    )
    
    diagnosis = enhanced_validation['diagnosis']
    print_result("Step 2: Diagnosis", diagnosis)
    
    # Step 3: Get recommended technique
    recommended_technique = diagnosis['recommended_techniques'][0]
    escalation_level = diagnosis['escalation_level']
    
    print_result("Step 3: Recommended Technique", recommended_technique)
    print_result("Step 3: Escalation Level", escalation_level)
    
    # Step 4: Apply technique
    technique_result = get_technique_for_issue(
        diagnosis['primary_issue'],
        escalation_level=escalation_level,
        query=MOCK_QUERY,
        chunks=MOCK_CHUNKS,
        original_answer=MOCK_ANSWER_MISSING_CITATIONS
    )
    
    print_result("Step 4: Technique Result", {
        'technique': technique_result['technique'],
        'has_enhanced_prompt': 'enhanced_prompt' in technique_result,
        'temperature': technique_result.get('temperature', 'N/A')
    })
    
    # Assertions
    assert diagnosis['primary_issue'] == 'missing_citations', "Wrong primary issue"
    assert recommended_technique == 'add_citation_emphasis', "Wrong technique"
    assert technique_result['technique'] == 'add_citation_emphasis', "Technique not applied"
    assert 'enhanced_prompt' in technique_result, "No enhanced prompt"
    
    print_pass("Full Phase 3 workflow complete")

# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all Phase 3 tests."""
    print("\n" + "=" * 80)
    print("  PHASE 3: INTELLIGENT SELF-HEALING - OFFLINE TEST SUITE")
    print("=" * 80)
    print("\n✓ Running offline tests (no network required)")
    
    total_tests = 5
    passed_tests = 0
    
    try:
        test_diagnostic_engine()
        passed_tests += 1
    except AssertionError as e:
        print_fail(f"Diagnostic Engine: {e}")
    except Exception as e:
        print_fail(f"Diagnostic Engine ERROR: {e}")
    
    try:
        test_healing_techniques()
        passed_tests += 1
    except AssertionError as e:
        print_fail(f"Healing Techniques: {e}")
    except Exception as e:
        print_fail(f"Healing Techniques ERROR: {e}")
    
    try:
        test_technique_selector()
        passed_tests += 1
    except AssertionError as e:
        print_fail(f"Technique Selector: {e}")
    except Exception as e:
        print_fail(f"Technique Selector ERROR: {e}")
    
    try:
        test_validation_enhancement()
        passed_tests += 1
    except AssertionError as e:
        print_fail(f"Validation Enhancement: {e}")
    except Exception as e:
        print_fail(f"Validation Enhancement ERROR: {e}")
    
    try:
        test_integration()
        passed_tests += 1
    except AssertionError as e:
        print_fail(f"Integration Test: {e}")
    except Exception as e:
        print_fail(f"Integration Test ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"  TEST SUMMARY: {passed_tests}/{total_tests} PASSED")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print("\n✅ ALL PHASE 3 TESTS PASSED!")
        print("\nPhase 3 components are working correctly.")
        print("Ready for integration testing once network is available.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TEST(S) FAILED")
        print("\nReview failures above and fix before proceeding.")
        return 1

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
