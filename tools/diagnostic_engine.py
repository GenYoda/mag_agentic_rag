

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class DiagnosisResult:
    """Result of diagnostic analysis"""
    root_cause: str  # Primary failure reason
    primary_issue_type: str  # Most critical issue
    recommended_techniques: List[str]  # Ordered list of techniques to try
    escalation_level: int  # 1=light, 2=medium, 3=heavy
    issue_summary: str  # Human-readable summary
    confidence: float  # Diagnostic confidence (0.0-1.0)
    multi_issue: bool  # Multiple issues detected
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Issue Type → Root Cause Mapping
# ============================================================================

ISSUE_ROOT_CAUSES = {
    'missing_citations': 'llm_instruction_following',
    'invalid_citations': 'citation_formatting',
    'explicit_uncertainty': 'insufficient_context',
    'suspiciously_verbose': 'answer_format_mismatch',
    'too_short': 'incomplete_generation',
    'hallucination': 'context_grounding_failure',
    'low_relevance': 'query_context_mismatch',
    'low_quality': 'generation_quality',
    'citation_mismatch': 'source_mapping_error',
    'incomplete_answer': 'insufficient_context',
}

# ============================================================================
# Issue Type → Recommended Techniques Mapping
# ============================================================================

ISSUE_TO_TECHNIQUES = {
    'missing_citations': [
        'add_citation_emphasis',
        'strict_grounding',
        'enforce_format'
    ],
    'invalid_citations': [
        'fix_citation_format',
        'add_citation_emphasis'
    ],
    'explicit_uncertainty': [
        'expand_context',
        'rephrase_query',
        'strict_grounding'
    ],
    'hallucination': [
        'strict_grounding',
        'expand_context',
        'add_citation_emphasis'
    ],
    'incomplete_answer': [
        'expand_context',
        'rephrase_query',
        'decompose_query'
    ],
    'low_relevance': [
        'rephrase_query',
        'expand_context',
        'decompose_query'
    ],
    'suspiciously_verbose': [
        'enforce_format',
        'simplify_answer'
    ],
    'too_short': [
        'expand_context',
        'regenerate_with_emphasis'
    ],
    'low_quality': [
        'enforce_format',
        'regenerate_with_emphasis',
        'expand_context'
    ],
}

# ============================================================================
# Severity Weights (for prioritization)
# ============================================================================

SEVERITY_WEIGHTS = {
    'critical': 10,
    'high': 7,
    'medium': 4,
    'low': 1
}

# ============================================================================
# Diagnostic Engine Functions
# ============================================================================

def diagnose_validation_failure(
    validation_result: Dict[str, Any],
    query: str,
    chunks: List[Dict[str, Any]],
    retry_count: int = 0
) -> DiagnosisResult:
    """
    Analyze validation failure and recommend healing techniques.
    
    Args:
        validation_result: Output from quick_validate_answer() or full validation
        query: Original user query
        chunks: Retrieved context chunks
        retry_count: Current retry attempt (0-indexed)
        
    Returns:
        DiagnosisResult with recommended techniques and escalation level
        
    Example:
        >>> validation = quick_validate_answer(query, answer, chunks)
        >>> diagnosis = diagnose_validation_failure(validation, query, chunks, retry_count=0)
        >>> print(diagnosis.recommended_techniques)
        ['add_citation_emphasis', 'strict_grounding']
    """
    issues = validation_result.get('issues', [])
    
    if not issues:
        # No issues - shouldn't reach here
        return DiagnosisResult(
            root_cause='no_issues_detected',
            primary_issue_type='none',
            recommended_techniques=[],
            escalation_level=0,
            issue_summary='No issues detected',
            confidence=1.0,
            multi_issue=False
        )
    
    # Prioritize issues by severity
    prioritized_issues = _prioritize_issues(issues)
    primary_issue = prioritized_issues[0]
    
    # Determine root cause
    root_cause = ISSUE_ROOT_CAUSES.get(
        primary_issue['type'],
        'unknown_issue_type'
    )
    
    # Get recommended techniques based on issue type
    techniques = _get_techniques_for_issue(
        primary_issue,
        retry_count,
        len(prioritized_issues) > 1
    )
    
    # Determine escalation level based on retry count
    escalation_level = _determine_escalation_level(retry_count, primary_issue)
    
    # Build issue summary
    issue_summary = _build_issue_summary(prioritized_issues)
    
    # Calculate diagnostic confidence
    confidence = _calculate_diagnostic_confidence(
        primary_issue,
        len(chunks),
        retry_count
    )
    
    logger.info(
        f"Diagnosis: {root_cause} | "
        f"Primary: {primary_issue['type']} | "
        f"Techniques: {techniques[:2]} | "
        f"Escalation: L{escalation_level}"
    )
    
    return DiagnosisResult(
        root_cause=root_cause,
        primary_issue_type=primary_issue['type'],
        recommended_techniques=techniques,
        escalation_level=escalation_level,
        issue_summary=issue_summary,
        confidence=confidence,
        multi_issue=len(prioritized_issues) > 1,
        metadata={
            'total_issues': len(issues),
            'primary_severity': primary_issue.get('severity', 'unknown'),
            'retry_count': retry_count,
            'chunk_count': len(chunks)
        }
    )

def _prioritize_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prioritize issues by severity (critical > high > medium > low).
    
    Returns sorted list with most critical first.
    """
    def get_severity_weight(issue):
        severity = issue.get('severity', 'medium')
        return SEVERITY_WEIGHTS.get(severity, 1)
    
    return sorted(issues, key=get_severity_weight, reverse=True)

def _get_techniques_for_issue(
    issue: Dict[str, Any],
    retry_count: int,
    multi_issue: bool
) -> List[str]:
    """
    Get ordered list of healing techniques for an issue.
    
    Techniques are ordered by escalation level:
    - Retry 0: Use first (lightest) technique
    - Retry 1: Use second (medium) technique  
    - Retry 2+: Use third (heavy) technique
    """
    issue_type = issue['type']
    
    # Get base techniques for this issue type
    base_techniques = ISSUE_TO_TECHNIQUES.get(
        issue_type,
        ['regenerate_with_emphasis']  # Fallback
    )
    
    # Select technique based on retry count (escalation)
    if retry_count < len(base_techniques):
        # Return single technique for this escalation level
        return [base_techniques[retry_count]]
    else:
        # Max escalation - return last (heaviest) technique
        return [base_techniques[-1]]

def _determine_escalation_level(
    retry_count: int,
    issue: Dict[str, Any]
) -> int:
    """
    Determine escalation level (1=light, 2=medium, 3=heavy).
    
    Based on retry count:
    - Retry 0 → Level 1 (light touch)
    - Retry 1 → Level 2 (medium fix)
    - Retry 2+ → Level 3 (heavy fix)
    """
    severity = issue.get('severity', 'medium')
    
    # Base escalation from retry count
    base_level = min(retry_count + 1, 3)
    
    # Boost for critical issues
    if severity == 'critical':
        return min(base_level + 1, 3)
    
    return base_level

def _build_issue_summary(issues: List[Dict[str, Any]]) -> str:
    """Build human-readable issue summary."""
    if len(issues) == 1:
        issue = issues[0]
        return f"{issue['type']}: {issue.get('description', 'No description')}"
    else:
        types = [i['type'] for i in issues[:3]]
        return f"Multiple issues: {', '.join(types)}"

def _calculate_diagnostic_confidence(
    issue: Dict[str, Any],
    chunk_count: int,
    retry_count: int
) -> float:
    """
    Calculate confidence in diagnostic recommendation.
    
    Higher confidence for:
    - Clear, well-known issue types
    - Sufficient context chunks
    - Early retry attempts
    """
    base_confidence = 0.8
    
    # Reduce confidence for unknown issue types
    if issue['type'] not in ISSUE_TO_TECHNIQUES:
        base_confidence *= 0.7
    
    # Reduce confidence if insufficient context
    if chunk_count < 3:
        base_confidence *= 0.8
    
    # Reduce confidence on later retries
    if retry_count >= 2:
        base_confidence *= 0.6
    
    return min(max(base_confidence, 0.0), 1.0)

# ============================================================================
# Utility Functions
# ============================================================================

def classify_issue_severity(issue_type: str, validation_result: Dict) -> str:
    """
    Classify issue severity based on type and context.
    
    Args:
        issue_type: Type of issue (e.g., 'missing_citations')
        validation_result: Full validation result with scores
        
    Returns:
        'critical', 'high', 'medium', or 'low'
    """
    # Critical issues that block usage
    critical_issues = ['explicit_uncertainty', 'hallucination', 'invalid_citations']
    if issue_type in critical_issues:
        return 'critical'
    
    # High priority issues
    high_issues = ['missing_citations', 'incomplete_answer', 'low_relevance']
    if issue_type in high_issues:
        return 'high'
    
    # Check quality score
    quality_score = validation_result.get('quality_score', 1.0)
    if quality_score < 0.5:
        return 'high'
    
    # Medium priority
    medium_issues = ['suspiciously_verbose', 'low_quality']
    if issue_type in medium_issues:
        return 'medium'
    
    # Low priority (cosmetic)
    return 'low'

def recommend_healing_technique(
    issue_type: str,
    escalation_level: int = 1
) -> str:
    """
    Recommend a single healing technique for an issue.
    
    Args:
        issue_type: Type of issue
        escalation_level: 1=light, 2=medium, 3=heavy
        
    Returns:
        Technique name (e.g., 'add_citation_emphasis')
    """
    techniques = ISSUE_TO_TECHNIQUES.get(
        issue_type,
        ['regenerate_with_emphasis']
    )
    
    # Select technique based on escalation level
    idx = min(escalation_level - 1, len(techniques) - 1)
    return techniques[idx]

# ============================================================================
# Export
# ============================================================================

__all__ = [
    'DiagnosisResult',
    'diagnose_validation_failure',
    'classify_issue_severity',
    'recommend_healing_technique',
]
