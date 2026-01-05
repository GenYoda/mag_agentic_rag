"""
================================================================================
CREW TASKS - Validation & Healing Task Definitions
================================================================================

Purpose: Define reusable tasks for the medical RAG Crew

Tasks:
1. create_validation_task: Validate answer + decide action
2. create_healing_task: Execute healing strategy based on decision

These tasks are used by medical_rag_crew.py to orchestrate the 
EnhancedValidationAgent + EnhancedSelfHealerAgent workflow.

================================================================================
"""

from crewai import Task
from typing import Dict, List, Any


def create_validation_task(
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    agent=None,
    retry_count: int = 0
) -> Task:
    """
    Create a validation task for EnhancedValidationAgent.
    
    This task asks the ValidationAgent to:
    1. Validate the answer quality
    2. Detect issues (citations, hallucinations, etc.)
    3. DECIDE action: ACCEPT / RERANK / EXPAND / HEAL
    4. Provide diagnosis for healing
    
    Args:
        query: Original user question
        answer: Generated answer to validate
        chunks: Retrieved context chunks used for generation
        agent: Agent to assign (will be set by Crew if None)
        retry_count: Current retry attempt (for escalation)
        
    Returns:
        Task instance
        
    Example:
        >>> task = create_validation_task(
        ...     query="What medication was prescribed?",
        ...     answer="Lisinopril was prescribed [doc:0]",
        ...     chunks=[{...}]
        ... )
    """
    
    # Build context summary for task description
    chunk_summary = f"{len(chunks)} chunks"
    if chunks:
        first_chunk_source = chunks[0].get('metadata', {}).get('source', 'Unknown')
        chunk_summary += f" (e.g., {first_chunk_source})"
    
    # Build task description
    description = f"""
You are validating an answer and deciding the appropriate action.

═══════════════════════════════════════════════════════════════════════════
INPUT DATA
═══════════════════════════════════════════════════════════════════════════

Query: {query}

Answer to validate:
{answer}

Context: {chunk_summary}

Retry attempt: {retry_count + 1}

═══════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════

Step 1: VALIDATE the answer
- Check citation grounding (every claim must have [doc:X])
- Detect hallucinations (LLM-as-Judge)
- Check completeness (does it fully answer the query?)
- Check quality (clarity, structure)
- Assign quality_score (0.0-1.0)

Step 2: DECIDE the action
- ACCEPT: If quality_score ≥ 0.80 and no critical issues
- RERANK: If 0.60-0.80 and root_cause='poor_context_quality'
- EXPAND: If 0.60-0.80 and root_cause='incomplete_answer'
- HEAL: If <0.60 or critical issues (hallucination, missing citations)

Step 3: PROVIDE diagnosis (if action != ACCEPT)
- Identify root_cause
- Categorize primary_issue_type
- Recommend healing techniques
- Set escalation_level (1, 2, or 3 based on retry_count)

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════════════════

{{
  "is_valid": bool,
  "quality_score": 0.0-1.0,
  "confidence_score": 0.0-1.0,
  
  "decision": "ACCEPT" | "RERANK" | "EXPAND" | "HEAL",
  "decision_reasoning": "Clear explanation of why you chose this action",
  
  "has_hallucination": bool,
  "has_citations": bool,
  "citation_count": int,
  
  "issues": [
    {{
      "severity": "critical" | "high" | "medium" | "low",
      "category": "citations" | "hallucination" | "completeness" | "quality",
      "message": "Description of issue",
      "auto_fixable": bool
    }}
  ],
  
  "diagnosis": {{
    "root_cause": "poor_context_quality" | "incomplete_answer" | "llm_instruction_following" | etc.,
    "primary_issue_type": "missing_citations" | "hallucination" | etc.,
    "recommended_techniques": ["technique1", "technique2"],
    "escalation_level": {retry_count + 1},
    "issue_summary": "Brief summary of all issues"
  }},
  
  "llm_judge_reasoning": "LLM-as-Judge's reasoning"
}}

CRITICAL: Your decision must be DECISIVE and CLEAR. Don't hedge - choose the best action.
    """
    
    expected_output = (
        "A complete validation result with:\n"
        "1. quality_score (0.0-1.0)\n"
        "2. decision (ACCEPT/RERANK/EXPAND/HEAL)\n"
        "3. decision_reasoning (clear explanation)\n"
        "4. diagnosis (if action != ACCEPT)\n"
        "5. issues list with severity and category"
    )
    
    task = Task(
        description=description.strip(),
        expected_output=expected_output,
        agent=agent
    )
    
    return task


def create_healing_task(
    validation_result: Dict[str, Any],
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    agent=None
) -> Task:
    """
    Create a healing task for EnhancedSelfHealerAgent.
    
    This task asks the SelfHealerAgent to:
    1. Read the decision from ValidationAgent
    2. Execute the appropriate fixing strategy (RERANK/EXPAND/HEAL)
    3. Return improved answer
    
    Args:
        validation_result: Output from ValidationAgent with decision + diagnosis
        query: Original user question
        answer: Answer that needs fixing
        chunks: Original retrieved chunks
        agent: Agent to assign (will be set by Crew if None)
        
    Returns:
        Task instance
        
    Example:
        >>> task = create_healing_task(
        ...     validation_result={'decision': 'RERANK', ...},
        ...     query="What medication?",
        ...     answer="Lisinopril [doc:0]",
        ...     chunks=[...]
        ... )
    """
    
    # Extract key info from validation result
    decision = validation_result.get('decision', 'HEAL')
    decision_reasoning = validation_result.get('decision_reasoning', 'No reasoning provided')
    diagnosis = validation_result.get('diagnosis', {})
    issues = validation_result.get('issues', [])
    
    # Format issues for display
    issues_summary = "\n".join([
        f"  - [{issue.get('severity', 'unknown')}] {issue.get('category', 'unknown')}: {issue.get('message', '')}"
        for issue in issues[:5]  # Show first 5 issues
    ])
    if len(issues) > 5:
        issues_summary += f"\n  ... and {len(issues) - 5} more issues"
    
    # Build task description based on decision
    description = f"""
You are fixing an answer based on the ValidationAgent's decision.

═══════════════════════════════════════════════════════════════════════════
VALIDATION DECISION
═══════════════════════════════════════════════════════════════════════════

Decision: {decision}
Reasoning: {decision_reasoning}

Issues found:
{issues_summary if issues_summary else "  (No issues listed)"}

Diagnosis:
- Root cause: {diagnosis.get('root_cause', 'Unknown')}
- Primary issue: {diagnosis.get('primary_issue_type', 'Unknown')}
- Recommended techniques: {', '.join(diagnosis.get('recommended_techniques', ['None']))}
- Escalation level: {diagnosis.get('escalation_level', 1)}

═══════════════════════════════════════════════════════════════════════════
INPUT DATA
═══════════════════════════════════════════════════════════════════════════

Query: {query}

Answer to fix:
{answer}

Context: {len(chunks)} chunks available

═══════════════════════════════════════════════════════════════════════════
YOUR TASK: Execute {decision} strategy
═══════════════════════════════════════════════════════════════════════════
"""
    
    # Add decision-specific instructions
    if decision == "RERANK":
        description += """
🟡 RERANK STRATEGY:

1. Use rerank_smart_fallback_tool to rerank the chunks:
   reranked = rerank_smart_fallback_tool(
       query=query,
       chunks=original_chunks,
       top_k=3,
       method='crossencoder'
   )

2. Regenerate answer with top 3 reranked chunks:
   improved = generate_answer_with_context_tool(
       query=query,
       context_chunks=reranked,
       temperature=0.3
   )

3. Return the improved answer with reranked chunks

CRITICAL: Actually call the rerank tool - don't skip this step!
"""
    
    elif decision == "EXPAND":
        description += """
🟠 EXPAND STRATEGY:

1. Build expansion prompt:
   expansion_prompt = f'''
   Original question: {query}
   Previous answer (too brief): {answer}
   
   Task: Expand this answer with MORE DETAIL:
   - Add specific information from the context
   - Include additional relevant facts
   - Provide better explanations
   - Keep all citations and add more
   Target: 2-3x longer, stay focused on the question.
   '''

2. Regenerate with expansion prompt:
   expanded = generate_answer_with_context_tool(
       query=expansion_prompt,
       context_chunks=original_chunks,
       temperature=0.4
   )

3. Return the expanded answer

CRITICAL: Use the expansion prompt template above!
"""
    
    elif decision == "HEAL":
        recommended = diagnosis.get('recommended_techniques', ['regenerate_with_emphasis'])[0]
        description += f"""
🔴 HEAL STRATEGY:

Recommended technique: {recommended}

Apply the diagnostic technique based on the primary issue:

- If missing_citations → add_citation_emphasis
- If hallucination → strict_grounding
- If insufficient_context → expand_context (retrieve more chunks)
- If format_mismatch → enforce_format
- If poor_retrieval → rephrase_query
- If over_complex → simplify_answer
- If invalid_citations → fix_citation_format
- If unknown → regenerate_with_emphasis

Current escalation level: {diagnosis.get('escalation_level', 1)}
(Higher levels = more aggressive techniques)

CRITICAL: Apply the SPECIFIC technique recommended, not generic retry!
"""
    
    description += """

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════════════════

{{
  "success": bool,
  "answer": "The improved answer with proper citations",
  "technique_applied": "rerank" | "expand" | "add_citation_emphasis" | etc.,
  "chunks": [...],  // Chunks used (may be reranked)
  "metadata": {{
    "decision": "{decision}",
    "escalation_level": {diagnosis.get('escalation_level', 1)},
    "improvements_made": ["list", "of", "improvements"],
    "tools_used": ["list", "of", "tools"]
  }}
}}

Execute the strategy NOW and return the improved answer.
    """
    
    expected_output = (
        f"An improved answer using {decision} strategy with:\n"
        "1. answer (improved text with citations)\n"
        "2. technique_applied (name of strategy used)\n"
        "3. chunks (chunks used, may be reranked)\n"
        "4. metadata (improvements made, tools used)"
    )
    
    task = Task(
        description=description.strip(),
        expected_output=expected_output,
        agent=agent
    )
    
    return task


# Export for easy imports
__all__ = [
    'create_validation_task',
    'create_healing_task'
]
