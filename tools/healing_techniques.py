

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Technique 1: Add Citation Emphasis
# ============================================================================

def add_citation_emphasis(
    query: str,
    chunks: List[Dict[str, Any]],
    original_answer: str
) -> Dict[str, Any]:
    """
    Add strong emphasis on citation requirements.
    
    Use for: missing_citations
    
    Returns enhanced prompt with citation emphasis.
    """
    enhanced_prompt = f"""
CRITICAL INSTRUCTION: You MUST cite EVERY factual claim with [doc:X] format.

Original query: {query}

CITATION RULES:
1. Every sentence with facts needs [doc:X]
2. X is the chunk index (0-{len(chunks)-1})
3. Multiple claims in one sentence → multiple citations
4. Example: "Patient has diabetes [doc:1] and takes Metformin [doc:2]."

BAD (no citations): "Patient has diabetes and takes Metformin."
GOOD: "Patient has diabetes [doc:1] and takes Metformin [doc:2]."

Now answer the query with MANDATORY citations:
"""
    
    logger.info("Technique: add_citation_emphasis")
    
    return {
        'technique': 'add_citation_emphasis',
        'enhanced_prompt': enhanced_prompt,
        'temperature': 0.1,  # Very low for factual accuracy
        'emphasis': 'citations',
        'metadata': {
            'original_had_citations': '[doc:' in original_answer,
            'chunk_count': len(chunks)
        }
    }

# ============================================================================
# Technique 2: Strict Grounding
# ============================================================================

def strict_grounding(
    query: str,
    chunks: List[Dict[str, Any]],
    original_answer: str
) -> Dict[str, Any]:
    """
    Emphasize that ONLY context information should be used.
    
    Use for: hallucination, explicit_uncertainty
    
    Returns enhanced prompt with strict grounding instructions.
    """
    enhanced_prompt = f"""
STRICT GROUNDING REQUIREMENT: Use ONLY information EXPLICITLY stated in the context.

Original query: {query}

GROUNDING RULES:
1. DO NOT add information not in the context
2. DO NOT use general medical knowledge
3. If context doesn't have the answer, say: "The available documents do not contain information about [topic]."
4. Quote or paraphrase directly from context
5. Cite sources with [doc:X]

IMPORTANT: If you're unsure, say so. Don't guess or invent.

Now answer using ONLY the provided context:
"""
    
    logger.info("Technique: strict_grounding")
    
    return {
        'technique': 'strict_grounding',
        'enhanced_prompt': enhanced_prompt,
        'temperature': 0.05,  # Extremely low - no creativity
        'emphasis': 'grounding',
        'metadata': {
            'chunk_count': len(chunks)
        }
    }

# ============================================================================
# Technique 3: Expand Context
# ============================================================================

def expand_context(
    query: str,
    current_chunks: List[Dict[str, Any]],
    current_top_k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve more context chunks.
    
    Use for: incomplete_answer, explicit_uncertainty, low_relevance
    
    Returns parameters for retrieving more chunks.
    """
    new_top_k = min(current_top_k + 3, 10)  # Add 3 more, max 10
    
    logger.info(f"Technique: expand_context (top_k: {current_top_k} → {new_top_k})")
    
    return {
        'technique': 'expand_context',
        'action': 'retrieve_more',
        'new_top_k': new_top_k,
        'rerank': True,  # Rerank expanded set
        'metadata': {
            'original_top_k': current_top_k,
            'new_top_k': new_top_k,
            'chunks_added': new_top_k - current_top_k
        }
    }

# ============================================================================
# Technique 4: Enforce Format
# ============================================================================

def enforce_format(
    query: str,
    chunks: List[Dict[str, Any]],
    expected_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Match answer format to question type.
    
    Use for: suspiciously_verbose, low_quality, format mismatch
    
    Returns format-specific instructions.
    """
    # Detect question type
    query_lower = query.lower()
    
    if not expected_format:
        # Auto-detect format
        if any(q in query_lower for q in ['select', 'choose', 'option', 'a)', 'b)', 'c)']):
            expected_format = 'multiple_choice'
        elif any(q in query_lower for q in ['list', 'enumerate', 'bullet']):
            expected_format = 'bullet_list'
        elif any(q in query_lower for q in ['true or false', 'yes or no']):
            expected_format = 'boolean'
        elif any(q in query_lower for q in ['is it', 'is the', 'did the', 'was the']):
            expected_format = 'short_answer'
        else:
            expected_format = 'paragraph'
    
    # Format-specific instructions
    format_instructions = {
        'multiple_choice': """
FORMAT REQUIREMENT: Multiple Choice Answer

Provide answer in this exact format:
Answer: [Letter]
Explanation: [Brief reasoning with citation]

Example:
Answer: B
Explanation: The document states the patient was prescribed Lisinopril [doc:1].
""",
        'bullet_list': """
FORMAT REQUIREMENT: Bullet List

Provide answer as bullet points:
- Point 1 [doc:X]
- Point 2 [doc:Y]
- Point 3 [doc:Z]
""",
        'boolean': """
FORMAT REQUIREMENT: True/False or Yes/No

Provide answer in this format:
Answer: [True/False or Yes/No]
Reasoning: [Brief explanation with citation]
""",
        'short_answer': """
FORMAT REQUIREMENT: Short Answer

Provide a concise 1-2 sentence answer with citations.
Direct and factual, no extra details.
""",
        'paragraph': """
FORMAT REQUIREMENT: Paragraph Answer

Provide a well-structured 2-4 sentence answer with:
1. Direct answer first
2. Supporting details
3. Proper citations [doc:X]
"""
    }
    
    enhanced_prompt = f"""
{format_instructions.get(expected_format, format_instructions['paragraph'])}

Original query: {query}

Now answer following the format requirement:
"""
    
    logger.info(f"Technique: enforce_format (format: {expected_format})")
    
    return {
        'technique': 'enforce_format',
        'enhanced_prompt': enhanced_prompt,
        'expected_format': expected_format,
        'temperature': 0.2,
        'metadata': {
            'detected_format': expected_format
        }
    }

# ============================================================================
# Technique 5: Rephrase Query
# ============================================================================

def rephrase_query(
    query: str,
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Rephrase query for better retrieval.
    
    Use for: low_relevance, explicit_uncertainty, low_confidence
    
    Returns rephrased query variants.
    """
    # Generate query variations (simple rule-based)
    variations = [query]  # Original
    
    # Add question words if missing
    query_lower = query.lower()
    if not any(q in query_lower for q in ['what', 'who', 'when', 'where', 'why', 'how']):
        if 'medication' in query_lower or 'drug' in query_lower:
            variations.append(f"What medications are mentioned regarding: {query}")
        elif 'diagnosis' in query_lower or 'condition' in query_lower:
            variations.append(f"What diagnosis or condition is described: {query}")
        else:
            variations.append(f"What information is available about: {query}")
    
    # Expand abbreviations (simple examples)
    if 'meds' in query_lower:
        variations.append(query.replace('meds', 'medications'))
    if 'dx' in query_lower:
        variations.append(query.replace('dx', 'diagnosis'))
    if 'tx' in query_lower:
        variations.append(query.replace('tx', 'treatment'))
    
    # Add medical context
    if len(query.split()) < 5:  # Short query
        variations.append(f"patient medical history: {query}")
    
    logger.info(f"Technique: rephrase_query ({len(variations)} variations)")
    
    return {
        'technique': 'rephrase_query',
        'action': 'retrieve_with_variations',
        'query_variations': variations,
        'use_best_variant': True,
        'metadata': {
            'original_query': query,
            'variation_count': len(variations)
        }
    }

# ============================================================================
# Technique 6: Simplify Answer
# ============================================================================

def simplify_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    original_answer: str
) -> Dict[str, Any]:
    """
    Request simpler, more concise answer.
    
    Use for: suspiciously_verbose, over-complex
    
    Returns simplified generation instructions.
    """
    enhanced_prompt = f"""
SIMPLIFICATION REQUIREMENT: Provide a clear, concise answer.

Original query: {query}

SIMPLIFICATION RULES:
1. Use simple language
2. Maximum 3-4 sentences
3. Avoid unnecessary medical jargon
4. Still cite sources [doc:X]
5. Answer the question directly

Example:
BAD: "The patient's hypertensive condition was managed through the administration of an angiotensin-converting enzyme inhibitor..."
GOOD: "The patient's high blood pressure was treated with Lisinopril [doc:1]."

Now provide a simplified answer:
"""
    
    logger.info("Technique: simplify_answer")
    
    return {
        'technique': 'simplify_answer',
        'enhanced_prompt': enhanced_prompt,
        'temperature': 0.15,
        'max_tokens': 200,  # Force brevity
        'emphasis': 'simplicity',
        'metadata': {
            'original_length': len(original_answer.split())
        }
    }

# ============================================================================
# Technique 7: Fix Citation Format
# ============================================================================

def fix_citation_format(
    query: str,
    chunks: List[Dict[str, Any]],
    original_answer: str
) -> Dict[str, Any]:
    """
    Fix broken citation syntax.
    
    Use for: invalid_citations, citation_mismatch
    
    Returns corrected citation instructions.
    """
    enhanced_prompt = f"""
CITATION FORMAT CORRECTION:

CORRECT FORMAT: [doc:X] where X is 0 to {len(chunks)-1}

WRONG formats (DO NOT USE):
- [doc:] (missing number)
- [docX] (missing colon)
- [doc X] (space instead of colon)
- [source:X] (wrong tag)

CORRECT examples:
- "Patient has diabetes [doc:0]."
- "Prescribed Metformin [doc:1] and Lisinopril [doc:2]."

Original query: {query}

Now answer with CORRECT citation format [doc:X]:
"""
    
    logger.info("Technique: fix_citation_format")
    
    return {
        'technique': 'fix_citation_format',
        'enhanced_prompt': enhanced_prompt,
        'temperature': 0.1,
        'emphasis': 'citation_format',
        'metadata': {
            'chunk_count': len(chunks),
            'valid_range': f"0-{len(chunks)-1}"
        }
    }

# ============================================================================
# Technique 8: Regenerate With Emphasis (Fallback)
# ============================================================================

def regenerate_with_emphasis(
    query: str,
    chunks: List[Dict[str, Any]],
    original_answer: str,
    emphasis_points: List[str] = None
) -> Dict[str, Any]:
    """
    General regeneration with custom emphasis points.
    
    Use for: Generic retry, unknown issue types
    
    Returns general enhancement instructions.
    """
    if not emphasis_points:
        emphasis_points = [
            "Cite every factual claim with [doc:X]",
            "Use ONLY information from provided context",
            "Answer format should match question type",
            "Be concise but complete"
        ]
    
    emphasis_text = "\n".join(f"- {point}" for point in emphasis_points)
    
    enhanced_prompt = f"""
REGENERATION WITH EMPHASIS:

Key requirements:
{emphasis_text}

Original query: {query}

Now generate an improved answer:
"""
    
    logger.info("Technique: regenerate_with_emphasis (generic)")
    
    return {
        'technique': 'regenerate_with_emphasis',
        'enhanced_prompt': enhanced_prompt,
        'temperature': 0.15,
        'emphasis': 'general',
        'metadata': {
            'emphasis_count': len(emphasis_points)
        }
    }

# ============================================================================
# Technique Selector
# ============================================================================

def get_technique_for_issue(
    issue_type: str,
    escalation_level: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Get healing technique function result for an issue type.
    
    Args:
        issue_type: Type of issue (e.g., 'missing_citations')
        escalation_level: 1=light, 2=medium, 3=heavy
        **kwargs: Additional parameters (query, chunks, original_answer)
        
    Returns:
        Technique result dict with enhanced_prompt or action instructions
    """
    query = kwargs.get('query', '')
    chunks = kwargs.get('chunks', [])
    original_answer = kwargs.get('original_answer', '')
    
    # Map issue type to technique (with escalation)
    technique_map = {
        'missing_citations': [
            add_citation_emphasis,
            strict_grounding,
            fix_citation_format
        ],
        'invalid_citations': [
            fix_citation_format,
            add_citation_emphasis
        ],
        'hallucination': [
            strict_grounding,
            expand_context,
            add_citation_emphasis
        ],
        'explicit_uncertainty': [
            expand_context,
            rephrase_query,
            strict_grounding
        ],
        'incomplete_answer': [
            expand_context,
            rephrase_query,
            regenerate_with_emphasis
        ],
        'low_relevance': [
            rephrase_query,
            expand_context,
            regenerate_with_emphasis
        ],
        'suspiciously_verbose': [
            simplify_answer,
            enforce_format
        ],
        'low_quality': [
            enforce_format,
            regenerate_with_emphasis
        ],
    }
    
    # Get techniques for this issue
    techniques = technique_map.get(
        issue_type,
        [regenerate_with_emphasis]  # Fallback
    )
    
    # Select based on escalation level
    idx = min(escalation_level - 1, len(techniques) - 1)
    technique_func = techniques[idx]
    
    # Execute technique
    try:
        if technique_func in [expand_context, rephrase_query]:
            # These don't need original_answer
            result = technique_func(query, chunks)
        else:
            result = technique_func(query, chunks, original_answer)
        
        result['escalation_level'] = escalation_level
        return result
        
    except Exception as e:
        logger.error(f"Technique execution failed: {e}")
        # Fallback
        return regenerate_with_emphasis(query, chunks, original_answer)

# ============================================================================
# Export
# ============================================================================

__all__ = [
    'add_citation_emphasis',
    'strict_grounding',
    'expand_context',
    'enforce_format',
    'rephrase_query',
    'simplify_answer',
    'fix_citation_format',
    'regenerate_with_emphasis',
    'get_technique_for_issue',
]
