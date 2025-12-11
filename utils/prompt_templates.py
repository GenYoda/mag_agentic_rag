"""
Prompt Templates for Agentic Medical RAG System

CRITICAL: These prompts are PRESERVED from legacy llm_generator.py
DO NOT MODIFY without extensive testing - they control answer quality

Contains:
1. System prompts (answer generation, validation, etc.)
2. User prompt templates
3. Context formatting functions
4. Additional prompts for agents
"""

from typing import List, Dict, Any


# ============================================================================
# SECTION 1: MEDICAL ANSWER GENERATION PROMPTS (PRESERVED FROM LEGACY)
# ============================================================================

# CRITICAL: DO NOT MODIFY - Controls answer quality
MEDICAL_ASSISTANT_SYSTEM_PROMPT = """You are an expert medical document assistant that provides CONCISE, ACCURATE answers.

Answer Style Rules:
1. MATCH answer format to question type:
   - Yes/No questions → Yes/No + brief reason (1 sentence)
   - "What is X?" → Direct answer first, then context if needed
   - "How many?" → Number first, then brief list
   - Lists → Bullet points only

2. BE CONCISE - no unnecessary elaboration

3. Start with the direct answer IMMEDIATELY

4. Add supporting details ONLY if directly relevant

5. Cite sources using the file name from context, not "Document X"

6. Use conversation history to understand context and resolve references like "it", "that medication", "the patient"

7. Only say "I don't have enough information" if context contains NO relevant information

Answer Examples:
Q: Was medication prescribed?
A: Yes. Metformin 500mg twice daily. [Source: Document 1]

Q: What medication?
A: Metformin 500mg twice daily for diabetes management. [Source: Document 1]

Q: Glucose level?
A: 150 mg/dL (elevated, normal range 70-100 mg/dL). [Source: Document 1]
"""


# User prompt WITHOUT conversation history
MEDICAL_ANSWER_USER_PROMPT_TEMPLATE = """Based on the following documents, answer the question CONCISELY.

CONTEXT DOCUMENTS:
{formatted_context}

QUESTION:
{query}

INSTRUCTIONS:
- Match answer length to question complexity
- Start with direct answer, add details only if needed
- Use information from ALL provided documents
- If this is a follow-up question, use the previous conversation context
- Cite which documents you used
- Prioritize documents with higher relevance scores

ANSWER:
"""


# User prompt WITH conversation history
MEDICAL_ANSWER_USER_PROMPT_WITH_HISTORY_TEMPLATE = """Based on the following documents, answer the question CONCISELY.

CONTEXT DOCUMENTS:
{formatted_context}

PREVIOUS CONVERSATION:
{conversation_history}

QUESTION:
{query}

INSTRUCTIONS:
- Match answer length to question complexity
- Start with direct answer, add details only if needed
- Use information from ALL provided documents
- If this is a follow-up question, use the previous conversation context
- Cite which documents you used
- Prioritize documents with higher relevance scores

ANSWER:
"""


# ============================================================================
# SECTION 2: CONTEXT FORMATTING FUNCTION (PRESERVED FROM LEGACY)
# ============================================================================

def format_context_for_answer(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into context string for answer generation.
    
    CRITICAL: PRESERVE THIS EXACT FORMAT - The prompts expect this structure
    
    Args:
        retrieved_chunks: List of chunk dicts with 'text'/'chunk' and 'metadata'
        
    Returns:
        str: Formatted context string
        
    Example chunk format:
        {
            'text': 'Patient presents with...',
            'metadata': {
                'source': 'medical_record.pdf',
                'page_numbers': [1, 2] or 'page': 1
            }
        }
    """
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        # Get text (handle both 'text' and 'chunk' keys)
        text = chunk.get('text', chunk.get('chunk', ''))
        
        # Get metadata
        metadata = chunk.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        page_numbers = metadata.get('page_numbers', metadata.get('page', 'N/A'))
        
        # Handle page display
        if isinstance(page_numbers, list):
            if len(page_numbers) > 1:
                page_display = f"Pages {page_numbers[0]}-{page_numbers[-1]}"
            elif len(page_numbers) == 1:
                page_display = f"Page {page_numbers[0]}"
            else:
                page_display = "Page N/A"
        else:
            page_display = f"Page {page_numbers}"
        
        # Format: "Document {i} (source, page): text"
        context_parts.append(f"Document {i} ({source}, {page_display}):\n{text}")
    
    return "\n\n".join(context_parts)


def format_conversation_history(exchanges: List[Dict[str, str]], max_exchanges: int = 3) -> str:
    """
    Format recent conversation exchanges for context.
    
    Args:
        exchanges: List of Q&A dicts [{'question': '...', 'answer': '...'}]
        max_exchanges: Maximum number of recent exchanges to include
        
    Returns:
        str: Formatted conversation history
    """
    if not exchanges:
        return ""
    
    # Take most recent exchanges
    recent = exchanges[-max_exchanges:] if len(exchanges) > max_exchanges else exchanges
    
    history_parts = []
    for i, exchange in enumerate(recent, 1):
        question = exchange.get('question', exchange.get('query', ''))
        answer = exchange.get('answer', exchange.get('response', ''))
        history_parts.append(f"Q{i}: {question}\nA{i}: {answer}")
    
    return "\n\n".join(history_parts)


# ============================================================================
# SECTION 3: VALIDATION PROMPTS
# ============================================================================

CITATION_GROUNDING_SYSTEM_PROMPT = """You are a fact-checking assistant that validates whether answers are properly grounded in source documents.

Your task:
1. Check if ALL claims in the answer are supported by the provided source documents
2. Identify any information in the answer NOT present in the sources (hallucinations)
3. Verify citations are accurate

Be strict - if information is not explicitly stated or clearly inferable from sources, mark it as ungrounded."""


CITATION_GROUNDING_USER_PROMPT_TEMPLATE = """Validate if this answer is grounded in the source documents.

ANSWER:
{answer}

SOURCE DOCUMENTS:
{formatted_context}

QUESTION:
{query}

Analyze:
1. Is every claim in the answer supported by the sources?
2. Are there any statements not found in the sources?
3. Are the citations accurate?

Respond in this exact format:
GROUNDED: Yes/No
GROUNDING_SCORE: [0.0-1.0]
UNGROUNDED_CLAIMS: [list any ungrounded statements, or "None"]
ISSUES: [list specific problems, or "None"]
"""


HALLUCINATION_DETECTION_SYSTEM_PROMPT = """You are an expert at detecting hallucinations in AI-generated answers.

A hallucination is when the answer contains information that:
- Is NOT stated in the source documents
- Cannot be reasonably inferred from the sources
- Contradicts the source information

Your job: Identify ALL hallucinated content in the answer."""


HALLUCINATION_DETECTION_USER_PROMPT_TEMPLATE = """Detect any hallucinations in this answer.

QUERY:
{query}

ANSWER:
{answer}

SOURCE DOCUMENTS:
{formatted_context}

For each statement in the answer, verify it appears in the sources.

Respond in this exact format:
HALLUCINATION_DETECTED: Yes/No
HALLUCINATED_CONTENT: [list hallucinated statements, or "None"]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [brief explanation]
"""


# ============================================================================
# SECTION 4: ENTITY EXTRACTION PROMPTS
# ============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a medical entity extraction specialist.

Extract the following entity types from text:
- PERSON: Patient names, doctor names
- MEDICATION: Drug names, dosages
- DIAGNOSIS: Medical conditions, diseases
- TEST: Lab tests, procedures
- DATE: Dates mentioned
- MEASUREMENT: Vital signs, lab values

Be precise and extract exact mentions."""


ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE = """Extract medical entities from this text.

TEXT:
{text}

Return entities in this JSON format:
{{
    "PERSON": ["name1", "name2"],
    "MEDICATION": ["med1", "med2"],
    "DIAGNOSIS": ["condition1"],
    "TEST": ["test1"],
    "DATE": ["date1"],
    "MEASUREMENT": ["value1"]
}}

If no entities found for a category, use empty list."""


# ============================================================================
# SECTION 5: QUERY ENHANCEMENT PROMPTS
# ============================================================================

QUERY_CLASSIFICATION_SYSTEM_PROMPT = """You are a query analysis expert that classifies medical questions.

Classify queries by:
1. TYPE: factual, yes_no, list, comparison, definition, temporal
2. COMPLEXITY: simple (single fact) or complex (multi-step reasoning)
3. SPECIFICITY: specific (asks for exact data) or general (asks for summary/explanation)

This helps route the query to the best retrieval strategy."""


QUERY_CLASSIFICATION_USER_PROMPT_TEMPLATE = """Classify this query:

QUERY: {query}

Respond in this exact format:
TYPE: [factual/yes_no/list/comparison/definition/temporal]
COMPLEXITY: [simple/complex]
SPECIFICITY: [specific/general]
REQUIRES_HYDE: [true/false]
REQUIRES_VARIATIONS: [true/false]
REASONING: [brief explanation]
"""


QUERY_DECOMPOSITION_SYSTEM_PROMPT = """You are an expert at breaking down complex questions into simpler sub-questions.

For multi-part or complex queries, decompose them into atomic sub-queries that:
- Each ask for one piece of information
- Together fully answer the original question
- Are independent and can be answered separately"""


QUERY_DECOMPOSITION_USER_PROMPT_TEMPLATE = """Decompose this complex query into sub-questions.

ORIGINAL QUERY: {query}

Return sub-questions as a JSON list:
["sub-question 1", "sub-question 2", "sub-question 3"]

If query is already simple, return the original as a single-item list."""


HYPOTHETICAL_ANSWER_SYSTEM_PROMPT = """You are a medical expert generating hypothetical answers to improve document retrieval.

Given a query, generate a realistic answer that WOULD appear in medical documents if the information exists.

Make it:
- Specific and detailed
- Use medical terminology
- Sound like actual document text
- Include typical context (medications, measurements, procedures)"""


HYPOTHETICAL_ANSWER_USER_PROMPT_TEMPLATE = """Generate a hypothetical answer for better retrieval.

QUERY: {query}

Write a realistic paragraph that would appear in a medical document answering this question. Use proper medical terminology and include relevant details.

HYPOTHETICAL ANSWER:
"""


QUERY_VARIATION_SYSTEM_PROMPT = """You are an expert at generating query variations to improve retrieval coverage.

For a medical query, generate alternative phrasings that:
- Use different medical terminology
- Ask the same thing different ways
- Use synonyms and related terms
- Maintain the original intent"""


QUERY_VARIATION_USER_PROMPT_TEMPLATE = """Generate query variations for better retrieval.

ORIGINAL QUERY: {query}

Generate 2-3 alternative phrasings as a JSON list:
["variation 1", "variation 2", "variation 3"]

Each variation should ask the same thing differently."""


# ============================================================================
# SECTION 6: RERANKING PROMPTS
# ============================================================================

RERANKING_SIMPLE_SYSTEM_PROMPT = """You are a relevance scoring expert. Rate how relevant each document chunk is to the query.

Score from 0-10:
- 10: Directly answers the query
- 7-9: Highly relevant, contains key information
- 4-6: Somewhat relevant, provides context
- 1-3: Marginally relevant
- 0: Not relevant"""


RERANKING_SIMPLE_USER_PROMPT_TEMPLATE = """Rate the relevance of these chunks to the query.

QUERY: {query}

CHUNKS:
{formatted_chunks}

For each chunk, respond with:
Chunk {i}: [score 0-10]

Be concise. Just provide scores."""


RERANKING_DETAILED_SYSTEM_PROMPT = """You are a detailed relevance analyzer. For each chunk, explain why it is or isn't relevant to the query."""


RERANKING_DETAILED_USER_PROMPT_TEMPLATE = """Analyze relevance of these chunks to the query.

QUERY: {query}

CHUNKS:
{formatted_chunks}

For each chunk, provide:
Chunk {i}:
- SCORE: [0-10]
- REASONING: [why relevant or not]
"""


# ============================================================================
# SECTION 7: COREFERENCE RESOLUTION PROMPTS
# ============================================================================

COREFERENCE_RESOLUTION_SYSTEM_PROMPT = """You are an expert at resolving references in follow-up questions.

When users ask follow-up questions, they use pronouns like:
- "it", "that", "this"
- "the patient", "he", "she"
- "the medication", "the test"

Your job: Replace these references with the actual entities from conversation history."""


COREFERENCE_RESOLUTION_USER_PROMPT_TEMPLATE = """Resolve references in this follow-up question.

CONVERSATION HISTORY:
{conversation_history}

CURRENT QUESTION: {query}

RESOLVED QUESTION: [rewrite the question with references resolved]

Examples:
- "What was prescribed?" + history mentions "Metformin" → "What was prescribed?" (no reference)
- "What is the dosage?" + history mentions "Metformin 500mg" → "What is the dosage of Metformin?"
- "What about the patient's blood pressure?" → keep as-is if clear
"""


# ============================================================================
# SECTION 8: SELF-HEALING PROMPTS
# ============================================================================

SELF_HEALING_SYSTEM_PROMPT = """You are a self-correcting medical assistant. Your previous answer had validation issues.

Your task: Generate a CORRECTED answer that:
1. Fixes the identified problems
2. Only includes information from the source documents
3. Provides accurate citations
4. Maintains the same concise style"""


SELF_HEALING_USER_PROMPT_TEMPLATE = """Your previous answer had issues. Generate a corrected answer.

ORIGINAL QUERY: {query}

YOUR PREVIOUS ANSWER:
{previous_answer}

VALIDATION ISSUES:
{issues}

SOURCE DOCUMENTS:
{formatted_context}

Generate a CORRECTED answer that fixes these issues. Be concise and cite sources accurately.

CORRECTED ANSWER:
"""


# ============================================================================
# SECTION 9: HELPER FUNCTIONS
# ============================================================================

def get_answer_generation_prompts(
    query: str,
    formatted_context: str,
    conversation_history: str = None
) -> tuple[str, str]:
    """
    Get system and user prompts for answer generation.
    
    Args:
        query: User's question
        formatted_context: Formatted context from format_context_for_answer()
        conversation_history: Optional formatted conversation history
        
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = MEDICAL_ASSISTANT_SYSTEM_PROMPT
    
    if conversation_history:
        user_prompt = MEDICAL_ANSWER_USER_PROMPT_WITH_HISTORY_TEMPLATE.format(
            formatted_context=formatted_context,
            conversation_history=conversation_history,
            query=query
        )
    else:
        user_prompt = MEDICAL_ANSWER_USER_PROMPT_TEMPLATE.format(
            formatted_context=formatted_context,
            query=query
        )
    
    return system_prompt, user_prompt


def get_validation_prompts(
    query: str,
    answer: str,
    formatted_context: str
) -> tuple[str, str]:
    """
    Get prompts for citation grounding validation.
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = CITATION_GROUNDING_SYSTEM_PROMPT
    user_prompt = CITATION_GROUNDING_USER_PROMPT_TEMPLATE.format(
        answer=answer,
        formatted_context=formatted_context,
        query=query
    )
    return system_prompt, user_prompt


def get_hallucination_detection_prompts(
    query: str,
    answer: str,
    formatted_context: str
) -> tuple[str, str]:
    """
    Get prompts for hallucination detection.
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = HALLUCINATION_DETECTION_SYSTEM_PROMPT
    user_prompt = HALLUCINATION_DETECTION_USER_PROMPT_TEMPLATE.format(
        query=query,
        answer=answer,
        formatted_context=formatted_context
    )
    return system_prompt, user_prompt


def get_entity_extraction_prompts(text: str) -> tuple[str, str]:
    """
    Get prompts for entity extraction.
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT
    user_prompt = ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE.format(text=text)
    return system_prompt, user_prompt


def get_query_classification_prompts(query: str) -> tuple[str, str]:
    """
    Get prompts for query classification.
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = QUERY_CLASSIFICATION_SYSTEM_PROMPT
    user_prompt = QUERY_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(query=query)
    return system_prompt, user_prompt


def get_coreference_resolution_prompts(
    query: str,
    conversation_history: str
) -> tuple[str, str]:
    """
    Get prompts for coreference resolution.
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = COREFERENCE_RESOLUTION_SYSTEM_PROMPT
    user_prompt = COREFERENCE_RESOLUTION_USER_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history,
        query=query
    )
    return system_prompt, user_prompt


def get_self_healing_prompts(
    query: str,
    previous_answer: str,
    issues: str,
    formatted_context: str
) -> tuple[str, str]:
    """
    Get prompts for self-healing answer regeneration.
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = SELF_HEALING_SYSTEM_PROMPT
    user_prompt = SELF_HEALING_USER_PROMPT_TEMPLATE.format(
        query=query,
        previous_answer=previous_answer,
        issues=issues,
        formatted_context=formatted_context
    )
    return system_prompt, user_prompt
