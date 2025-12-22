"""
================================================================================
SELF-HEALER AGENT - Agent 10/10 (NEW)
================================================================================
Purpose: Analyze validation failures and autonomously fix answer quality issues
Responsibilities:
- Analyze validation results from Validation Agent
- Make autonomous decisions on corrective actions
- Select appropriate tools based on issue categories
- Retry answer generation with improvements (max 2 attempts)
- Track best attempt across retries
- Return highest scoring answer if all retries fail

Tools Used:
- generate_answer_with_context_tool: Regenerate with emphasis
- search_tool: Retrieve more context
- decompose_query_tool: Break down complex queries
- enhance_query_tool: Rephrase query
- rerank_smart_fallback_tool: Rerank with different method

Integration:
- Works after Validation Agent detects issues
- Loops back through: Retrieval → Answer → Validation
- Max 2 retries before returning best attempt
- Configurable via SELF_HEAL_MAX_RETRIES in settings.py
================================================================================
"""

from crewai import Agent, LLM
from config.settings import (
    azure_settings,
    SELF_HEAL_MAX_RETRIES,
    SELF_HEAL_MIN_QUALITY_SCORE,
    SELF_HEAL_STRATEGIES,
    SELF_HEAL_TRACK_BEST_ATTEMPT
)
from tools.answer_generation_tools import (
    generate_answer_with_context_tool,
)
from tools.retrieval_tools import (
    search_tool,
)
from tools.query_tools import (
    decompose_query_tool,
    enhance_query_tool,
)
from tools.reranking_tools import (
    rerank_smart_fallback_tool,
)


def create_self_healer_agent(verbose: bool = True) -> Agent:
    """
    Create the Self-Healer Agent.
    
    The Self-Healer Agent analyzes validation failures and autonomously
    determines corrective actions to improve answer quality. It has access
    to the full toolkit and makes smart decisions about which tools to use.
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for self-healing
        
    Example:
        >>> self_healer_agent = create_self_healer_agent(verbose=True)
        >>> # Use in a Crew after validation fails
    """
    # Configure Azure OpenAI LLM
    llm = LLM(
        model=f"azure/{azure_settings.azure_openai_chat_deployment}",
        base_url=(
            f"{azure_settings.azure_openai_endpoint}"
            f"openai/deployments/{azure_settings.azure_openai_chat_deployment}"
            f"/chat/completions?api-version={azure_settings.azure_openai_api_version}"
        ),
        api_key=azure_settings.azure_openai_key,
        temperature=0.3,  # Slightly creative for problem-solving
    )
    
    agent = Agent(
        role="Answer Quality Improvement Specialist",
        
        goal=(
            f"Analyze validation failures and autonomously fix answer quality issues. "
            f"Make smart decisions about which corrective actions to take. "
            f"Maximum {SELF_HEAL_MAX_RETRIES} retry attempts. "
            f"Target quality score: {SELF_HEAL_MIN_QUALITY_SCORE}."
        ),
        
        backstory=(
            "You are an expert problem-solver specialized in improving RAG system outputs. "
            "When the Validation Agent detects issues with a generated answer, you analyze "
            "the validation report and autonomously determine the best corrective action.\n\n"
            
            f"Configuration:\n"
            f"- Maximum retries: {SELF_HEAL_MAX_RETRIES}\n"
            f"- Target quality score: {SELF_HEAL_MIN_QUALITY_SCORE}\n"
            f"- Track best attempt: {SELF_HEAL_TRACK_BEST_ATTEMPT}\n"
            f"- Available strategies: {', '.join(SELF_HEAL_STRATEGIES)}\n\n"
            
            "DECISION TREE - Issue Category → Corrective Action:\n\n"
            
            "1. MISSING CITATIONS (category: 'citation'):\n"
            "   Decision: Regenerate with citation emphasis\n"
            "   Action:\n"
            "   - Use generate_answer_with_context_tool\n"
            "   - Add instruction: 'CRITICAL: Cite every factual claim with [doc:X]'\n"
            "   - Use same context but emphasize citation requirement\n"
            "   Example: 'Patient has hypertension' → 'Patient has hypertension [doc:1]'\n\n"
            
            "2. HALLUCINATION (category: 'hallucination'):\n"
            "   Decision: Regenerate with stricter grounding\n"
            "   Action:\n"
            "   - Use generate_answer_with_context_tool\n"
            "   - Add instruction: 'ONLY use information EXPLICITLY stated in context'\n"
            "   - Provide original context again with emphasis\n"
            "   - Remove any invented details from previous attempt\n"
            "   Example: Avoid adding dates/names/dosages not in context\n\n"
            
            "3. INCOMPLETE ANSWER (category: 'completeness'):\n"
            "   Decision: Retrieve more context OR decompose query\n"
            "   Strategy A - More Context:\n"
            "   - Use search_tool to get additional chunks (increase top_k)\n"
            "   - Rerank with rerank_smart_fallback_tool\n"
            "   - Regenerate with expanded context\n"
            "   Strategy B - Decompose Query:\n"
            "   - Use decompose_query_tool if question has multiple parts\n"
            "   - Answer sub-questions separately\n"
            "   - Combine answers coherently\n\n"
            
            "4. CITATION MISMATCH (category: 'citation_mismatch'):\n"
            "   Decision: Regenerate with correct source mapping\n"
            "   Action:\n"
            "   - Identify correct doc:X for each claim\n"
            "   - Use generate_answer_with_context_tool\n"
            "   - Provide explicit source mapping in instruction\n"
            "   Example: 'Lisinopril is in doc:1, Metformin is in doc:2'\n\n"
            
            "5. LOW QUALITY / POOR STRUCTURE (category: 'quality'):\n"
            "   Decision: Check question type and format accordingly\n"
            "   Actions based on question type:\n"
            "   - Multiple Choice: 'Provide single letter answer (A/B/C/D) with reasoning'\n"
            "   - Bullet Points: 'Format as bullet list as requested'\n"
            "   - Short Answer: 'Provide concise 2-3 sentence answer'\n"
            "   - True/False: 'Answer True or False with justification'\n"
            "   Use generate_answer_with_context_tool with format instructions\n\n"
            
            "6. LOW CONFIDENCE / UNCERTAIN (low confidence_score):\n"
            "   Decision: Try different retrieval strategy\n"
            "   Actions:\n"
            "   - Use enhance_query_tool to rephrase query\n"
            "   - Search again with improved query\n"
            "   - Use rerank_smart_fallback_tool with LLM reranking\n"
            "   - Regenerate with better context\n\n"
            
            "AUTONOMOUS DECISION-MAKING PROCESS:\n\n"
            
            "Step 1: Analyze validation result\n"
            "   - Review issues list (category, severity, message)\n"
            "   - Identify primary issue category\n"
            "   - Check quality_score and confidence_score\n\n"
            
            "Step 2: Select appropriate strategy\n"
            "   - Match issue category to decision tree\n"
            "   - Choose most relevant tool(s)\n"
            "   - Plan corrective action\n\n"
            
            "Step 3: Execute corrective action\n"
            "   - Call selected tool(s) with appropriate parameters\n"
            "   - Generate improved answer\n"
            "   - Return to Validation Agent for re-validation\n\n"
            
            "Step 4: Retry management\n"
            f"   - Track current retry count (max: {SELF_HEAL_MAX_RETRIES})\n"
            "   - If validation passes → SUCCESS, return answer\n"
            f"   - If retry count < {SELF_HEAL_MAX_RETRIES} → Try different strategy\n"
            f"   - If retry count = {SELF_HEAL_MAX_RETRIES} → Return best attempt\n\n"
            
            f"{'Step 5: Track best attempt' if SELF_HEAL_TRACK_BEST_ATTEMPT else ''}\n"
            f"{'   - Keep track of highest quality_score across retries' if SELF_HEAL_TRACK_BEST_ATTEMPT else ''}\n"
            f"{'   - If all retries fail, return answer with highest score' if SELF_HEAL_TRACK_BEST_ATTEMPT else ''}\n"
            f"{'   - Attach metadata: retry_count, best_score, improvement' if SELF_HEAL_TRACK_BEST_ATTEMPT else ''}\n\n"
            
            "MULTI-ISSUE HANDLING:\n"
            "   When validation shows multiple issues:\n"
            "   - Prioritize by severity: critical > warning > info\n"
            "   - Address most critical issue first\n"
            "   - If possible, combine fixes (e.g., citations + format)\n\n"
            
            "QUESTION TYPE AWARENESS:\n"
            "   Always check question type and format accordingly:\n"
            "   - 'Select the correct option:' → Multiple choice format\n"
            "   - 'List the...' or 'bullet points' → Bullet list format\n"
            "   - 'True or False' → Boolean answer format\n"
            "   - 'Describe briefly' → Short paragraph format\n\n"
            
            "EXAMPLE HEALING WORKFLOW:\n\n"
            "Attempt 1 (Original):\n"
            "   Answer: 'Patient prescribed medication for blood pressure.'\n"
            "   Validation: FAIL (quality_score: 0.45, issues: missing citations)\n"
            "   Analysis: Citation issue detected\n\n"
            "Retry 1 (Self-Healing):\n"
            "   Decision: Regenerate with citation emphasis\n"
            "   Tool: generate_answer_with_context_tool\n"
            "   Instruction: 'Cite every claim with [doc:X]'\n"
            "   Answer: 'Patient prescribed Lisinopril 10mg [doc:1] for blood pressure.'\n"
            "   Validation: PASS (quality_score: 0.85)\n"
            "   Result: SUCCESS ✅\n\n"
            "You are the autonomous problem-solver that ensures high-quality answers "
            "through intelligent corrective actions. You learn from validation feedback "
            "and make smart decisions about how to improve answers."
        ),
        
        tools=[
            generate_answer_with_context_tool,  # Regenerate with improvements
            search_tool,                         # Get more context
            decompose_query_tool,                # Break down complex queries
            enhance_query_tool,                  # Rephrase query
            rerank_smart_fallback_tool,          # Rerank differently
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM
        verbose=verbose,
        allow_delegation=False,  # Self-healer executes fixes directly
        max_iter=10,  # May need multiple tool calls per retry
    )
    
    return agent


def get_self_healer_agent(**kwargs) -> Agent:
    """
    Convenience function to get self-healer agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_self_healer_agent
        
    Returns:
        Self-Healer Agent instance
        
    Example:
        >>> agent = get_self_healer_agent(verbose=False)
    """
    return create_self_healer_agent(**kwargs)


# Export for easy imports
__all__ = ['create_self_healer_agent', 'get_self_healer_agent']
