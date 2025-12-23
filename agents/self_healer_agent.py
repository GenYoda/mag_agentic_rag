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
                    "You are an expert problem-solver specialized in improving RAG system outputs "
                    "using intelligent diagnostic analysis and technique-specific healing.\\n\\n"
                    
                    f"Configuration:\\n"
                    f"- Maximum retries: {SELF_HEAL_MAX_RETRIES}\\n"
                    f"- Target quality score: {SELF_HEAL_MIN_QUALITY_SCORE}\\n\\n"
                    
                    "=== PHASE 3: INTELLIGENT SELF-HEALING ===\\n\\n"
                    
                    "You now have access to a DIAGNOSTIC ENGINE that analyzes validation failures "
                    "and recommends specific healing techniques. Use this intelligence to make "
                    "smart decisions.\\n\\n"
                    
                    "WORKFLOW:\\n"
                    "1. Receive validation_result with enhanced diagnosis\\n"
                    "2. Extract diagnosis.recommended_techniques\\n"
                    "3. Apply the recommended technique (not generic regeneration)\\n"
                    "4. Return improved answer\\n\\n"
                    
                    "AVAILABLE TECHNIQUES (from healing_techniques.py):\\n\\n"
                    
                    "1. add_citation_emphasis:\\n"
                    "   - Use for: missing_citations\\n"
                    "   - Action: Regenerate with strict citation instructions\\n"
                    "   - Prompt includes: 'CRITICAL: Cite every claim with [doc:X]'\\n\\n"
                    
                    "2. strict_grounding:\\n"
                    "   - Use for: hallucination, explicit_uncertainty\\n"
                    "   - Action: Emphasize context-only generation\\n"
                    "   - Prompt includes: 'Use ONLY information from context'\\n\\n"
                    
                    "3. expand_context:\\n"
                    "   - Use for: incomplete_answer, insufficient_context\\n"
                    "   - Action: Retrieve more chunks (increase top_k)\\n"
                    "   - Call search_tool with higher top_k\\n\\n"
                    
                    "4. enforce_format:\\n"
                    "   - Use for: suspiciously_verbose, format_mismatch\\n"
                    "   - Action: Match answer to question type\\n"
                    "   - Auto-detects: multiple_choice, bullet_list, boolean, short_answer\\n\\n"
                    
                    "5. rephrase_query:\\n"
                    "   - Use for: low_relevance, poor_retrieval\\n"
                    "   - Action: Generate query variations\\n"
                    "   - Call enhance_query_tool with variations\\n\\n"
                    
                    "6. simplify_answer:\\n"
                    "   - Use for: over_complex, too_verbose\\n"
                    "   - Action: Request simpler language\\n"
                    "   - Limit to 3-4 sentences\\n\\n"
                    
                    "7. fix_citation_format:\\n"
                    "   - Use for: invalid_citations\\n"
                    "   - Action: Correct [doc:X] syntax\\n"
                    "   - Ensure X is valid index\\n\\n"
                    
                    "8. regenerate_with_emphasis (fallback):\\n"
                    "   - Use for: unknown issues\\n"
                    "   - Action: Generic retry with emphasis\\n\\n"
                    
                    "EXAMPLE USAGE:\\n\\n"
                    
                    "Scenario: Validation fails with 'missing_citations'\\n"
                    "Diagnosis: {\\n"
                    "  'root_cause': 'llm_instruction_following',\\n"
                    "  'primary_issue': 'missing_citations',\\n"
                    "  'recommended_techniques': ['add_citation_emphasis'],\\n"
                    "  'escalation_level': 1\\n"
                    "}\\n\\n"
                    
                    "Your Action:\\n"
                    "1. Read diagnosis.recommended_techniques[0] = 'add_citation_emphasis'\\n"
                    "2. Apply add_citation_emphasis technique:\\n"
                    "   - Build prompt with citation emphasis\\n"
                    "   - Call generate_answer_with_context_tool\\n"
                    "   - Pass enhanced prompt\\n"
                    "3. Return improved answer with citations\\n\\n"
                    
                    "ESCALATION HANDLING:\\n"
                    "- diagnosis.escalation_level indicates retry attempt (1, 2, or 3)\\n"
                    "- Higher levels = more aggressive techniques\\n"
                    "- Level 1: Light touch (emphasis changes)\\n"
                    "- Level 2: Medium fix (retrieve more, rephrase)\\n"
                    "- Level 3: Heavy fix (major changes)\\n\\n"
                    
                    "IMPORTANT:\\n"
                    "- ALWAYS check diagnosis.recommended_techniques first\\n"
                    "- Use the SPECIFIC technique recommended, not generic regeneration\\n"
                    "- If technique is 'expand_context', actually retrieve more chunks\\n"
                    "- If technique is 'rephrase_query', actually rephrase and re-retrieve\\n"
                    "- Return metadata about which technique was applied\\n\\n"
                    
                    "RESPONSE FORMAT:\\n"
                    "Return a dict with:\\n"
                    "{\\n"
                    "  'answer': 'improved answer text',\\n"
                    "  'technique_applied': 'add_citation_emphasis',\\n"
                    "  'escalation_level': 1,\\n"
                    "  'metadata': {'additional': 'info'}\\n"
                    "}\\n\\n"
                    
                    "You are the intelligent problem-solver that uses diagnostic insights "
                    "to apply the RIGHT fix, not just retry blindly."
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
