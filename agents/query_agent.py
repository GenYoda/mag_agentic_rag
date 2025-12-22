"""
================================================================================
QUERY ENHANCEMENT AGENT - Agent 3/9
================================================================================
Purpose: Optimize queries for better retrieval
Responsibilities:
- Classify query type and complexity (factual, analytical, comparison, list)
- Decompose complex multi-part queries into atomic sub-queries
- Generate hypothetical answers (HyDE) for semantic search
- Create query variations for broader coverage
- Add medical terminology and synonyms

Tools Used:
- classify_query_tool: Classify query type and complexity
- decompose_query_tool: Break complex queries into sub-queries
- generate_hypothetical_answer_tool: HyDE for better retrieval
- generate_query_variations_tool: Create alternative phrasings
- enhance_query_tool: Full pipeline (classify → decompose → HyDE → variations)

Integration:
- Works after Memory Agent provides context
- Feeds enhanced queries to Retrieval Agent
- Handles both simple and complex medical queries
- Improves retrieval recall and precision
================================================================================
"""

from crewai import Agent, LLM
from config.settings import azure_settings
from tools.query_tools import (
    classify_query_tool,
    decompose_query_tool,
    generate_hypothetical_answer_tool,
    generate_query_variations_tool,
    enhance_query_tool,
)


def create_query_agent(verbose: bool = True) -> Agent:
    """
    Create the Query Enhancement Agent.
    
    The Query Enhancement Agent optimizes user queries before retrieval by
    classifying them, decomposing complex questions, generating hypothetical
    answers (HyDE), and creating query variations for broader coverage.
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for query enhancement
        
    Example:
        >>> query_agent = create_query_agent(verbose=True)
        >>> # Use in a Crew to enhance queries before retrieval
    """
    # Configure Azure OpenAI LLM (same pattern as memory_agent)
    llm = LLM(
        model=f"azure/{azure_settings.azure_openai_chat_deployment}",
        base_url=(
            f"{azure_settings.azure_openai_endpoint}"
            f"openai/deployments/{azure_settings.azure_openai_chat_deployment}"
            f"/chat/completions?api-version={azure_settings.azure_openai_api_version}"
        ),
        api_key=azure_settings.azure_openai_key,
        temperature=0.3,  # Slightly higher for creative query variations
    )
    
    agent = Agent(
        role="Query Enhancement Specialist",
        
        goal=(
            "Optimize queries for better retrieval through classification, decomposition, "
            "HyDE generation, and query expansion with medical terminology. Ensure queries "
            "are well-formed, complete, and likely to retrieve relevant documents."
        ),
        
        backstory=(
            "You are an expert in query understanding and semantic search optimization. "
            "You analyze user questions to determine their type (factual, analytical, "
            "comparison, list) and complexity (simple vs complex).\n\n"
            "For complex questions like 'What was the patient diagnosed with, who was the "
            "treating physician, and what medications were prescribed?', you break them into "
            "simpler sub-questions:\n"
            "1. What was the patient diagnosed with?\n"
            "2. Who was the treating physician?\n"
            "3. What medications were prescribed?\n\n"
            "You generate hypothetical answers (HyDE) that help find semantically similar "
            "documents. For example, for the query 'What treatment was provided for hypertension?', "
            "you generate: 'The patient was treated for hypertension with Lisinopril 10mg daily, "
            "along with lifestyle modifications including a low-sodium diet and regular exercise.'\n\n"
            "You expand queries with medical synonyms and alternative phrasings:\n"
            "- 'patient diagnosis' → ['patient diagnosis', 'what was diagnosed', 'medical condition identified']\n"
            "- 'medications prescribed' → ['medications prescribed', 'drugs administered', 'pharmaceutical treatment']\n\n"
            "You understand medical terminology and can rephrase clinical questions in multiple "
            "ways to maximize recall without losing precision. You ensure every query is optimized "
            "for the best possible retrieval results."
        ),
        
        tools=[
            classify_query_tool,                  # Classify query type
            decompose_query_tool,                 # Break into sub-queries
            generate_hypothetical_answer_tool,    # HyDE generation
            generate_query_variations_tool,       # Alternative phrasings
            enhance_query_tool,                   # Full pipeline
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM
        verbose=verbose,
        allow_delegation=False,
        max_iter=7,  # May need classify → decompose → HyDE → variations
    )
    
    return agent


def get_query_agent(**kwargs) -> Agent:
    """
    Convenience function to get query enhancement agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_query_agent
        
    Returns:
        Query Enhancement Agent instance
        
    Example:
        >>> agent = get_query_agent(verbose=False)
    """
    return create_query_agent(**kwargs)


# Export for easy imports
__all__ = ['create_query_agent', 'get_query_agent']
