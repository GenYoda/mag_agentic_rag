"""
================================================================================
RERANKING AGENT - Agent 5/9
================================================================================
Purpose: Rerank retrieved chunks for optimal relevance
Responsibilities:
- Cross-encoder reranking (local, fast, accurate)
- LLM-based reranking (fallback for low-quality scores)
- Smart fallback on low-confidence cross-encoder results
- Score normalization and ranking optimization
- Quality-aware reranking strategy selection

Tools Used:
- rerank_crossencoder_tool: Fast local cross-encoder reranking
- rerank_llm_tool: LLM-based relevance scoring
- rerank_smart_fallback_tool: Automatic fallback strategy

Integration:
- Works after Retrieval Agent provides initial candidates
- Refines ranking to prioritize most relevant chunks
- Feeds reranked results to Answer Generation Agent
- Reduces false positives from initial retrieval
================================================================================
"""
from config.settings import CREW_MAX_RPM
from crewai import Agent, LLM
from config.settings import azure_settings
from tools.reranking_tools import (
    rerank_crossencoder_tool,
    rerank_llm_tool,
    rerank_smart_fallback_tool,
)


def create_reranking_agent(verbose: bool = True) -> Agent:
    """
    Create the Reranking Agent.
    
    The Reranking Agent refines initial retrieval results using advanced
    relevance scoring models. It uses fast cross-encoder models by default
    and falls back to LLM scoring when needed.
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for result reranking
        
    Example:
        >>> reranking_agent = create_reranking_agent(verbose=True)
        >>> # Use in a Crew to rerank retrieved chunks
    """
    # Configure Azure OpenAI LLM
    llm = LLM(
        model=f"azure/{azure_settings.azure_openai_chat_deployment}",
        base_url=azure_settings.azure_openai_endpoint,  # ✅ Just the base URL
        api_key=azure_settings.azure_openai_key,
        api_version=azure_settings.azure_openai_api_version,  # ✅ Pass api_version separately
        temperature=0.0,
    )
    
    agent = Agent(
        role="Relevance Optimizer",
        
        goal=(
            "Rerank retrieved chunks to prioritize the most relevant content using "
            "cross-encoder models and LLM-based scoring with smart fallback. Ensure "
            "the top-ranked results are highly relevant to the query."
        ),
        
        backstory=(
            "You are an expert at relevance scoring and semantic similarity refinement. "
            "After the Retrieval Agent returns initial candidates using vector similarity, "
            "you re-score them using more sophisticated models to ensure the most relevant "
            "chunks rank highest.\n\n"
            "Your reranking strategies:\n\n"
            "1. Cross-Encoder Reranking (Default - Fast & Accurate):\n"
            "   - Uses models like 'cross-encoder/ms-marco-MiniLM-L-6-v2'\n"
            "   - Jointly encodes query + chunk for better relevance scoring\n"
            "   - Much more accurate than bi-encoder similarity\n"
            "   - Fast enough for real-time use (5-10ms per chunk)\n"
            "   - Example: Query 'patient diagnosis' + Chunk 'diagnosed with hypertension' → Score: 0.92\n\n"
            "2. LLM-Based Reranking (Fallback - Highest Quality):\n"
            "   - Uses GPT-4 or similar to score relevance\n"
            "   - Understands complex medical context and reasoning\n"
            "   - Slower but highest quality (100-200ms per chunk)\n"
            "   - Used when cross-encoder scores are low or ambiguous\n"
            "   - Example: LLM judges 'Does this chunk answer the question?' → Yes/No + Score\n\n"
            "3. Smart Fallback Strategy:\n"
            "   - Start with cross-encoder reranking\n"
            "   - If max score < threshold (e.g., 0.5), fall back to LLM\n"
            "   - Automatically selects best strategy based on quality\n"
            "   - Balances speed and accuracy\n\n"
            "Why reranking matters:\n"
            "- Initial retrieval uses bi-encoder (query embedding vs chunk embedding separately)\n"
            "- This can return false positives (high embedding similarity but not actually relevant)\n"
            "- Cross-encoders jointly process query+chunk, catching nuances\n"
            "- Example: Query 'patient current medications' might retrieve 'patient discontinued medications' "
            "with high embedding similarity, but cross-encoder scores it low because 'discontinued' != 'current'\n\n"
            "You ensure the Answer Generation Agent receives the BEST possible context by "
            "filtering out irrelevant chunks that slipped through initial retrieval."
        ),
        
        tools=[
            rerank_crossencoder_tool,      # Fast cross-encoder (default)
            rerank_llm_tool,               # LLM-based scoring (fallback)
            rerank_smart_fallback_tool,    # Automatic strategy selection (recommended)
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM
        verbose=verbose,
        allow_delegation=False,
        max_iter=4,  # Reranking is typically fast
        max_rpm=CREW_MAX_RPM
    )
    
    return agent


def get_reranking_agent(**kwargs) -> Agent:
    """
    Convenience function to get reranking agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_reranking_agent
        
    Returns:
        Reranking Agent instance
        
    Example:
        >>> agent = get_reranking_agent(verbose=False)
    """
    return create_reranking_agent(**kwargs)


# Export for easy imports
__all__ = ['create_reranking_agent', 'get_reranking_agent']
