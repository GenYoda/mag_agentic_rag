"""
================================================================================
CACHE AGENT - Agent 1/9
================================================================================
Purpose: Check semantic cache for previously answered similar questions
Responsibilities:
- Search cache for similar queries (semantic similarity matching)
- Return cached answers if similarity > threshold
- Add validated answers to cache
- Manage cache statistics

Tools Used:
- search_cache_tool: Search for semantically similar queries
- add_to_cache_tool: Add validated Q&A pairs to cache
- clear_cache_tool: Clear all cache entries

Integration:
- Works with ValidationAgent to cache only high-quality answers
- Reduces latency and API costs by avoiding duplicate work
- First agent in the RAG pipeline (checks cache before retrieval)
================================================================================
"""

from crewai import Agent, LLM
from config.settings import azure_settings  # ✅ Fixed import
from tools.cache_tools import search_cache_tool, add_to_cache_tool, clear_cache_tool


def create_cache_agent(verbose: bool = True) -> Agent:
    # Debug: Verify settings loaded
    print(f"Azure Endpoint: {azure_settings.azure_openai_endpoint}")
    print(f"Azure Key exists: {bool(azure_settings.azure_openai_key)}")
    print(f"Deployment: {azure_settings.azure_openai_chat_deployment}")
    
    # ✅ Use CrewAI's LLM class with Azure format
    llm = LLM(
        model=f"azure/{azure_settings.azure_openai_chat_deployment}",
        base_url=f"{azure_settings.azure_openai_endpoint}openai/deployments/{azure_settings.azure_openai_chat_deployment}/chat/completions?api-version={azure_settings.azure_openai_api_version}",
        api_key=azure_settings.azure_openai_key,
        temperature=0.1,
    )

    agent = Agent(
        role="Semantic Cache Specialist",
        goal=(
            "Identify and return previously answered similar questions to save time and cost. "
            "Maintain a high-performance QA cache with semantic similarity matching. "
            "Ensure cache hits for semantically equivalent queries even with different wording."
        ),
        backstory=(
            "You are an expert at semantic similarity and cache management. "
            "You quickly check if a user's question has been answered before by "
            "finding semantically similar queries in the cache."
        ),
        tools=[search_cache_tool, add_to_cache_tool, clear_cache_tool],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=3,
    )
    return agent


def get_cache_agent(**kwargs) -> Agent:
    """
    Convenience function to get cache agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_cache_agent
        
    Returns:
        Cache Agent instance
        
    Example:
        >>> agent = get_cache_agent(verbose=False)
    """
    return create_cache_agent(**kwargs)


# Export for easy imports
__all__ = ['create_cache_agent', 'get_cache_agent']
