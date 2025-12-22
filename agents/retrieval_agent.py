"""
================================================================================
RETRIEVAL AGENT - Agent 4/9
================================================================================
Purpose: Find most relevant document chunks using semantic search
Responsibilities:
- Perform FAISS vector search on knowledge base
- Handle multi-query retrieval (query variations + HyDE)
- Apply distance filtering to remove irrelevant results
- Return ranked results with similarity scores
- Support fusion strategies (reciprocal rank, max-score)

Tools Used:
- search_tool: Single query semantic search
- search_multiple_queries_tool: Multi-query fusion retrieval
- get_retrieval_stats_tool: Get retrieval statistics

Integration:
- Works after Query Enhancement Agent provides optimized queries
- Feeds retrieved chunks to Reranking Agent
- Uses FAISS index built by KB indexing pipeline
- Supports both single and multi-query retrieval
================================================================================
"""

from crewai import Agent, LLM
from config.settings import azure_settings
from tools.retrieval_tools import (
    search_tool,
    search_multiple_queries_tool,
    get_retrieval_stats_tool,
)


def create_retrieval_agent(verbose: bool = True) -> Agent:
    """
    Create the Retrieval Agent.
    
    The Retrieval Agent performs semantic search on the knowledge base using
    FAISS vector similarity. It can handle single queries or multiple query
    variations, fusing results for better coverage.
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for semantic retrieval
        
    Example:
        >>> retrieval_agent = create_retrieval_agent(verbose=True)
        >>> # Use in a Crew to retrieve relevant document chunks
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
        temperature=0.0,  # Deterministic for retrieval
    )
    
    agent = Agent(
        role="Contextual Retriever",
        
        goal=(
            "Find the most relevant medical document chunks using advanced retrieval "
            "techniques including vector search, multi-query fusion, and distance filtering. "
            "Ensure high recall while maintaining precision for medical queries."
        ),
        
        backstory=(
            "You are an expert in semantic search and vector similarity. You search through "
            "indexed medical documents to find the most relevant chunks for a given query. "
            "You work with FAISS vector indices that contain embeddings of document chunks.\n\n"
            "Your retrieval process:\n"
            "1. For a query, you generate its embedding using the same model used for indexing\n"
            "2. You search the FAISS index for chunks with highest cosine similarity\n"
            "3. You filter results by distance threshold to remove irrelevant matches\n"
            "4. You return ranked chunks with similarity scores and metadata\n\n"
            "When given multiple query variations (from Query Enhancement Agent), you perform "
            "multi-query retrieval and fuse results using strategies like:\n"
            "- Reciprocal Rank Fusion (RRF): Combines rankings from multiple queries\n"
            "- Max Score: Takes highest similarity score across all queries\n\n"
            "For example, if given queries:\n"
            "- 'What is the patient diagnosis?'\n"
            "- 'What was the patient diagnosed with?'\n"
            "- 'What medical condition does the patient have?'\n\n"
            "You search each query independently and merge results, ensuring chunks that appear "
            "in multiple result sets rank higher (they're likely more relevant).\n\n"
            "You understand that medical queries need high recall (find all relevant info) but "
            "also need to filter noise. You work closely with the Reranking Agent who refines "
            "your initial results."
        ),
        
        tools=[
            search_tool,                    # Single query search
            search_multiple_queries_tool,   # Multi-query fusion
            get_retrieval_stats_tool,       # Statistics
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM
        verbose=verbose,
        allow_delegation=False,
        max_iter=5,  # May need multiple searches for complex queries
    )
    
    return agent


def get_retrieval_agent(**kwargs) -> Agent:
    """
    Convenience function to get retrieval agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_retrieval_agent
        
    Returns:
        Retrieval Agent instance
        
    Example:
        >>> agent = get_retrieval_agent(verbose=False)
    """
    return create_retrieval_agent(**kwargs)


# Export for easy imports
__all__ = ['create_retrieval_agent', 'get_retrieval_agent']
