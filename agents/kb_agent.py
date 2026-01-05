"""
================================================================================
KB AGENT - Agent 7/9
================================================================================
Purpose: Manage knowledge base operations
Responsibilities:
- Index new documents into FAISS vector store
- Update existing documents and rebuild index
- Delete documents and update metadata
- Get knowledge base statistics and health
- Manage KB versioning and updates

Tools Used:
- index_document_tool: Index single document into KB
- index_directory_tool: Batch index all documents in directory
- update_document_tool: Update existing document and reindex
- delete_document_tool: Remove document from KB
- get_kb_stats_tool: Get KB statistics and metadata
- rebuild_index_tool: Full index rebuild

Integration:
- Works independently for KB management tasks
- Supports document ingestion pipeline
- Maintains FAISS index and metadata consistency
- Provides stats to other agents
================================================================================
"""
from config.settings import CREW_MAX_RPM
from crewai import Agent, LLM
from config.settings import azure_settings
from tools.kb_tools import (
    index_document_tool,
    index_directory_tool,
    update_document_tool,
    delete_document_tool,
    get_kb_stats_tool,
    rebuild_index_tool,
)


def create_kb_agent(verbose: bool = True) -> Agent:
    """
    Create the Knowledge Base Agent.
    
    The KB Agent manages all knowledge base operations including indexing,
    updating, deleting documents, and maintaining the FAISS vector store.
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for KB management
        
    Example:
        >>> kb_agent = create_kb_agent(verbose=True)
        >>> # Use in a Crew to manage knowledge base
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
        role="Knowledge Base Manager",
        
        goal=(
            "Manage the medical knowledge base efficiently, ensuring documents are properly "
            "indexed, metadata is accurate, and the FAISS vector store is optimized for retrieval. "
            "Maintain KB health and provide statistics."
        ),
        
        backstory=(
            "You are an expert in knowledge base management and vector store operations. "
            "You manage a medical document knowledge base that powers the RAG system.\n\n"
            "Your responsibilities:\n\n"
            "1. DOCUMENT INDEXING:\n"
            "   - Extract text from medical documents (PDF, DOCX, TXT)\n"
            "   - Chunk documents into semantic units (typically 500-1000 tokens)\n"
            "   - Generate embeddings using text-embedding-3-large\n"
            "   - Store embeddings in FAISS IndexFlatL2 index\n"
            "   - Maintain metadata (source file, page numbers, timestamps)\n\n"
            "2. INDEX MAINTENANCE:\n"
            "   - Update documents when content changes\n"
            "   - Delete removed documents and rebuild index\n"
            "   - Optimize index structure for fast retrieval\n"
            "   - Verify index integrity and consistency\n\n"
            "3. BATCH OPERATIONS:\n"
            "   - Index entire directories of documents\n"
            "   - Handle large-scale ingestion pipelines\n"
            "   - Progress tracking and error handling\n"
            "   - Resume interrupted operations\n\n"
            "4. STATISTICS & MONITORING:\n"
            "   - Track total indexed documents and chunks\n"
            "   - Monitor embedding dimension (should be 3072)\n"
            "   - Provide KB version and update history\n"
            "   - Report indexing errors and warnings\n\n"
            "5. VERSIONING:\n"
            "   - Track KB updates with version hashes\n"
            "   - Invalidate cache when KB changes\n"
            "   - Maintain update history and audit trail\n\n"
            "You ensure the knowledge base is always up-to-date, accurate, and optimized "
            "for the retrieval pipeline. You work closely with the Retrieval Agent which "
            "searches the FAISS index you maintain."
        ),
        
        tools=[
            index_document_tool,     # Index single document
            index_directory_tool,    # Batch index directory
            update_document_tool,    # Update existing document
            delete_document_tool,    # Delete document
            get_kb_stats_tool,       # KB statistics
            rebuild_index_tool,      # Full rebuild
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM
        verbose=verbose,
        allow_delegation=False,
        max_iter=10,  # May need multiple operations for batch indexing
        max_rpm=CREW_MAX_RPM
    )
    
    return agent


def get_kb_agent(**kwargs) -> Agent:
    """
    Convenience function to get KB agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_kb_agent
        
    Returns:
        KB Agent instance
        
    Example:
        >>> agent = get_kb_agent(verbose=False)
    """
    return create_kb_agent(**kwargs)


# Export for easy imports
__all__ = ['create_kb_agent', 'get_kb_agent']
