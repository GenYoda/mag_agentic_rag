"""
================================================================================
EXTRACTOR AGENT - Agent 8/9
================================================================================
Purpose: Extract and process text from medical documents
Responsibilities:
- Extract text from PDFs, DOCX, TXT files
- Chunk documents into semantic units
- Extract metadata (pages, dates, entities)
- Clean and preprocess text
- Handle OCR for scanned documents
- Validate extraction quality

Tools Used:
- extract_pdf_tool: Extract text from PDF
- process_pdf_with_chunking_tool: Extract + chunk in one step
- extract_metadata_tool: Get document metadata
- validate_extraction_tool: Check extraction quality

Integration:
- Works with KB Agent for document ingestion
- Provides chunks to indexing pipeline
- Handles multi-format document processing
- Ensures text quality before indexing
================================================================================
"""

from crewai import Agent, LLM
from config.settings import azure_settings
from tools.extraction_tools import (
    extract_pdf_tool,
    process_pdf_with_chunking_tool,
    extract_metadata_tool,
    validate_extraction_tool,
)


def create_extractor_agent(verbose: bool = True) -> Agent:
    """
    Create the Extractor Agent.
    
    The Extractor Agent handles text extraction from medical documents,
    chunking, metadata extraction, and quality validation.
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for document extraction
        
    Example:
        >>> extractor_agent = create_extractor_agent(verbose=True)
        >>> # Use in a Crew to extract text from documents
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
        temperature=0.0,  # Deterministic for extraction
    )
    
    agent = Agent(
        role="Document Extraction Specialist",
        
        goal=(
            "Extract high-quality text from medical documents, chunk into semantic units, "
            "extract metadata, and ensure extraction quality. Handle PDFs, DOCX, and TXT files "
            "with robust error handling."
        ),
        
        backstory=(
            "You are an expert in document processing and text extraction. You handle medical "
            "documents of various formats and ensure clean, accurate text extraction for the "
            "RAG system.\n\n"
            "Your extraction pipeline:\n\n"
            "1. TEXT EXTRACTION:\n"
            "   - PDF: Use PyMuPDF (fitz) for fast, accurate extraction\n"
            "   - Handle both text-based and scanned PDFs\n"
            "   - Extract page-by-page with page number tracking\n"
            "   - Preserve document structure and formatting\n"
            "   - Clean extracted text (remove extra whitespace, normalize)\n\n"
            "2. CHUNKING STRATEGY:\n"
            "   - Semantic chunking: Split on natural boundaries (paragraphs, sections)\n"
            "   - Configurable chunk size (default: 500-1000 tokens)\n"
            "   - Overlapping chunks (default: 100 tokens) for context continuity\n"
            "   - Preserve sentence boundaries - never split mid-sentence\n"
            "   - Each chunk gets metadata: source file, page numbers, chunk ID\n\n"
            "3. METADATA EXTRACTION:\n"
            "   - File metadata: name, size, modification date\n"
            "   - Document metadata: page count, author, creation date\n"
            "   - Content signatures for deduplication\n"
            "   - File hash (SHA-256) for change detection\n"
            "   - Medical entities: patient names, dates, diagnoses (if configured)\n\n"
            "4. TEXT PREPROCESSING:\n"
            "   - Remove headers/footers (page numbers, document IDs)\n"
            "   - Normalize whitespace (convert multiple spaces to single)\n"
            "   - Fix common OCR errors (if OCR was used)\n"
            "   - Preserve medical terminology and abbreviations\n"
            "   - Handle special characters and Unicode properly\n\n"
            "5. QUALITY VALIDATION:\n"
            "   - Check for minimum text length (avoid empty extractions)\n"
            "   - Verify chunk count is reasonable (not too few/many)\n"
            "   - Detect extraction errors (garbled text, encoding issues)\n"
            "   - Flag scanned PDFs that may need OCR\n"
            "   - Validate medical content structure\n\n"
            "6. ERROR HANDLING:\n"
            "   - Handle corrupted PDFs gracefully\n"
            "   - Retry with alternative extraction methods\n"
            "   - Log detailed error messages for debugging\n"
            "   - Skip problematic pages while processing rest\n"
            "   - Provide partial results when possible\n\n"
            "Common extraction scenarios:\n"
            "- Medical record PDF → Extract text → Chunk by section → Index\n"
            "- Scanned document → OCR → Clean text → Chunk → Index\n"
            "- Multi-page report → Extract page-by-page → Merge → Chunk → Index\n"
            "- Batch processing → Process folder → Track progress → Report errors\n\n"
            "You ensure the KB Agent receives clean, well-structured chunks ready for "
            "embedding generation and indexing."
        ),
        
        tools=[
            extract_pdf_tool,                  # Extract raw text from PDF
            process_pdf_with_chunking_tool,    # Extract + chunk in one step (recommended)
            extract_metadata_tool,             # Get document metadata
            validate_extraction_tool,          # Validate extraction quality
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM
        verbose=verbose,
        allow_delegation=False,
        max_iter=5,  # May need retries for problematic documents
    )
    
    return agent


def get_extractor_agent(**kwargs) -> Agent:
    """
    Convenience function to get extractor agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_extractor_agent
        
    Returns:
        Extractor Agent instance
        
    Example:
        >>> agent = get_extractor_agent(verbose=False)
    """
    return create_extractor_agent(**kwargs)


# Export for easy imports
__all__ = ['create_extractor_agent', 'get_extractor_agent']
