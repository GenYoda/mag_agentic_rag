"""
Tools package for Phase 3 - Orchestration Layer

Wraps Phase 1 & 2 core components into tool interfaces.
Plain Python classes (CrewAI integration in Phase 4).
"""

from .extraction_tools import ExtractionTools, extract_single_pdf, extract_multiple_pdfs

__all__ = [
    #Tool 1
    'ExtractionTools',
    'extract_single_pdf',
    'extract_multiple_pdfs',
    # Tool 2: Knowledge Base
    'KBTools',
    'build_knowledge_base',
    'load_knowledge_base',
    # Tool 3: Retrieval
    'RetrievalTools',
    'search_knowledge_base',

    # Tool 4: Answer Generation
    'AnswerGenerationTools',
    'ask_question',

    # Tool 5: Reranking
    'RerankingTools',
    'rerank_results',
]
