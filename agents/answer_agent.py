"""
================================================================================
ANSWER GENERATION AGENT - Agent 6/9
================================================================================
Purpose: Generate concise, well-cited answers from context
Responsibilities:
- Format retrieved and reranked context for LLM
- Generate accurate answers with proper citations
- Handle conversation history for follow-up questions
- Ensure all claims are grounded in source documents
- Add citation markers [doc:X] for traceability

Tools Used:
- generate_answer_tool: Full RAG pipeline (retrieval + generation)
- generate_answer_with_context_tool: Generate from pre-retrieved context

Integration:
- Works after Reranking Agent provides optimal context
- Uses conversation history from Memory Agent
- Feeds answers to Validation Agent for quality check
- Ensures medical accuracy through source grounding
================================================================================
"""

from crewai import Agent, LLM
from config.settings import azure_settings
from tools.answer_generation_tools import (
    generate_answer_tool,
    generate_answer_with_context_tool,
)


def create_answer_agent(verbose: bool = True) -> Agent:
    """
    Create the Answer Generation Agent.
    
    The Answer Generation Agent synthesizes information from retrieved context
    into clear, accurate, well-cited medical answers. It NEVER makes claims
    without grounding them in the provided context.
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for answer generation
        
    Example:
        >>> answer_agent = create_answer_agent(verbose=True)
        >>> # Use in a Crew to generate answers from context
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
        temperature=0.1,  # Low temperature for factual medical answers
    )
    
    agent = Agent(
        role="Medical Answer Specialist",
        
        goal=(
            "Generate concise, accurate, well-cited medical answers from retrieved context. "
            "Ensure all claims are grounded in source documents with proper citations. "
            "Never hallucinate or make unsupported claims."
        ),
        
        backstory=(
            "You are a medical documentation expert trained to synthesize information from "
            "medical records into clear, precise answers. You have several critical responsibilities:\n\n"
            "1. ALWAYS CITE YOUR SOURCES:\n"
            "   - Use [doc:X] format where X is the chunk/document ID\n"
            "   - Every factual claim MUST have a citation\n"
            "   - Example: 'The patient was diagnosed with hypertension [doc:1] and prescribed "
            "Lisinopril 10mg daily [doc:1].'\n\n"
            "2. GROUND ALL CLAIMS IN CONTEXT:\n"
            "   - Only state facts explicitly present in the provided context\n"
            "   - If the context doesn't contain the answer, explicitly say so\n"
            "   - NEVER make up information or use general medical knowledge\n"
            "   - Example: If asked about surgery but context only mentions medications, respond: "
            "'Based on the available documents, there is no information about surgical history.'\n\n"
            "3. UNDERSTAND MEDICAL TERMINOLOGY:\n"
            "   - You can explain complex medical concepts clearly\n"
            "   - Use precise medical terminology when appropriate\n"
            "   - Simplify for clarity while maintaining accuracy\n"
            "   - Example: 'Hypertension (high blood pressure) was managed with...' [doc:2]\n\n"
            "4. HANDLE CONVERSATION CONTEXT:\n"
            "   - Maintain awareness of previous exchanges\n"
            "   - Provide contextually aware responses to follow-up questions\n"
            "   - Reference previous answers when relevant\n"
            "   - Example: If previously discussed Dr. Smith, can say 'As mentioned, Dr. Smith...' [doc:3]\n\n"
            "5. STRUCTURE ANSWERS EFFECTIVELY:\n"
            "   - Start with direct answer to the question\n"
            "   - Provide supporting details with citations\n"
            "   - Keep answers concise (2-4 sentences typically)\n"
            "   - Use bullet points for lists\n\n"
            "6. HANDLE UNCERTAINTY:\n"
            "   - If context is ambiguous, acknowledge it\n"
            "   - If multiple interpretations exist, present them clearly\n"
            "   - Never guess or fill gaps with assumptions\n"
            "   - Example: 'The document mentions treatment but doesn't specify the exact dosage.'\n\n"
            "Your answers directly impact patient care decisions, so accuracy and proper "
            "source attribution are paramount. The Validation Agent will check your work, "
            "but your first responsibility is to generate high-quality, grounded answers."
        ),
        
        tools=[
            generate_answer_tool,               # Full RAG: query → retrieval → answer
            generate_answer_with_context_tool,  # Generate from pre-retrieved context
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM
        verbose=verbose,
        allow_delegation=False,
        max_iter=4,  # May need iteration for complex answers
    )
    
    return agent


def get_answer_agent(**kwargs) -> Agent:
    """
    Convenience function to get answer generation agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_answer_agent
        
    Returns:
        Answer Generation Agent instance
        
    Example:
        >>> agent = get_answer_agent(verbose=False)
    """
    return create_answer_agent(**kwargs)


# Export for easy imports
__all__ = ['create_answer_agent', 'get_answer_agent']
