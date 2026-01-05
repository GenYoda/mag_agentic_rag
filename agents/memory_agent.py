"""
================================================================================
MEMORY AGENT - Agent 2/9
================================================================================
Purpose: Maintain conversation context and resolve references
Responsibilities:
- Track conversation history across multiple turns
- Extract and remember entities (patients, doctors, medications, dates)
- Resolve pronouns and references ("it", "that", "he/she", "the patient")
- Provide relevant context for follow-up questions
- Detect follow-up questions vs new queries

Tools Used:
- get_context_for_query_tool: Get conversation context and resolve references
- add_exchange_tool: Store Q&A exchange in memory
- extract_entities_tool: Extract medical entities from text
- resolve_coreference_tool: Resolve pronouns using entity memory

Integration:
- Works after Cache Agent (if cache miss)
- Provides context to Query Enhancement Agent
- Enables multi-turn conversations with context awareness
- Maintains entity memory for pronoun resolution
================================================================================
"""
from config.settings import CREW_MAX_RPM
from crewai import Agent, LLM
from config.settings import azure_settings
from tools.memory_tools import (
    get_context_for_query_tool,
    add_exchange_tool,
    extract_entities_tool,
    resolve_coreference_tool,
)


def create_memory_agent(verbose: bool = True) -> Agent:
    """
    Create the Memory Agent.

    The Memory Agent maintains conversation state across multiple turns, enabling
    contextual follow-up questions. It tracks entities and resolves references so
    users can ask "What about his diagnosis?" after previously asking about a patient.

    Args:
        verbose: Enable verbose logging for debugging

    Returns:
        CrewAI Agent instance configured for conversation memory management
    """
    # Debug: verify Azure settings are loaded (optional)
    print(f"Azure Endpoint: {azure_settings.azure_openai_endpoint}")
    print(f"Azure Key exists: {bool(azure_settings.azure_openai_key)}")
    print(f"Deployment: {azure_settings.azure_openai_chat_deployment}")

    # Use CrewAI's LLM class with Azure format (same pattern as cache_agent)
    llm = LLM(
        model=f"azure/{azure_settings.azure_openai_chat_deployment}",
        base_url=azure_settings.azure_openai_endpoint,  # ✅ Just the base URL
        api_key=azure_settings.azure_openai_key,
        api_version=azure_settings.azure_openai_api_version,  # ✅ Pass api_version separately
        temperature=0.1,
    )

    agent = Agent(
        role="Conversation Memory Specialist",
        goal=(
            "Maintain conversation context, resolve references, and provide relevant history. "
            "Track entities (patients, doctors, medications, dates) across conversation turns. "
            "Ensure follow-up questions have proper context by resolving pronouns and references."
        ),
        backstory=(
            "You are an expert at tracking conversation flow and contextual understanding. "
            "You remember what has been discussed in previous turns, who has been mentioned, "
            "and what entities (patients, doctors, medications, dates, diagnoses) are relevant.\n\n"
            "When users ask follow-up questions using pronouns like 'he', 'she', 'it', or 'that', "
            "you know exactly what they're referring to based on conversation history. For example, "
            "if a user asks 'Who is the patient?' and you learn it's John Smith, then when they "
            "ask 'What is his diagnosis?', you understand 'his' refers to John Smith.\n\n"
            "You extract and maintain entity memory from every conversation turn. You track:\n"
            "- Patient names and attributes\n"
            "- Doctor/provider names and roles\n"
            "- Medications and dosages\n"
            "- Dates and temporal references\n"
            "- Diagnoses and medical conditions\n"
            "- Healthcare facilities and locations\n\n"
            "You provide rich context for every query by retrieving relevant previous exchanges "
            "and resolving ambiguous references. Your memory spans the entire conversation session, "
            "enabling natural multi-turn dialogues."
        ),
        tools=[
            get_context_for_query_tool,   # Get context and resolve references
            add_exchange_tool,            # Store Q&A in memory
            extract_entities_tool,        # Extract medical entities
            resolve_coreference_tool,     # Resolve pronouns
        ],
        llm=llm,                         # ← same pattern as cache_agent
        verbose=verbose,
        allow_delegation=False,
        max_iter=5,
        max_rpm=CREW_MAX_RPM
    )

    return agent


def get_memory_agent(**kwargs) -> Agent:
    """
    Convenience function to get memory agent with default settings.

    Args:
        **kwargs: Additional arguments passed to create_memory_agent

    Returns:
        Memory Agent instance
    """
    return create_memory_agent(**kwargs)


__all__ = ["create_memory_agent", "get_memory_agent"]
