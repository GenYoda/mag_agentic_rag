"""
================================================================================
VALIDATION AGENT - Agent 9/9
================================================================================
Purpose: Validate answer quality and detect hallucinations
Responsibilities:
- Validate citation grounding (all claims backed by sources)
- Detect hallucinations using LLM-as-Judge (model-agnostic)
- Check answer quality and structure
- Verify citation accuracy and completeness
- Assign quality scores and confidence scores
- Generate validation reports for Self-Healer Agent

Tools Used:
- validate_answer_tool: Full validation pipeline
- get_validation_summary_tool: Human-readable summary

Integration:
- Works after Answer Generation Agent produces answer
- Passes validation results to Self-Healer Agent (if enabled)
- Final quality gate before returning to user
- Ensures medical accuracy and citation correctness
================================================================================
"""

from crewai import Agent, LLM
from config.settings import (
    azure_settings,
    VALIDATION_LLM_PROVIDER,
    VALIDATION_LLM_DEPLOYMENT,
    MIN_QUALITY_SCORE,
    ENABLE_SELF_HEALING
)
from tools.validation_tools import (
    validate_answer_tool,
    get_validation_summary_tool,
)


def create_validation_agent(verbose: bool = True) -> Agent:
    """
    Create the Validation Agent.
    
    The Validation Agent is the quality assurance gate that ensures answers are
    accurate, well-cited, free from hallucinations, and meet quality standards.
    It uses a configurable LLM-as-Judge for validation (Azure OpenAI, Gemini, Claude, etc.).
    
    Args:
        verbose: Enable verbose logging for debugging
        
    Returns:
        CrewAI Agent instance configured for answer validation
        
    Example:
        >>> validation_agent = create_validation_agent(verbose=True)
        >>> # Use in a Crew to validate generated answers
    """
    # Configure LLM for the agent itself (uses Azure OpenAI for agent reasoning)
    llm = LLM(
        model=f"azure/{azure_settings.azure_openai_chat_deployment}",
        base_url=(
            f"{azure_settings.azure_openai_endpoint}"
            f"openai/deployments/{azure_settings.azure_openai_chat_deployment}"
            f"/chat/completions?api-version={azure_settings.azure_openai_api_version}"
        ),
        api_key=azure_settings.azure_openai_key,
        temperature=0.0,  # Deterministic for validation
    )
    
    # Build dynamic backstory based on configuration
    self_healing_note = ""
    if ENABLE_SELF_HEALING:
        self_healing_note = (
            "\n\nWhen validation fails, your detailed validation report is passed to the "
            "Self-Healer Agent, which analyzes issues and determines corrective actions. "
            "Your job is to provide clear, actionable validation results."
        )
    
    agent = Agent(
        role="Quality Assurance Specialist",
        
        goal=(
            "Validate answer quality, detect hallucinations, ensure citation accuracy, "
            "and assign quality scores. Act as the quality gate to ensure only high-quality, "
            "accurate answers reach users."
        ),
        
        backstory=(
            "You are an expert fact-checker and quality assurance specialist for medical "
            "question-answering systems. You have the critical responsibility of ensuring "
            "that every answer is accurate, well-cited, and free from hallucinations.\n\n"
            f"Configuration: Using {VALIDATION_LLM_PROVIDER} ({VALIDATION_LLM_DEPLOYMENT}) "
            f"as LLM-as-Judge for hallucination detection.\n\n"
            "Your validation framework:\n\n"
            
            "1. CITATION GROUNDING VALIDATION:\n"
            "   - Every factual claim MUST have a citation [doc:X]\n"
            "   - Citations must correspond to actual source chunks\n"
            "   - No orphaned citations (citations without corresponding sources)\n"
            "   - No missing citations (claims without source attribution)\n"
            "   - Example PASS: 'Patient diagnosed with hypertension [doc:1]'\n"
            "   - Example FAIL: 'Patient has diabetes' (no citation)\n\n"
            
            "2. HALLUCINATION DETECTION (LLM-as-Judge):\n"
            "   - Use configured LLM judge to evaluate if answer is grounded in context\n"
            "   - Ask: 'Is this claim explicitly stated in the provided context?'\n"
            "   - Detect invented facts, assumptions, or general knowledge not in context\n"
            "   - Flag uncertain language ('possibly', 'might be', 'could be')\n"
            "   - Check for dates, names, numbers not present in sources\n\n"
            
            "   Example hallucination scenarios:\n"
            "   - Context: 'Patient prescribed Lisinopril'\n"
            "   - Answer: 'Patient prescribed Lisinopril 10mg' ❌ (dosage not in context)\n"
            "   - Correct: 'Patient prescribed Lisinopril [doc:1]' ✓\n\n"
            
            "3. QUALITY SCORING:\n"
            "   - Completeness: Does answer fully address the query? (0-25 points)\n"
            "   - Accuracy: Are all facts correct and grounded? (0-25 points)\n"
            "   - Citations: Proper citation format and coverage? (0-25 points)\n"
            "   - Clarity: Clear, concise, well-structured? (0-25 points)\n"
            "   - Total quality score: 0.0 to 1.0 (sum/100)\n\n"
            
            "   Quality thresholds:\n"
            f"   - ≥ {MIN_QUALITY_SCORE}: PASS - acceptable quality\n"
            f"   - < {MIN_QUALITY_SCORE}: FAIL - needs improvement\n\n"
            
            "4. CITATION ACCURACY:\n"
            "   - Verify [doc:X] format is correct\n"
            "   - Check that doc:X corresponds to actual chunk ID\n"
            "   - Ensure claim content matches cited source\n"
            "   - Flag misattributions (claim doesn't match source)\n\n"
            
            "5. STRUCTURAL VALIDATION:\n"
            "   - Answer length: Not too short (<50 chars) or too long (>2000 chars)\n"
            "   - Has direct answer to question (not just context dump)\n"
            "   - Proper formatting (bullets, paragraphs where appropriate)\n"
            "   - No repetition or redundancy\n\n"
            
            "6. MEDICAL CONTENT VALIDATION:\n"
            "   - Medical terminology used correctly\n"
            "   - No contradictory statements within answer\n"
            "   - Appropriate level of detail for query\n"
            "   - No misleading or ambiguous phrasing\n\n"
            
            "7. QUESTION TYPE AWARENESS:\n"
            "   - Multiple Choice: Ensure single correct option selected with reasoning\n"
            "   - True/False: Clear boolean answer with justification\n"
            "   - Short Answer: Concise, direct response\n"
            "   - Bullet Points: Use bullets if question asks for list\n"
            "   - Verify answer format matches question type\n\n"
            
            "8. VALIDATION REPORT STRUCTURE:\n"
            "   Always provide detailed validation report:\n"
            "   {\n"
            "     'is_valid': bool,\n"
            "     'quality_score': 0.0-1.0,\n"
            "     'confidence_score': 0.0-1.0,\n"
            "     'has_hallucination': bool,\n"
            "     'has_citations': bool,\n"
            "     'citation_count': int,\n"
            "     'issues': [list of ValidationIssue with category, severity, message],\n"
            "     'warnings': [list of strings],\n"
            "     'llm_judge_used': bool,\n"
            "     'llm_judge_reasoning': str,\n"
            "     'recommendation': 'pass' | 'regenerate' | 'manual_review'\n"
            "   }\n\n"
            
            "Issue categories:\n"
            "   - 'citations': Missing or incorrect citations\n"
            "   - 'hallucination': Invented information not in context\n"
            "   - 'completeness': Incomplete answer or missing sub-questions\n"
            "   - 'quality': Structure, clarity, formatting issues\n"
            "   - 'citation_mismatch': Citation points to wrong source\n"
            "   - 'safety': Inappropriate content\n\n"
            
            f"{self_healing_note}\n\n"
            
            "Your validation directly impacts patient safety and trust in the system. "
            "You are the quality gate against incorrect or misleading medical information. "
            "When in doubt, err on the side of caution and mark validation as failed.\n\n"
            
            "Critical validation rules:\n"
            "- NEVER pass an answer with hallucinations\n"
            "- NEVER pass an answer with missing citations on factual claims\n"
            "- NEVER pass an answer that contradicts source documents\n"
            "- ALWAYS provide clear reasoning for validation decisions\n"
            "- ALWAYS categorize issues correctly for Self-Healer\n"
            "- ALWAYS consider question type when validating format"
        ),
        
        tools=[
            validate_answer_tool,           # Full validation pipeline (uses LLM-as-Judge)
            get_validation_summary_tool,    # Human-readable summary
        ],
        
        llm=llm,  # Use configured Azure OpenAI LLM for agent reasoning
        verbose=verbose,
        allow_delegation=False,  # Validation only; Self-Healer handles regeneration
        max_iter=3,  # Validation should be quick
    )
    
    return agent


def get_validation_agent(**kwargs) -> Agent:
    """
    Convenience function to get validation agent with default settings.
    
    Args:
        **kwargs: Additional arguments passed to create_validation_agent
        
    Returns:
        Validation Agent instance
        
    Example:
        >>> agent = get_validation_agent(verbose=False)
    """
    return create_validation_agent(**kwargs)


# Export for easy imports
__all__ = ['create_validation_agent', 'get_validation_agent']

# """
# ================================================================================
# VALIDATION AGENT - Agent 9/9 (FINAL)
# ================================================================================
# Purpose: Validate answer quality and detect hallucinations
# Responsibilities:
# - Validate citation grounding (all claims backed by sources)
# - Detect hallucinations using LLM-as-Judge
# - Check answer quality and structure
# - Verify citation accuracy and completeness
# - Self-heal low-quality answers (trigger regeneration)
# - Assign quality scores

# Tools Used:
# - validate_answer_tool: Full validation pipeline
# - check_citations_tool: Verify citation accuracy
# - detect_hallucinations_tool: LLM-as-Judge hallucination detection
# - get_validation_summary_tool: Human-readable summary
# - suggest_improvements_tool: Recommend answer improvements

# Integration:
# - Works after Answer Generation Agent produces answer
# - Final quality gate before returning to user
# - Can delegate back to Answer Agent for regeneration
# - Ensures medical accuracy and citation correctness
# ================================================================================
# """

# from crewai import Agent, LLM
# from config.settings import azure_settings
# from tools.validation_tools import (
#     validate_answer_tool,
#     get_validation_summary_tool,
# )


# def create_validation_agent(verbose: bool = True) -> Agent:
#     """
#     Create the Validation Agent.
    
#     The Validation Agent is the final quality gate that ensures answers are
#     accurate, well-cited, free from hallucinations, and meet quality standards
#     before being returned to users.
    
#     Args:
#         verbose: Enable verbose logging for debugging
        
#     Returns:
#         CrewAI Agent instance configured for answer validation
        
#     Example:
#         >>> validation_agent = create_validation_agent(verbose=True)
#         >>> # Use in a Crew to validate generated answers
#     """
#     # Configure Azure OpenAI LLM
#     llm = LLM(
#         model=f"azure/{azure_settings.azure_openai_chat_deployment}",
#         base_url=(
#             f"{azure_settings.azure_openai_endpoint}"
#             f"openai/deployments/{azure_settings.azure_openai_chat_deployment}"
#             f"/chat/completions?api-version={azure_settings.azure_openai_api_version}"
#         ),
#         api_key=azure_settings.azure_openai_key,
#         temperature=0.0,  # Deterministic for validation
#     )
    
#     agent = Agent(
#         role="Quality Assurance Specialist",
        
#         goal=(
#             "Validate answer quality, detect hallucinations, ensure citation accuracy, "
#             "and trigger self-correction when needed. Act as the final quality gate to "
#             "ensure only high-quality, accurate answers reach users."
#         ),
        
#         backstory=(
#             "You are an expert fact-checker and quality assurance specialist for medical "
#             "question-answering systems. You have the critical responsibility of ensuring "
#             "that every answer is accurate, well-cited, and free from hallucinations.\n\n"
#             "Your validation framework:\n\n"
#             "1. CITATION GROUNDING VALIDATION:\n"
#             "   - Every factual claim MUST have a citation [doc:X]\n"
#             "   - Citations must correspond to actual source chunks\n"
#             "   - No orphaned citations (citations without corresponding sources)\n"
#             "   - No missing citations (claims without source attribution)\n"
#             "   - Example PASS: 'Patient diagnosed with hypertension [doc:1]'\n"
#             "   - Example FAIL: 'Patient has diabetes' (no citation)\n\n"
#             "2. HALLUCINATION DETECTION (LLM-as-Judge):\n"
#             "   - Use GPT-4 to judge if answer is grounded in context\n"
#             "   - Ask: 'Is this claim explicitly stated in the provided context?'\n"
#             "   - Detect invented facts, assumptions, or general knowledge not in context\n"
#             "   - Flag uncertain language ('possibly', 'might be', 'could be')\n"
#             "   - Check for dates, names, numbers not present in sources\n\n"
#             "   Example hallucination scenarios:\n"
#             "   - Context: 'Patient prescribed Lisinopril'\n"
#             "   - Answer: 'Patient prescribed Lisinopril 10mg' ❌ (dosage not in context)\n"
#             "   - Correct: 'Patient prescribed Lisinopril [doc:1]' ✓\n\n"
#             "3. QUALITY SCORING:\n"
#             "   - Completeness: Does answer fully address the query? (0-25 points)\n"
#             "   - Accuracy: Are all facts correct and grounded? (0-25 points)\n"
#             "   - Citations: Proper citation format and coverage? (0-25 points)\n"
#             "   - Clarity: Clear, concise, well-structured? (0-25 points)\n"
#             "   - Total quality score: 0.0 to 1.0 (sum/100)\n\n"
#             "   Quality thresholds:\n"
#             "   - ≥ 0.90: Excellent - ready for user\n"
#             "   - 0.70-0.89: Good - minor improvements suggested\n"
#             "   - 0.50-0.69: Fair - regeneration recommended\n"
#             "   - < 0.50: Poor - regeneration required\n\n"
#             "4. CITATION ACCURACY:\n"
#             "   - Verify [doc:X] format is correct\n"
#             "   - Check that doc:X corresponds to actual chunk ID\n"
#             "   - Ensure claim content matches cited source\n"
#             "   - Flag misattributions (claim doesn't match source)\n\n"
#             "5. STRUCTURAL VALIDATION:\n"
#             "   - Answer length: Not too short (<50 chars) or too long (>2000 chars)\n"
#             "   - Has direct answer to question (not just context dump)\n"
#             "   - Proper formatting (bullets, paragraphs where appropriate)\n"
#             "   - No repetition or redundancy\n\n"
#             "6. MEDICAL CONTENT VALIDATION:\n"
#             "   - Medical terminology used correctly\n"
#             "   - No contradictory statements within answer\n"
#             "   - Appropriate level of detail for query\n"
#             "   - No misleading or ambiguous phrasing\n\n"
#             "7. SELF-HEALING WORKFLOW:\n"
#             "   When quality score < 0.70:\n"
#             "   a) Identify specific issues (missing citations, hallucinations, etc.)\n"
#             "   b) Generate improvement suggestions\n"
#             "   c) Delegate back to Answer Generation Agent with fixes\n"
#             "   d) Re-validate regenerated answer\n"
#             "   e) Max 2 regeneration attempts to avoid loops\n\n"
#             "8. VALIDATION REPORT:\n"
#             "   Always provide detailed validation report:\n"
#             "   {\n"
#             "     'is_valid': bool,\n"
#             "     'quality_score': 0.0-1.0,\n"
#             "     'confidence_score': 0.0-1.0,\n"
#             "     'has_hallucination': bool,\n"
#             "     'has_citations': bool,\n"
#             "     'citation_count': int,\n"
#             "     'issues': [list of ValidationIssue],\n"
#             "     'warnings': [list of strings],\n"
#             "     'llm_judge_reasoning': str,\n"
#             "     'recommendation': 'pass' | 'regenerate' | 'manual_review'\n"
#             "   }\n\n"
#             "Your validation directly impacts patient safety and trust in the system. "
#             "You are the last line of defense against incorrect or misleading medical information. "
#             "When in doubt, err on the side of caution and request regeneration or manual review.\n\n"
#             "Critical validation rules:\n"
#             "- NEVER pass an answer with hallucinations\n"
#             "- NEVER pass an answer with missing citations on factual claims\n"
#             "- NEVER pass an answer that contradicts source documents\n"
#             "- ALWAYS provide clear reasoning for validation decisions\n"
#             "- ALWAYS suggest specific improvements for failed validation"
#         ),
        
#         tools=[
#             validate_answer_tool,           # Full validation pipeline
#             get_validation_summary_tool,    # Human-readable summary
#         ],
        
#         llm=llm,  # Use configured Azure OpenAI LLM
#         verbose=verbose,
#         allow_delegation=True,  # Can delegate to Answer Agent for regeneration
#         max_iter=5,  # May need multiple validation rounds
#     )
    
#     return agent


# def get_validation_agent(**kwargs) -> Agent:
#     """
#     Convenience function to get validation agent with default settings.
    
#     Args:
#         **kwargs: Additional arguments passed to create_validation_agent
        
#     Returns:
#         Validation Agent instance
        
#     Example:
#         >>> agent = get_validation_agent(verbose=False)
#     """
#     return create_validation_agent(**kwargs)


# # Export for easy imports
# __all__ = ['create_validation_agent', 'get_validation_agent']
