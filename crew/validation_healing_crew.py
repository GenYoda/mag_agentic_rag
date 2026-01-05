"""
Validation + Healing Crew for Medical RAG System

Purpose: Intelligent answer validation + self-healing with retry logic.

Architecture:
- ValidationAgent: Validates answer quality and provides diagnosis
- SelfHealerAgent: Applies healing techniques based on diagnosis
- Retry loop: Up to N attempts to achieve quality threshold
- Best attempt tracking: Returns best answer if all retries fail

Flow:
1. ValidationAgent validates initial answer
2. If quality < threshold → SelfHealerAgent heals answer
3. ValidationAgent re-validates healed answer
4. Repeat until quality >= threshold or max retries reached
5. Return best attempt (highest quality score)

Performance:
- ACCEPT path: ~800ms (validation only, no healing)
- HEAL path: ~2-6s (1-2 healing iterations)
"""

import json
import re
import logging
import os
from typing import Dict, List, Any, Optional
from crewai import Crew, Task, Process

# Import agents
from agents.validation_agent import create_validation_agent
from agents.self_healer_agent import create_self_healer_agent

# Import tasks
from crew.tasks import (
    create_validation_task,
    create_healing_task
)

# Import Azure settings
from config.settings import azure_settings

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Configuration for CrewAI Agents (Azure OpenAI)
# ============================================================================

def configure_crewai_for_azure():
    """
    Configure environment variables for CrewAI to use Azure OpenAI.
    
    CrewAI reads these env vars automatically when creating agents.
    This is simpler and more reliable than passing LLM objects directly.
    
    Sets required environment variables:
    - AZURE_API_KEY: Azure OpenAI API key
    - AZURE_API_BASE: Azure OpenAI endpoint
    - AZURE_API_VERSION: API version
    - OPENAI_API_KEY: Also set to Azure key (CrewAI fallback)
    - OPENAI_API_BASE: Also set to Azure endpoint
    
    Raises:
        ValueError: If Azure credentials are not configured
    """
    # Validate credentials exist
    if not azure_settings.azure_openai_key:
        raise ValueError("Azure OpenAI key not configured in settings")
    
    if not azure_settings.azure_openai_endpoint:
        raise ValueError("Azure OpenAI endpoint not configured in settings")
    
    # Set Azure OpenAI environment variables for CrewAI
    os.environ["AZURE_API_KEY"] = azure_settings.azure_openai_key
    os.environ["AZURE_API_BASE"] = azure_settings.azure_openai_endpoint
    os.environ["AZURE_API_VERSION"] = azure_settings.azure_openai_api_version
    os.environ["OPENAI_API_VERSION"] = azure_settings.azure_openai_api_version
    
    # Also set OPENAI_API_KEY to Azure key (CrewAI sometimes checks this)
    os.environ["OPENAI_API_KEY"] = azure_settings.azure_openai_key
    os.environ["OPENAI_API_BASE"] = azure_settings.azure_openai_endpoint
    
    logger.info(f"✅ Configured CrewAI environment for Azure OpenAI")
    logger.info(f"   Deployment: {azure_settings.azure_openai_chat_deployment}")
    logger.info(f"   Endpoint: {azure_settings.azure_openai_endpoint}")
    logger.info(f"   API Version: {azure_settings.azure_openai_api_version}")


class ValidationHealingCrew:
    """
    Intelligent validation + healing crew with retry logic.
    
    Features:
    - Two-agent architecture (validation + healing)
    - Automatic retry loop until quality threshold met
    - Best attempt tracking across retries
    - Robust JSON parsing with fallback handling
    - Detailed metadata for debugging
    
    Attributes:
        max_retries: Maximum healing attempts (default: 2)
        min_quality_score: Target quality threshold (default: 0.80)
        track_best: Track best attempt across retries
        verbose: Show detailed logs
    """
    
    def __init__(
        self,
        max_retries: int = 2,
        min_quality_score: float = 0.80,
        track_best: bool = True,
        verbose: bool = True
    ):
        """
        Initialize ValidationHealingCrew.
        
        Args:
            max_retries: Maximum healing attempts before giving up
            min_quality_score: Target quality score (0.0-1.0)
            track_best: Track and return best attempt if all fail
            verbose: Enable verbose logging
            
        Example:
            >>> crew = ValidationHealingCrew(max_retries=2, min_quality_score=0.80)
            >>> result = crew.validate_and_heal(query, answer, chunks)
        """
        self.max_retries = max_retries
        self.min_quality_score = min_quality_score
        self.track_best = track_best
        self.verbose = verbose
        
        # Configure CrewAI environment for Azure OpenAI
        logger.info(f"Initializing ValidationHealingCrew (max_retries={max_retries}, target={min_quality_score})")
        
        try:
            configure_crewai_for_azure()
        except Exception as e:
            logger.error(f"❌ Failed to configure Azure environment: {e}")
            raise
        
        # Create LLM string for Azure deployment
        # Format: "azure/<deployment-name>"
        llm_model = f"azure/{azure_settings.azure_openai_chat_deployment}"
        logger.info(f"Using LLM model: {llm_model}")
        
        # Create agents with Azure deployment string
        self.validation_agent = create_validation_agent(llm=llm_model, verbose=False)
        self.healer_agent = create_self_healer_agent(llm=llm_model, verbose=False)
        
        logger.info("✅ ValidationHealingCrew initialized")
    
    def validate_and_heal(
        self,
        query: str,
        answer: str,
        chunks: List[Dict],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate answer and heal if needed (with retry loop).
        
        Args:
            query: User question
            answer: Generated answer to validate/heal
            chunks: Context chunks used for generation
            metadata: Additional metadata (e.g., tier1_result)
            
        Returns:
            {
                'success': bool,
                'final_answer': str,
                'final_decision': 'ACCEPT' | 'HEAL' | 'REJECT',
                'confidence_score': float,
                'quality_score': float,
                'technique_applied': str | None,
                'retry_count': int,
                'validation_history': list,
                'metadata': dict
            }
        """
        if self.verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"🏥 ValidationHealingCrew - Starting validation + healing")
            logger.info(f"{'='*80}")
        
        # Initialize tracking
        current_answer = answer
        retry_count = 0
        validation_history = []
        best_attempt = {
            'answer': answer,
            'quality_score': 0.0,
            'confidence_score': 0.0,
            'retry': 0
        }
        
        try:
            # Main retry loop
            while retry_count <= self.max_retries:
                if self.verbose:
                    logger.info(f"\n--- Attempt {retry_count + 1}/{self.max_retries + 1} ---")
                
                # ============================================================
                # STEP 1: VALIDATION
                # ============================================================
                if self.verbose:
                    logger.info("🔍 Step 1: Validating answer...")
                
                validation_result = self._validate_answer(
                    query=query,
                    answer=current_answer,
                    chunks=chunks,
                    retry_count=retry_count
                )
                
                if not validation_result['success']:
                    logger.error(f"Validation failed: {validation_result.get('error')}")
                    return {
                        'success': False,
                        'error': f"Validation failed: {validation_result.get('error')}",
                        'final_answer': current_answer,
                        'final_decision': 'REJECT',
                        'confidence_score': 0.5,
                        'quality_score': 0.5,
                        'technique_applied': None,
                        'retry_count': retry_count,
                        'validation_history': validation_history
                    }
                
                # Extract validation results
                quality_score = validation_result['quality_score']
                confidence_score = validation_result['confidence_score']
                decision = validation_result['decision']
                
                # Track in history
                validation_history.append({
                    'retry': retry_count,
                    'quality_score': quality_score,
                    'confidence_score': confidence_score,
                    'decision': decision,
                    'issues': validation_result.get('issues', [])
                })
                
                # Update best attempt
                if self.track_best and quality_score > best_attempt['quality_score']:
                    best_attempt = {
                        'answer': current_answer,
                        'quality_score': quality_score,
                        'confidence_score': confidence_score,
                        'retry': retry_count
                    }
                
                if self.verbose:
                    logger.info(f"✅ Validation complete: decision={decision}, quality={quality_score:.2f}")
                
                # ============================================================
                # STEP 2: DECISION HANDLING
                # ============================================================
                
                # SUCCESS: Quality threshold met
                if decision == 'ACCEPT' or quality_score >= self.min_quality_score:
                    if self.verbose:
                        logger.info(f"✅ ACCEPT - Quality threshold met ({quality_score:.2f} >= {self.min_quality_score})")
                    
                    return {
                        'success': True,
                        'final_answer': current_answer,
                        'final_decision': 'ACCEPT',
                        'confidence_score': confidence_score,
                        'quality_score': quality_score,
                        'technique_applied': None if retry_count == 0 else validation_history[-2].get('technique_applied'),
                        'retry_count': retry_count,
                        'validation_history': validation_history,
                        'metadata': {
                            'attempts': retry_count + 1,
                            'final_attempt': retry_count,
                            'quality_improvement': quality_score - validation_history[0]['quality_score'] if retry_count > 0 else 0.0
                        }
                    }
                
                # REJECT: Quality too low, no more retries
                if decision == 'REJECT' and retry_count >= self.max_retries:
                    logger.warning(f"⚠️  Max retries reached, returning best attempt")
                    
                    return {
                        'success': True,
                        'final_answer': best_attempt['answer'],
                        'final_decision': 'HEAL',  # Attempted healing but didn't reach threshold
                        'confidence_score': best_attempt['confidence_score'],
                        'quality_score': best_attempt['quality_score'],
                        'technique_applied': 'best_attempt',
                        'retry_count': retry_count,
                        'validation_history': validation_history,
                        'metadata': {
                            'max_retries_reached': True,
                            'best_attempt_retry': best_attempt['retry'],
                            'quality_improvement': best_attempt['quality_score'] - validation_history[0]['quality_score']
                        }
                    }
                
                # HEAL: Apply healing technique
                if decision == 'HEAL' and retry_count < self.max_retries:
                    if self.verbose:
                        logger.info(f"🔧 Step 2: Healing answer (attempt {retry_count + 1}/{self.max_retries})...")
                    
                    healing_result = self._heal_answer(
                        query=query,
                        answer=current_answer,
                        chunks=chunks,
                        diagnosis=validation_result.get('diagnosis', {}),
                        retry_count=retry_count
                    )
                    
                    if not healing_result['success']:
                        logger.error(f"Healing failed: {healing_result.get('error')}")
                        retry_count += 1
                        continue  # Try next iteration with original answer
                    
                    # Update current answer for next validation
                    healed_answer = healing_result.get('answer')
                    technique_applied = healing_result.get('technique_applied', 'unknown')
                    
                    if healed_answer and healed_answer != current_answer:
                        current_answer = healed_answer
                        if self.verbose:
                            logger.info(f"✅ Healing complete: technique={technique_applied}")
                        
                        # Track technique in history
                        validation_history[-1]['technique_applied'] = technique_applied
                    else:
                        logger.warning("⚠️  Healing returned unchanged answer, continuing...")
                    
                    retry_count += 1
                    continue  # Next iteration will validate healed answer
                
                # Fallback: shouldn't reach here
                logger.warning(f"Unexpected decision state: {decision}, quality={quality_score}")
                retry_count += 1
            
            # Max retries exhausted without success
            logger.warning(f"⚠️  Max retries ({self.max_retries}) exhausted")
            
            return {
                'success': True,
                'final_answer': best_attempt['answer'],
                'final_decision': 'HEAL',
                'confidence_score': best_attempt['confidence_score'],
                'quality_score': best_attempt['quality_score'],
                'technique_applied': 'best_attempt',
                'retry_count': retry_count,
                'validation_history': validation_history,
                'metadata': {
                    'max_retries_exhausted': True,
                    'best_attempt_retry': best_attempt['retry']
                }
            }
        
        except Exception as e:
            logger.error(f"ValidationHealingCrew error: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Crew execution failed: {str(e)}',
                'final_answer': answer,
                'final_decision': 'REJECT',
                'confidence_score': 0.5,
                'quality_score': 0.5,
                'technique_applied': None,
                'retry_count': retry_count,
                'validation_history': validation_history
            }
    
    def _validate_answer(
        self,
        query: str,
        answer: str,
        chunks: List[Dict],
        retry_count: int
    ) -> Dict[str, Any]:
        """
        Run validation task.
        
        Returns:
            Parsed validation result with quality_score, decision, etc.
        """
        try:
            # Create validation task
            task = create_validation_task(
                agent=self.validation_agent,
                query=query,
                answer=answer,
                chunks=chunks
            )
            
            # Create single-agent crew
            crew = Crew(
                agents=[self.validation_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            # Execute
            output = crew.kickoff()
            
            # Parse output
            result = self._parse_validation_output(output)
            
            return {
                'success': True,
                **result
            }
        
        except Exception as e:
            logger.error(f"Validation task error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _heal_answer(
        self,
        query: str,
        answer: str,
        chunks: List[Dict],
        diagnosis: Dict,
        retry_count: int
    ) -> Dict[str, Any]:
        """
        Run healing task.
        
        Returns:
            Parsed healing result with improved answer and technique_applied.
        """
        try:
            # Build validation_result dict (what create_healing_task expects)
            validation_result = {
                'decision': 'HEAL',
                'decision_reasoning': diagnosis.get('issue_summary', 'Answer needs healing'),
                'diagnosis': diagnosis,
                'issues': [{
                    'severity': 'high',
                    'category': diagnosis.get('primary_issue_type', 'quality'),
                    'message': diagnosis.get('root_cause', 'Quality issues detected'),
                    'auto_fixable': True
                }]
            }
            
            # Create healing task with correct parameters
            task = create_healing_task(
                validation_result=validation_result,  # First parameter!
                query=query,
                answer=answer,
                chunks=chunks,
                agent=self.healer_agent
            )
            
            # Create single-agent crew
            crew = Crew(
                agents=[self.healer_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            # Execute
            output = crew.kickoff()
            
            # Parse output
            result = self._parse_healing_output(output)
            
            return {
                'success': True,
                **result
            }
        
        except Exception as e:
            logger.error(f"Healing task error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    
    def _parse_validation_output(self, output: Any) -> Dict[str, Any]:
        """
        Parse validation output with robust error handling.
        
        Handles:
        - Raw string output
        - CrewAI output objects
        - Markdown-wrapped JSON
        - Malformed JSON
        """
        try:
            # Extract raw text
            if hasattr(output, 'raw'):
                raw = output.raw
            elif hasattr(output, 'result'):
                raw = output.result
            else:
                raw = str(output)
            
            # Clean up JSON
            cleaned_json = self._extract_json_from_text(raw)
            
            if not cleaned_json:
                logger.warning("Could not extract JSON from validation output")
                return self._get_fallback_validation_result()
            
            # Parse JSON
            result = json.loads(cleaned_json)
            
            # Validate required fields
            required_fields = ['decision', 'quality_score', 'confidence_score']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Validation output missing '{field}', using fallback")
                    return self._get_fallback_validation_result()
            
            # Normalize decision
            decision = result.get('decision', 'HEAL').upper()
            if decision not in ['ACCEPT', 'HEAL', 'REJECT']:
                logger.warning(f"Invalid decision '{decision}', defaulting to HEAL")
                result['decision'] = 'HEAL'
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in validation: {e}")
            logger.error(f"Raw output: {raw[:300]}...")
            return self._get_fallback_validation_result()
        
        except Exception as e:
            logger.error(f"Error parsing validation output: {e}", exc_info=True)
            return self._get_fallback_validation_result()
    
    def _parse_healing_output(self, output: Any) -> Dict[str, Any]:
        """
        Parse healing output with robust error handling.
        
        Handles:
        - Raw string output
        - CrewAI output objects
        - Markdown-wrapped JSON
        - Malformed JSON
        - Missing answer field
        """
        try:
            # Extract raw text
            if hasattr(output, 'raw'):
                raw = output.raw
            elif hasattr(output, 'result'):
                raw = output.result
            else:
                raw = str(output)
            
            # Clean up JSON
            cleaned_json = self._extract_json_from_text(raw)
            
            if not cleaned_json:
                logger.warning("Could not extract JSON from healing output, treating as text answer")
                return {
                    'success': True,
                    'answer': raw.strip(),
                    'technique_applied': 'text_fallback',
                    'confidence_score': 0.70,
                    'quality_improvement': 0.0,
                    'metadata': {'parsing_fallback': True}
                }
            
            # Parse JSON
            result = json.loads(cleaned_json)
            
            # Validate answer field exists
            if 'answer' not in result or not result['answer']:
                logger.warning("Healing output missing 'answer' field")
                return {
                    'success': False,
                    'error': 'Healing output missing answer field'
                }
            
            # Set defaults for optional fields
            result.setdefault('technique_applied', 'unknown')
            result.setdefault('confidence_score', 0.75)
            result.setdefault('quality_improvement', 0.0)
            result.setdefault('metadata', {})
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in healing: {e}")
            logger.error(f"Raw output: {raw[:300]}...")
            logger.warning("Could not parse healing output as JSON, treating as text answer")
            
            # Fallback: treat entire output as answer
            return {
                'success': True,
                'answer': raw.strip(),
                'technique_applied': 'json_parse_error',
                'confidence_score': 0.70,
                'quality_improvement': 0.0,
                'metadata': {'json_error': str(e)}
            }
        
        except Exception as e:
            logger.error(f"Error parsing healing output: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from text that may contain markdown or extra text.
        
        Handles:
        - ```json ... ``` code blocks
        - Text before/after JSON
        - Multiple JSON objects (takes first)
        
        Returns:
            Cleaned JSON string or None if not found
        """
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find first { and last }
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start >= 0 and json_end > json_start:
            return text[json_start:json_end+1]
        
        return None
    
    def _get_fallback_validation_result(self) -> Dict[str, Any]:
        """
        Return safe fallback validation result when parsing fails.
        """
        return {
            'is_valid': False,
            'quality_score': 0.50,
            'confidence_score': 0.50,
            'decision': 'HEAL',
            'decision_reasoning': 'Validation output parsing failed, triggering healing',
            'has_hallucination': False,
            'has_citations': True,
            'citation_count': 0,
            'issues': ['Validation parsing error'],
            'diagnosis': {
                'root_cause': 'parsing_error',
                'primary_issue_type': 'unknown',
                'recommended_techniques': ['regenerate_with_emphasis'],
                'escalation_level': 1
            }
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_validation_healing_crew(verbose: bool = True, **kwargs) -> ValidationHealingCrew:
    """
    Convenience function to create Validation Healing Crew.
    
    Args:
        verbose: Enable verbose logging
        **kwargs: Additional arguments (max_retries, min_quality_score)
        
    Returns:
        ValidationHealingCrew instance
        
    Example:
        >>> crew = create_validation_healing_crew(verbose=True, max_retries=3)
    """
    return ValidationHealingCrew(verbose=verbose, **kwargs)


# Export for easy imports
__all__ = ['ValidationHealingCrew', 'create_validation_healing_crew']
