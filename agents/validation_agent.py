"""
VALIDATION AGENT - External Quality Validator
Purpose: Score answer quality and identify specific fixable issues
Responsibilities:
- Score answer on 4 dimensions (0-40 total)
- Decide ACCEPT (≥32) or HEAL (<32)
- Identify specific actionable issues
- Configurable LLM: Gemini OR Azure OpenAI
Integration:
- Called after answer generation
- Passes issues to Self Healing Agent
- Independent module (no inline logic in runner)
"""

import logging
import json
from typing import Dict, List, Any, Optional
from config.settings import (
    azure_settings,              # ← Import the Pydantic object
    VALIDATOR_LLM,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    AZURE_VALIDATOR_DEPLOYMENT,
    VALIDATION_SCORE_THRESHOLD
)


logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    External validation agent that scores quality and identifies issues.
    
    Features:
    - Single LLM call for scoring + diagnosis (~300ms)
    - Configurable: Gemini (default) or Azure OpenAI
    - 4-dimensional scoring (relevance, completeness, citations, accuracy)
    - Decision: ACCEPT or HEAL
    - Returns actionable issues for healing
    """
    
    def __init__(self, llm_provider: str = None):
        """
        Initialize Validation Agent.
        
        Args:
            llm_provider: "gemini" or "azure" (defaults to VALIDATOR_LLM setting)
        """
        self.llm_provider = llm_provider or VALIDATOR_LLM
        logger.info(f"ValidationAgent initialized with LLM: {self.llm_provider}")
        
        # Validate configuration
        if self.llm_provider == "gemini":
            if not GEMINI_API_KEY:
                logger.warning("Gemini API key not set, falling back to Azure")
                self.llm_provider = "azure"
        
        if self.llm_provider == "azure":
            if not azure_settings.azure_openai_key or not azure_settings.azure_openai_endpoint:
                raise ValueError("Azure OpenAI credentials not configured")

    
    def evaluate(self, query: str, answer: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate answer quality and identify issues.
        
        Args:
            query: Original user question
            answer: Generated answer to evaluate
            chunks: Context chunks used for generation
            
        Returns:
            {
                "score": 28,  # 0-40
                "decision": "HEAL",  # ACCEPT or HEAL
                "confidence": 0.70,  # 0.0-1.0
                "relevance": 7,  # 0-10
                "completeness": 6,  # 0-10
                "citations": 5,  # 0-10
                "accuracy": 10,  # 0-10
                "issues": ["Missing citation...", "Date unclear..."],
                "reasoning": "Answer incomplete..."
            }
        """
        logger.info(f"Evaluating answer quality with {self.llm_provider}...")
        
        try:
            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(query, answer, chunks)
            
            # Call LLM
            if self.llm_provider == "gemini":
                response = self._call_gemini(prompt)
                
                # ✅ ADD THIS DEBUG BLOCK - TEMPORARY!
                logger.error("=" * 80)
                logger.error("🔍 GEMINI DEBUG - FULL RESPONSE:")
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response length: {len(response)} chars")
                logger.error("Raw response:")
                logger.error(repr(response))  # Shows exact string with \n, \t, etc.
                logger.error("Pretty printed:")
                logger.error(response)
                logger.error("=" * 80)
            else:
                response = self._call_azure(prompt)
            
            # Parse response
            result = self._parse_response(response)
            
            # Calculate decision
            score = result.get("score", 0)
            result["decision"] = "ACCEPT" if score >= VALIDATION_SCORE_THRESHOLD else "HEAL"
            result["confidence"] = self._calculate_confidence(result)
            
            logger.info(f"Evaluation complete: score={score}/40, decision={result['decision']}")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            # Return safe fallback
            return {
                "score": 20,
                "decision": "HEAL",
                "confidence": 0.5,
                "relevance": 5,
                "completeness": 5,
                "citations": 5,
                "accuracy": 5,
                "issues": [f"Evaluation error: {str(e)}"],
                "reasoning": "Validation failed, defaulting to healing"
            }
    
    def _build_evaluation_prompt(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Build evaluation prompt for LLM."""
        
        # Prepare context summary
        context_text = "\n\n".join([
            f"[Chunk {i+1}]: {chunk.get('text', str(chunk))[:300]}..."
            for i, chunk in enumerate(chunks[:5])
        ])
        
        prompt = f"""You are a medical RAG answer quality evaluator. Score this answer on 4 dimensions.

QUERY: {query}

ANSWER TO EVALUATE:
{answer}

AVAILABLE CONTEXT (5 chunks):
{context_text}

SCORING INSTRUCTIONS:

1. RELEVANCE (0-10): Does answer directly address the query?
   - 10 = Directly answers, on-topic
   - 7-9 = Mostly relevant, some tangent
   - 4-6 = Partially relevant
   - 0-3 = Off-topic or unclear

2. COMPLETENESS (0-10): Is answer sufficiently detailed for query complexity?
   - Simple query ("Who is X?"): 10 = name + role + key fact
   - Complex query ("What procedures?"): 10 = all procedures + dates + outcomes
   - 7-9 = Adequate detail
   - 4-6 = Too brief
   - 0-3 = Severely incomplete

3. CITATIONS (0-10): Are factual claims properly cited with [doc:X]?
   - 10 = Multiple citations, all claims cited
   - 7-9 = 2-3 citations, most claims cited
   - 4-6 = 1 citation or sparse citations
   - 0-3 = No citations or wrong format

4. ACCURACY (0-10): Do facts match the context?
   - 10 = All facts directly from context
   - 7-9 = Mostly accurate, minor interpretation
   - 4-6 = Some questionable facts
   - 0-3 = Contradicts context or hallucinations

IDENTIFY ISSUES:
List specific actionable problems (if any):
- "Missing citation for [specific claim]"
- "Date not specified for [event]"
- "Hospital name not mentioned"
- "Procedure details incomplete"
- etc.

Respond ONLY with JSON:
{{
  "relevance": 0-10,
  "completeness": 0-10,
  "citations": 0-10,
  "accuracy": 0-10,
  "score": 0-40 (sum of above),
  "issues": [
    "specific issue 1",
    "specific issue 2"
  ],
  "reasoning": "brief explanation (2-3 sentences)"
}}

Evaluate now:"""
        
        return prompt
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API using new google-genai package."""
        try:
            from google import genai
            from google.genai import types
            
            # Configure client
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            # ✅ HARDCODED: Force high token limit to prevent truncation
            max_tokens = 2000  # Way more than needed
            
            logger.debug(f"Calling Gemini with max_output_tokens={max_tokens}")
            
            # Generate response
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=max_tokens,  # ✅ Hardcoded
                    response_mime_type="application/json"
                )
            )
            
            # Extract response with detailed checking
            response_text = None
            
            # Method 1: Try accessing through candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Log finish reason
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
                    logger.debug(f"Gemini finish reason: {finish_reason}")
                    
                    if "MAX_TOKENS" in finish_reason or "LENGTH" in finish_reason:
                        logger.error(f"⚠️ Gemini response truncated! Finish reason: {finish_reason}")
                
                # Extract text from parts
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    if parts:
                        response_text = parts[0].text
                        logger.debug(f"Extracted from parts: {len(response_text)} chars")
            
            # Method 2: Try direct text attribute
            if not response_text and hasattr(response, 'text'):
                response_text = response.text
                logger.debug(f"Extracted from .text: {len(response_text)} chars")
            
            # Method 3: Try str conversion
            if not response_text:
                response_text = str(response)
                logger.warning(f"Fell back to str(response): {len(response_text)} chars")
            
            # Validate response
            if not response_text:
                raise ValueError("Gemini returned empty response")
            
            if len(response_text.strip()) < 50:
                raise ValueError(f"Gemini response too short ({len(response_text)} chars): {response_text}")
            
            if not response_text.strip().endswith('}'):
                logger.warning(f"⚠️ Gemini response might be incomplete - no closing brace")
                logger.warning(f"Response: {response_text}")
            
            logger.debug(f"✅ Gemini full response: {len(response_text)} chars")
            
            return response_text
            
        except ImportError:
            logger.error("google-genai package not installed. Run: pip install google-genai")
            raise
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            raise


    
    def _call_azure(self, prompt: str) -> str:
        """Call Azure OpenAI API."""
        try:
            from utils.azure_clients import get_openai_client
            from config.settings import azure_settings
            
            client = get_openai_client()
            response = client.chat.completions.create(
                model=AZURE_VALIDATOR_DEPLOYMENT,  # ← Use model= not deployment_name=
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content

            
        except Exception as e:
            logger.error(f"Azure API call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract scores and issues."""
        try:
            # Clean response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0]
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0]
            
            # Parse JSON
            result = json.loads(response.strip())
            
            # Validate required fields
            required = ["relevance", "completeness", "citations", "accuracy"]
            for field in required:
                if field not in result:
                    logger.warning(f"Missing field: {field}, defaulting to 5")
                    result[field] = 5
            
            # Calculate total score if missing
            if "score" not in result:
                result["score"] = (
                    result["relevance"] + 
                    result["completeness"] + 
                    result["citations"] + 
                    result["accuracy"]
                )
            
            # Ensure issues list
            if "issues" not in result:
                result["issues"] = []
            
            # Ensure reasoning
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided"
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response was: {response[:200]}...")
            # Return fallback
            return {
                "relevance": 5,
                "completeness": 5,
                "citations": 5,
                "accuracy": 5,
                "score": 20,
                "issues": ["Parse error in validation"],
                "reasoning": f"Failed to parse validation response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected parse error: {e}")
            return {
                "relevance": 5,
                "completeness": 5,
                "citations": 5,
                "accuracy": 5,
                "score": 20,
                "issues": [f"Parse error: {str(e)}"],
                "reasoning": "Unexpected parsing error"
            }

    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score (0.0-1.0) based on validation results."""
        score = result.get("score", 0)
        
        # Base confidence from score
        confidence = score / 40.0
        
        # Boost for strong citations
        if result.get("citations", 0) >= 8:
            confidence += 0.05
        
        # Boost for high accuracy
        if result.get("accuracy", 0) >= 9:
            confidence += 0.05
        
        # Penalty for issues
        issue_count = len(result.get("issues", []))
        if issue_count > 3:
            confidence -= 0.10
        
        # Cap at 0.95
        return min(max(confidence, 0.0), 0.95)


# Convenience function
def create_validation_agent(llm_provider: str = None) -> ValidationAgent:
    """
    Create a ValidationAgent instance.
    
    Args:
        llm_provider: "gemini" or "azure" (defaults to config setting)
    
    Returns:
        ValidationAgent instance
    
    Example:
        validator = create_validation_agent("gemini")
        result = validator.evaluate(query, answer, chunks)
        print(f"Score: {result['score']}/40, Decision: {result['decision']}")
    """
    return ValidationAgent(llm_provider=llm_provider)
