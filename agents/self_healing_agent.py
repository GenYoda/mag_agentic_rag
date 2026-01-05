"""
SELF-HEALING AGENT - External Issue Fixer
Purpose: Fix specific issues using 8 healing strategies
Responsibilities:
- Receive issues from Validation Agent
- Map issues to appropriate healing strategies
- Apply strategies using existing tools
- Return improved answer with metadata
Healing Strategies:
1. add_citation_emphasis - Regenerate with citation instructions
2. strict_grounding - Emphasize context-only generation
3. expand_context - Retrieve more chunks
4. enforce_format - Match answer to question type
5. rephrase_query - Generate query variations
6. simplify_answer - Request simpler language
7. fix_citation_format - Correct [doc:X] syntax
8. regenerate_with_emphasis - Generic retry (fallback)
Integration:
- Called when Validation Agent decides HEAL
- Uses 5 existing tools (no custom functions)
- Returns healed answer for re-validation
"""

import logging
from typing import Dict, List, Any, Optional
from tools.answer_generation_tools import AnswerGenerationTools
from tools.retrieval_tools import RetrievalTools
from tools.query_tools import QueryTools
from tools.reranking_tools import get_reranker

logger = logging.getLogger(__name__)


class SelfHealingAgent:
    """
    External self-healing agent that fixes answer quality issues.
    
    Features:
    - 8 healing strategies (prompt engineering + tools)
    - Maps issues to strategies automatically
    - Uses existing tools (no custom functions)
    - Returns healed answer with metadata
    """
    
    def __init__(self):
        """Initialize Self-Healing Agent."""
        self.answer_gen = AnswerGenerationTools()
        self.retrieval = RetrievalTools()
        self.query_tools = QueryTools()
        logger.info("SelfHealingAgent initialized")
    
    def heal(
        self,
        query: str,
        answer: str,
        chunks: List[Dict],
        issues: List[str],
        validation_score: int = 0
    ) -> Dict[str, Any]:
        """
        Heal answer by addressing specific issues.
        
        Args:
            query: Original user question
            answer: Current answer (needs improvement)
            chunks: Context chunks used
            issues: List of specific issues from Validation Agent
            validation_score: Score from validation (0-40)
        
        Returns:
            {
                "healed_answer": "improved answer text",
                "strategy_used": "add_citation_emphasis",
                "issues_addressed": 3,
                "success": True,
                "metadata": {...}
            }
        """
        logger.info(f"Starting healing for {len(issues)} issues...")
        
        try:
            # Determine primary strategy based on issues
            strategy = self._select_strategy(issues, validation_score, answer)
            logger.info(f"Selected strategy: {strategy}")
            
            # Apply strategy
            if strategy == "add_citation_emphasis":
                result = self._add_citation_emphasis(query, answer, chunks)
            elif strategy == "strict_grounding":
                result = self._strict_grounding(query, answer, chunks)
            elif strategy == "expand_context":
                result = self._expand_context(query, answer, chunks)
            elif strategy == "enforce_format":
                result = self._enforce_format(query, answer, chunks)
            elif strategy == "rephrase_query":
                result = self._rephrase_query(query, answer, chunks)
            elif strategy == "simplify_answer":
                result = self._simplify_answer(query, answer, chunks)
            elif strategy == "fix_citation_format":
                result = self._fix_citation_format(query, answer, chunks)
            else:  # regenerate_with_emphasis (fallback)
                result = self._regenerate_with_emphasis(query, answer, chunks)
            
            return {
                "healed_answer": result,
                "strategy_used": strategy,
                "issues_addressed": len(issues),
                "success": True,
                "metadata": {
                    "original_score": validation_score,
                    "issues": issues
                }
            }
            
        except Exception as e:
            logger.error(f"Healing failed: {e}", exc_info=True)
            return {
                "healed_answer": answer,  # Return original
                "strategy_used": "none",
                "issues_addressed": 0,
                "success": False,
                "metadata": {"error": str(e)}
            }
    
    def _select_strategy(self, issues: List[str], score: int, answer: str) -> str:
        """Select healing strategy based on issues."""
        
        # Convert issues to lowercase for matching
        issues_lower = [issue.lower() for issue in issues]
        
        # Strategy selection logic
        if any("citation" in issue or "doc:" in issue for issue in issues_lower):
            if any("missing" in issue for issue in issues_lower):
                return "add_citation_emphasis"
            else:
                return "fix_citation_format"
        
        if any("hallucination" in issue or "not in context" in issue for issue in issues_lower):
            return "strict_grounding"
        
        if any("incomplete" in issue or "brief" in issue or "detail" in issue for issue in issues_lower):
            # Check answer length
            if len(answer.split()) < 30:
                return "expand_context"
            else:
                return "simplify_answer"
        
        if any("date" in issue or "name" in issue or "hospital" in issue for issue in issues_lower):
            return "expand_context"
        
        if any("unclear" in issue or "vague" in issue for issue in issues_lower):
            return "rephrase_query"
        
        if any("format" in issue or "structure" in issue for issue in issues_lower):
            return "enforce_format"
        
        # Fallback
        return "regenerate_with_emphasis"
    
    # ========================================
    # STRATEGY 1: Add Citation Emphasis
    # ========================================
    def _add_citation_emphasis(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Regenerate with strict citation instructions."""
        logger.info("Applying: add_citation_emphasis")
        
        enhanced_prompt = f"""CRITICAL: You MUST cite EVERY factual claim with [doc:X] format.

Original Query: {query}

Previous Answer (missing citations): {answer}

Task: Rewrite the answer with PROPER CITATIONS for EVERY claim.

Rules:
- Every fact, date, name, number MUST have [doc:X]
- Use format: "claim [doc:1]" or "claim [doc:1][doc:2]" for multiple sources
- X must be 1-{len(chunks)} (chunk index)
- NO claim without citation!

Rewrite with citations:"""
        
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_prompt,
            context_chunks=chunks,
            temperature=0.2  # Low temperature for precise citations
        )
        
        return result.get("answer", answer)
    
    # ========================================
    # STRATEGY 2: Strict Grounding
    # ========================================
    def _strict_grounding(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Emphasize context-only generation (prevent hallucinations)."""
        logger.info("Applying: strict_grounding")
        
        enhanced_prompt = f"""CRITICAL: Use ONLY information EXPLICITLY stated in the context below. NO external knowledge!

Query: {query}

Previous Answer (had hallucinations): {answer}

Task: Rewrite answer using ONLY facts from context.

Rules:
- ONLY use information directly from context chunks
- Do NOT add general knowledge or assumptions
- Do NOT infer beyond what's stated
- If information is missing, say "not mentioned in available records"
- Every claim must have [doc:X] citation

Rewrite (context-only):"""
        
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_prompt,
            context_chunks=chunks,
            temperature=0.1  # Very low for strict grounding
        )
        
        return result.get("answer", answer)
    
    # ========================================
    # STRATEGY 3: Expand Context
    # ========================================
    def _expand_context(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Retrieve more chunks and regenerate."""
        logger.info("Applying: expand_context")
        
        # Retrieve more chunks (top_k=10 instead of 5)
        try:
            retrieval_result = self.retrieval.search(
                query=query,
                top_k=10,
                max_distance=1.5
            )
            
            if retrieval_result.get("success"):
                expanded_chunks = retrieval_result.get("results", [])
                logger.info(f"Expanded context: {len(chunks)} → {len(expanded_chunks)} chunks")
            else:
                expanded_chunks = chunks
        except Exception as e:
            logger.warning(f"Context expansion failed: {e}")
            expanded_chunks = chunks
        
        # Regenerate with more context
        enhanced_prompt = f"""Query: {query}

Previous answer was incomplete: {answer}

Task: Provide a MORE COMPLETE answer using the expanded context.

Add:
- Specific dates, names, locations
- Additional relevant details
- More comprehensive information
- Proper [doc:X] citations

Complete answer:"""
        
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_prompt,
            context_chunks=expanded_chunks,
            temperature=0.3
        )
        
        return result.get("answer", answer)
    
    # ========================================
    # STRATEGY 4: Enforce Format
    # ========================================
    def _enforce_format(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Match answer to question type."""
        logger.info("Applying: enforce_format")
        
        # Detect question type
        query_lower = query.lower()
        if any(word in query_lower for word in ["list", "enumerate", "what are"]):
            format_instruction = "Format as a numbered or bulleted list."
        elif any(word in query_lower for word in ["yes or no", "is it", "does", "did"]):
            format_instruction = "Provide a clear YES or NO answer, then explain."
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            format_instruction = "Structure as a comparison with clear distinctions."
        else:
            format_instruction = "Provide a clear, well-structured answer (2-3 paragraphs)."
        
        enhanced_prompt = f"""Query: {query}

Previous answer (poor format): {answer}

Task: Rewrite with proper format and structure.

Format requirement: {format_instruction}

Rules:
- Clear structure
- Proper citations [doc:X]
- Easy to read
- Directly answers the question

Reformatted answer:"""
        
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_prompt,
            context_chunks=chunks,
            temperature=0.3
        )
        
        return result.get("answer", answer)
    
    # ========================================
    # STRATEGY 5: Rephrase Query
    # ========================================
    def _rephrase_query(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Generate query variations and re-retrieve."""
        logger.info("Applying: rephrase_query")
        
        try:
            # Enhance query (get variations)
            enhanced_result = self.query_tools.enhance_query(query)
            enhanced_query = enhanced_result.get("enhanced_query", query)
            
            logger.info(f"Query rephrased: '{query}' → '{enhanced_query}'")
            
            # Re-retrieve with new query
            retrieval_result = self.retrieval.search(
                query=enhanced_query,
                top_k=7,
                max_distance=1.5
            )
            
            if retrieval_result.get("success"):
                new_chunks = retrieval_result.get("results", [])
                logger.info(f"Re-retrieved {len(new_chunks)} chunks with rephrased query")
            else:
                new_chunks = chunks
        except Exception as e:
            logger.warning(f"Query rephrasing failed: {e}")
            enhanced_query = query
            new_chunks = chunks
        
        # Regenerate with rephrased query and new chunks
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_query,
            context_chunks=new_chunks,
            temperature=0.3
        )
        
        return result.get("answer", answer)
    
    # ========================================
    # STRATEGY 6: Simplify Answer
    # ========================================
    def _simplify_answer(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Request simpler, more concise language."""
        logger.info("Applying: simplify_answer")
        
        enhanced_prompt = f"""Query: {query}

Previous answer (too complex/verbose): {answer}

Task: Simplify this answer.

Requirements:
- Use plain, simple language
- Maximum 3-4 sentences (unless query requires more)
- Remove unnecessary complexity
- Keep all important facts and citations [doc:X]
- Direct and clear

Simplified answer:"""
        
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_prompt,
            context_chunks=chunks,
            temperature=0.3
        )
        
        return result.get("answer", answer)
    
    # ========================================
    # STRATEGY 7: Fix Citation Format
    # ========================================
    def _fix_citation_format(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Correct [doc:X] syntax."""
        logger.info("Applying: fix_citation_format")
        
        enhanced_prompt = f"""Query: {query}

Previous answer (incorrect citation format): {answer}

Task: Fix citation format to proper [doc:X] syntax.

Rules:
- Use format [doc:1], [doc:2], etc.
- X must be 1-{len(chunks)} (valid chunk index)
- Place citation after the claim it supports
- Remove invalid citations

Corrected answer:"""
        
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_prompt,
            context_chunks=chunks,
            temperature=0.2
        )
        
        return result.get("answer", answer)
    
    # ========================================
    # STRATEGY 8: Regenerate with Emphasis (Fallback)
    # ========================================
    def _regenerate_with_emphasis(self, query: str, answer: str, chunks: List[Dict]) -> str:
        """Generic retry with quality emphasis."""
        logger.info("Applying: regenerate_with_emphasis (fallback)")
        
        enhanced_prompt = f"""Query: {query}

Previous answer had quality issues: {answer}

Task: Regenerate with BETTER QUALITY.

Requirements:
- Directly answer the question
- Use proper citations [doc:X]
- Be complete and accurate
- Clear and well-structured

Improved answer:"""
        
        result = self.answer_gen.generate_answer_with_custom_context(
            query=enhanced_prompt,
            context_chunks=chunks,
            temperature=0.3
        )
        
        return result.get("answer", answer)


# Convenience function
def create_self_healing_agent() -> SelfHealingAgent:
    """
    Create a SelfHealingAgent instance.
    
    Returns:
        SelfHealingAgent instance
    
    Example:
        healer = create_self_healing_agent()
        result = healer.heal(query, answer, chunks, issues, score=28)
        print(f"Healed: {result['healed_answer']}")
    """
    return SelfHealingAgent()
