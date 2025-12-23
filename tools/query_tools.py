"""
Query Tools - Query Enhancement Specialist

Enhances user queries before retrieval:
- Query classification (type, complexity)
- Query decomposition (break complex queries)
- HyDE generation (hypothetical answers)
- Query variations (alternative phrasings)

Features:
- LLM-based classification
- Multi-part query decomposition
- Hypothetical document embeddings
- Synonym generation

Integrates:
- utils/azure_clients.py (LLM access)
- config/settings.py (configuration)
"""



from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from crewai.tools import tool

import logging
import re
from config.settings import (
    ENABLE_QUERY_CLASSIFICATION,
    ENABLE_QUERY_DECOMPOSITION,
    ENABLE_HYDE,
    HYDE_FOR_SIMPLE_QUERIES,
    HYDE_TEMPERATURE,
    HYDE_MAX_TOKENS,
    ENABLE_QUERY_VARIATIONS,
    MAX_QUERY_VARIATIONS,
    MAX_SUBQUERIES,
    QUERY_VARIATION_METHOD,
    QUERY_ENHANCEMENT_TEMPERATURE,
    QUERY_ENHANCEMENT_MAX_TOKENS,
    QUERY_ENHANCEMENT_MODEL
)


from utils.azure_clients import get_openai_client

logger = logging.getLogger(__name__)


class QueryTools:
    """
    Enhances queries before retrieval.
    
    Features:
    - Query classification (simple vs complex)
    - Query decomposition (break into parts)
    - HyDE (hypothetical answers for better retrieval)
    - Query variations (alternative phrasings)
    """
    
    def __init__(self):
        """Initialize query tools."""
        self.client = get_openai_client()
        logger.info("QueryTools initialized")
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query type and complexity.
        
        Args:
            query: User query
            
        Returns:
            Dict with classification results
        """
        if not ENABLE_QUERY_CLASSIFICATION:
            logger.info("Query classification disabled")
            return {
                'type': 'unknown',
                'complexity': 'simple',
                'requires_decomposition': False,
                'requires_hyde': False,
                'requires_variations': False,
                'is_list_query': False
            }
        
        logger.info(f"Classifying query: '{query[:50]}...'")
        
        try:
            # Simple heuristic checks first (fast)
            query_lower = query.lower()
            
            # Check for multi-part queries (requires decomposition)
            multi_part_indicators = [' and ', ' or ', ';', '?', 'also']
            has_multiple_parts = any(ind in query_lower for ind in multi_part_indicators)
            question_marks = query.count('?')
            
            # Check for list queries
            list_indicators = ['all', 'list', 'what are', 'who are', 'every', 'each']
            is_list_query = any(ind in query_lower for ind in list_indicators)
            
            # Determine complexity
            word_count = len(query.split())
            if word_count > 15 or has_multiple_parts or question_marks > 1:
                complexity = 'complex'
            else:
                complexity = 'simple'
            
            # Determine query type
            if any(word in query_lower for word in ['who', 'what', 'when', 'where']):
                query_type = 'factual'
            elif any(word in query_lower for word in ['why', 'how', 'explain']):
                query_type = 'analytical'
            elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
                query_type = 'comparison'
            elif is_list_query:
                query_type = 'list'
            else:
                query_type = 'general'
            
            result = {
                'type': query_type,
                'complexity': complexity,
                'requires_decomposition': has_multiple_parts and ENABLE_QUERY_DECOMPOSITION,
                'requires_hyde': (complexity == 'complex' or HYDE_FOR_SIMPLE_QUERIES) and ENABLE_HYDE,
                'requires_variations': ENABLE_QUERY_VARIATIONS,
                'is_list_query': is_list_query,
                'word_count': word_count
            }
            
            logger.info(f"Classification: {result['type']} / {result['complexity']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return {
                'type': 'unknown',
                'complexity': 'simple',
                'requires_decomposition': False,
                'requires_hyde': False,
                'requires_variations': False,
                'is_list_query': False,
                'error': str(e)
            }
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex multi-part query into atomic sub-queries.
        
        Args:
            query: Complex query to decompose
            
        Returns:
            List of sub-queries
        """
        if not ENABLE_QUERY_DECOMPOSITION:
            logger.info("Query decomposition disabled")
            return [query]
        
        logger.info(f"Decomposing query: '{query[:50]}...'")
        
        try:
            prompt = f"""Break this complex question into simple, atomic sub-questions.
Each sub-question should ask about ONE specific thing.

COMPLEX QUESTION:
{query}

INSTRUCTIONS:
- Split into 2-{MAX_SUBQUERIES} simple questions
- Each question should be complete and standalone
- Maintain the original context
- Number each question (1., 2., etc.)

SUB-QUESTIONS:"""

            response = self.client.chat.completions.create(
               model=QUERY_ENHANCEMENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=QUERY_ENHANCEMENT_TEMPERATURE,
                max_tokens=QUERY_ENHANCEMENT_MAX_TOKENS
            )
            
            decomposition_text = response.choices[0].message.content.strip()
            
            # Parse numbered list
            sub_queries = []
            for line in decomposition_text.split('\n'):
                line = line.strip()
                # Match patterns like "1. Question?" or "1) Question?"
                match = re.match(r'^\d+[\.)]\s*(.+)$', line)
                if match:
                    sub_query = match.group(1).strip()
                    if sub_query and len(sub_query) > 5:
                        sub_queries.append(sub_query)
            
            if not sub_queries:
                # Fallback: return original
                logger.warning("No sub-queries parsed, returning original")
                return [query]
            
            # Limit to MAX_SUBQUERIES
            sub_queries = sub_queries[:MAX_SUBQUERIES]
            
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            for i, sq in enumerate(sub_queries, 1):
                logger.info(f"  [{i}] {sq}")
            
            return sub_queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]  # Fallback to original
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate hypothetical answer using HyDE technique.
        
        This creates a "fake" answer that helps retrieval by providing
        a document-like text to embed and search for.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical answer text
        """
        if not ENABLE_HYDE:
            logger.info("HyDE disabled")
            return query
        
        logger.info(f"Generating HyDE for: '{query[:50]}...'")
        
        try:
            prompt = f"""Generate a detailed, hypothetical answer to this medical legal question as if you were writing a medical report excerpt.

QUESTION:
{query}

INSTRUCTIONS:
- Write 2-3 sentences
- Use formal medical/legal language
- Include specific details (dates, names, conditions)
- Write as if this text came from an actual medical record or legal document
- Do NOT use phrases like "The answer is..." or "Based on..."

HYPOTHETICAL DOCUMENT EXCERPT:"""

            response = self.client.chat.completions.create(
               model=QUERY_ENHANCEMENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=HYDE_TEMPERATURE,  # Higher for creativity
                max_tokens=HYDE_MAX_TOKENS
            )
            
            hyde_text = response.choices[0].message.content.strip()
            
            logger.info(f"HyDE generated ({len(hyde_text)} chars)")
            logger.debug(f"HyDE text: {hyde_text[:100]}...")
            
            return hyde_text
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query  # Fallback to original
    
    def generate_query_variations(
        self,
        query: str,
        max_variations: int = MAX_QUERY_VARIATIONS
    ) -> List[str]:
        """
        Generate alternative phrasings of the query.
        
        Args:
            query: Original query
            max_variations: Number of variations to generate
            
        Returns:
            List of query variations (includes original)
        """
        if not ENABLE_QUERY_VARIATIONS:
            logger.info("Query variations disabled")
            return [query]
        
        logger.info(f"Generating {max_variations} variations for: '{query[:50]}...'")
        
        try:
            if QUERY_VARIATION_METHOD == "template":
                # Simple template-based variations (fast, free)
                return self._generate_template_variations(query, max_variations)
            else:
                # LLM-based variations (better quality, costs)
                return self._generate_llm_variations(query, max_variations)
                
        except Exception as e:
            logger.error(f"Query variation generation failed: {e}")
            return [query]
    
    def _generate_template_variations(self, query: str, max_variations: int) -> List[str]:
        """Generate variations using simple templates."""
        variations = [query]  # Always include original
        
        query_lower = query.lower()
        
        # Simple synonym replacements
        replacements = {
            'allegations': ['claims', 'accusations'],
            'defendants': ['accused parties', 'respondents'],
            'plaintiff': ['claimant', 'complainant'],
            'medical condition': ['diagnosis', 'health condition'],
            'treatment': ['care', 'medical intervention']
        }
        
        for original, synonyms in replacements.items():
            if original in query_lower:
                for synonym in synonyms[:max_variations-1]:
                    variation = query.lower().replace(original, synonym)
                    variations.append(variation.capitalize())
                    if len(variations) >= max_variations:
                        break
                break
        
        return variations[:max_variations]
    
    def _generate_llm_variations(self, query: str, max_variations: int) -> List[str]:
        """Generate variations using LLM."""
        prompt = f"""Generate {max_variations - 1} alternative phrasings of this question. Keep the same meaning but use different words.

ORIGINAL QUESTION:
{query}

INSTRUCTIONS:
- Use synonyms and different sentence structures
- Keep questions focused and concise
- Number each variation (1., 2., etc.)
- Don't add new information or change the intent

ALTERNATIVE PHRASINGS:"""

        try:
            response = self.client.chat.completions.create(
               model=QUERY_ENHANCEMENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Higher for diversity
                max_tokens=200
            )
            
            variations_text = response.choices[0].message.content.strip()
            
            # Parse variations
            variations = [query]  # Always include original first
            for line in variations_text.split('\n'):
                line = line.strip()
                match = re.match(r'^\d+[\.)]\s*(.+)$', line)
                if match:
                    variation = match.group(1).strip()
                    if variation and variation != query:
                        variations.append(variation)
            
            return variations[:max_variations]
            
        except Exception as e:
            logger.error(f"LLM variation generation failed: {e}")
            return [query]
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Full query enhancement pipeline.
        
        Args:
            query: Original user query
            
        Returns:
            Dict with enhanced query components
        """
        logger.info(f"Enhancing query: '{query[:50]}...'")
        
        # Step 1: Classify
        classification = self.classify_query(query)
        
        result = {
            'original_query': query,
            'classification': classification,
            'sub_queries': [query],
            'hyde_text': None,
            'variations': [query],
            'enhanced': False
        }
        
        # Step 2: Decompose if needed
        if classification.get('requires_decomposition', False):
            result['sub_queries'] = self.decompose_query(query)
            result['enhanced'] = True
        
        # Step 3: Generate HyDE if needed
        if classification.get('requires_hyde', False):
            result['hyde_text'] = self.generate_hypothetical_answer(query)
            result['enhanced'] = True
        
        # Step 4: Generate variations if needed
        if classification.get('requires_variations', False):
            result['variations'] = self.generate_query_variations(query)
            result['enhanced'] = True
        
        logger.info(f"Query enhancement complete. Enhanced: {result['enhanced']}")
        
        return result
    
    def get_query_stats(self) -> Dict[str, Any]:
        """
        Get query enhancement configuration and statistics.
        
        Returns:
            Statistics dict
        """
        return {
            'classification_enabled': ENABLE_QUERY_CLASSIFICATION,
            'decomposition_enabled': ENABLE_QUERY_DECOMPOSITION,
            'hyde_enabled': ENABLE_HYDE,
            'variations_enabled': ENABLE_QUERY_VARIATIONS,
            'max_subqueries': MAX_SUBQUERIES,
            'max_variations': MAX_QUERY_VARIATIONS,
            'variation_method': QUERY_VARIATION_METHOD
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def enhance_query(query: str) -> Dict[str, Any]:
    """
    Convenience function for quick query enhancement.
    
    Args:
        query: User query
        
    Returns:
        Enhanced query components
    """
    tools = QueryTools()
    return tools.enhance_query(query)

# ============================================================================
# CrewAI Tool Wrappers
# ============================================================================

@tool("Classify Query")
def classify_query_tool(query: str) -> dict:
    """
    Classify query type and complexity to determine enhancement strategy.
    
    Returns classification with recommendations for decomposition, HyDE, and variations.
    
    Args:
        query: User question to classify
        
    Returns:
        dict: {type, complexity, requires_decomposition, requires_hyde, 
               requires_variations, is_list_query}
    """
    tools = QueryTools()
    return tools.classify_query(query)


@tool("Decompose Complex Query")
def decompose_query_tool(query: str, max_subqueries: int = 3, force: bool = False) -> dict:
    """
    Decompose complex query into subqueries.
    
    Args:
        query: User query
        max_subqueries: Max subqueries to generate
        force: If False (default), skip decomposition. If True, run decomposition.
    
    Returns:
        dict: {strategy, subqueries, original_query, skipped}
    """
    # Skip if not forced (default behavior)
    if not force:
        logger.info("⏭️ Query decomposition skipped (force=False)")
        return {
            'strategy': 'single',
            'subqueries': [query],
            'original_query': query,
            'skipped': True
        }
    
    # force=True: Run actual decomposition
    logger.info(f"🔧 Query decomposition forced (force=True)")
    tools = QueryTools()
    sub_queries = tools.decompose_query(query)
    
    return {
        'strategy': 'decomposed' if len(sub_queries) > 1 else 'single',
        'subqueries': sub_queries,
        'original_query': query,
        'count': len(sub_queries),
        'skipped': False
    }


@tool("Generate Hypothetical Answer (HyDE)")
def generate_hypothetical_answer_tool(query: str, force: bool = False) -> dict:
    """
    Generate hypothetical answer for query expansion (HyDE).
    
    Args:
        query: User query
        force: If False (default), skip HyDE. If True, generate hypothetical answer.
    
    Returns:
        dict: {query, hypothetical_answer, expanded_query, skipped}
    """
    # Skip if not forced
    if not force:
        logger.info("⏭️ HyDE skipped (force=False)")
        return {
            'query': query,
            'hypothetical_answer': None,
            'expanded_query': query,
            'skipped': True
        }
    
    # force=True: Run actual HyDE generation
    logger.info(f"🔧 HyDE generation forced (force=True)")
    tools = QueryTools()
    hyde_text = tools.generate_hypothetical_answer(query)
    
    return {
        'query': query,
        'hypothetical_answer': hyde_text,
        'expanded_query': f"{query}\n\n{hyde_text}",
        'skipped': False
    }


@tool("Generate Query Variations")
def generate_query_variations_tool(query: str, num_variations: int = 2, force: bool = False) -> dict:
    """
    Generate alternative query phrasings.
    
    Args:
        query: User query
        num_variations: Number of variations
        force: If False (default), skip variations. If True, generate variations.
    
    Returns:
        dict: {original_query, variations, all_queries, skipped}
    """
    # Skip if not forced
    if not force:
        logger.info("⏭️ Query variations skipped (force=False)")
        return {
            'original_query': query,
            'variations': [],
            'all_queries': [query],
            'skipped': True
        }
    
    # force=True: Run actual variations generation
    logger.info(f"🔧 Query variations forced (force=True)")
    tools = QueryTools()
    variations = tools.generate_query_variations(query, num_variations)
    
    return {
        'original_query': query,
        'variations': variations[1:],  # Exclude original (first item)
        'all_queries': variations,
        'count': len(variations),
        'skipped': False
    }


@tool("Enhance Query (Full Pipeline)")
def enhance_query_tool(query: str) -> dict:
    """
    Complete query enhancement pipeline: classify, decompose, HyDE, variations.
    This is the main entry point for query enhancement.
    
    Args:
        query: User question
    
    Returns:
        dict: Enhanced query components with all enhancements applied
    """
    tools = QueryTools()
    return tools.enhance_query(query)
