"""
Answer Generation Tools - LLM Answer Specialist

Generates answers from retrieved context using Azure OpenAI.

Features:
- Context formatting (retrieved chunks → LLM prompt)
- Answer generation with conversation history
- Streaming support (future)
- Answer validation (future)

Integrates:
- RetrievalTools (gets relevant chunks)
- utils/azure_clients.py (LLM calls)
- utils/prompt_templates.py (prompts)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from tools.retrieval_tools import RetrievalTools
from utils.azure_clients import get_chat_completion
from utils.prompt_templates import (
    get_answer_generation_prompts,
    format_context_for_answer,
    format_conversation_history
)
from config.settings import (
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)

logger = logging.getLogger(__name__)


class AnswerGenerationTools:
    """
    Generates answers from retrieved context using LLM.
    
    Features:
    - RAG answer generation (query → retrieve → generate)
    - Context formatting
    - Conversation history support
    - Configurable LLM parameters
    """
    
    def __init__(self, retrieval_tools: Optional[RetrievalTools] = None):
        """
        Initialize answer generation tools.
        
        Args:
            retrieval_tools: RetrievalTools instance (optional, will create if None)
        """
        if retrieval_tools is None:
            retrieval_tools = RetrievalTools()
        
        self.retrieval = retrieval_tools
        
        logger.info("AnswerGenerationTools initialized")
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve context + generate answer.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            max_tokens: Max tokens to generate
            conversation_history: Previous Q&A exchanges
            include_sources: Include source citations in response
            
        Returns:
            {
                'success': bool,
                'query': str,
                'answer': str,
                'sources': list,  # Retrieved chunks used
                'metadata': dict,  # Generation metadata
                'error': str | None
            }
        """
        try:
            logger.info(f"Generating answer for: '{query[:50]}...'")
            
            # Step 1: Retrieve relevant context
            retrieval_result = self.retrieval.search(
                query=query,
                top_k=top_k,
                return_scores=True
            )
            
            if not retrieval_result['success']:
                return {
                    'success': False,
                    'query': query,
                    'error': f"Retrieval failed: {retrieval_result.get('error')}"
                }
            
            retrieved_chunks = retrieval_result['results']
            
            if not retrieved_chunks:
                return {
                    'success': False,
                    'query': query,
                    'error': 'No relevant context found'
                }
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            
            # Step 2: Format context for LLM
            formatted_context = format_context_for_answer(retrieved_chunks)
            
            # Step 3: Format conversation history (if provided)
            formatted_history = None
            if conversation_history:
                formatted_history = format_conversation_history(
                    exchanges=conversation_history,
                    max_exchanges=3  # Last 3 turns
                )
            
            # Step 4: Build prompts
            system_prompt, user_prompt = get_answer_generation_prompts(
                query=query,
                formatted_context=formatted_context,
                conversation_history=formatted_history
            )
            
            # Step 5: Generate answer with LLM
            logger.info("Calling LLM for answer generation...")
            
            answer = get_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(f"Answer generated ({len(answer)} chars)")
            
            # Step 6: Prepare response
            result = {
                'success': True,
                'query': query,
                'answer': answer,
                'metadata': {
                    'chunks_retrieved': len(retrieved_chunks),
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'had_conversation_history': conversation_history is not None
                }
            }
            
            # Add sources if requested
            if include_sources:
                result['sources'] = [
                    {
                        'text': chunk['text'][:200] + '...',  # Truncate for brevity
                        'source': chunk['metadata'].get('source', 'Unknown'),
                        'page': chunk['metadata'].get('page_numbers', 'N/A'),
                        'similarity': chunk.get('similarity', 'N/A')
                    }
                    for chunk in retrieved_chunks
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return {
                'success': False,
                'query': query,
                'error': f'Answer generation failed: {str(e)}'
            }
    
    def generate_answer_with_custom_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer from pre-retrieved context (skip retrieval step).
        
        Useful when you already have specific chunks to use.
        
        Args:
            query: User question
            context_chunks: Pre-retrieved chunks to use as context
            temperature: LLM temperature
            max_tokens: Max tokens
            conversation_history: Previous Q&A
            
        Returns:
            Answer result dict
        """
        try:
            logger.info(f"Generating answer with {len(context_chunks)} custom chunks")
            
            # Format context
            formatted_context = format_context_for_answer(context_chunks)
            
            # Format history
            formatted_history = None
            if conversation_history:
                formatted_history = format_conversation_history(
                    exchanges=conversation_history,
                    max_exchanges=3
                )
            
            # Build prompts
            system_prompt, user_prompt = get_answer_generation_prompts(
                query=query,
                formatted_context=formatted_context,
                conversation_history=formatted_history
            )
            
            # Generate answer
            answer = get_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                'success': True,
                'query': query,
                'answer': answer,
                'metadata': {
                    'chunks_used': len(context_chunks),
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return {
                'success': False,
                'query': query,
                'error': str(e)
            }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get answer generation statistics.
        
        Returns:
            Statistics dict
        """
        return {
            'tool_name': 'AnswerGenerationTools',
            'status': 'operational',
            'default_temperature': LLM_TEMPERATURE,
            'default_max_tokens': LLM_MAX_TOKENS,
            'retrieval_available': self.retrieval is not None
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def ask_question(
    query: str,
    top_k: int = 5,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Convenience function for simple RAG Q&A.
    
    Args:
        query: User question
        top_k: Number of chunks to retrieve
        conversation_history: Previous exchanges
        
    Returns:
        Answer result dict
    """
    answer_gen = AnswerGenerationTools()
    return answer_gen.generate_answer(
        query=query,
        top_k=top_k,
        conversation_history=conversation_history
    )
