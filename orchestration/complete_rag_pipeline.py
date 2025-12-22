"""
================================================================================
COMPLETE RAG PIPELINE WITH SELF-HEALING
================================================================================
Purpose: Orchestrate all 10 agents into a complete medical RAG system

Agent Flow:
1. Cache Agent      → Check semantic cache for similar queries
2. Memory Agent     → Resolve references and track conversation
3. Query Agent      → Classify, enhance, decompose queries
4. Retrieval Agent  → Search knowledge base with multiple strategies
5. Reranking Agent  → Optimize relevance of retrieved chunks
6. Answer Agent     → Generate well-cited answer from context
7. Validation Agent → Validate quality and detect hallucinations
8. Self-Healer Agent → Fix issues if validation fails (max 2 retries)
9. (Re-validation loop if healing applied)
10. Final Answer    → Return to user or cache

Configuration:
- Semantic caching: ENABLE_SEMANTIC_CACHE
- Self-healing: ENABLE_SELF_HEALING (max SELF_HEAL_MAX_RETRIES)
- Validation threshold: SELF_HEAL_MIN_QUALITY_SCORE

Run:
    from orchestration.complete_rag_pipeline import run_rag_query
    result = run_rag_query("What medications were prescribed?")
================================================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from crewai import Crew, Task

# Import all agents
from agents.cache_agent import create_cache_agent
from agents.memory_agent import create_memory_agent
from agents.query_agent import create_query_agent
from agents.retrieval_agent import create_retrieval_agent
from agents.reranking_agent import create_reranking_agent
from agents.answer_agent import create_answer_agent
from agents.validation_agent import create_validation_agent
from agents.self_healer_agent import create_self_healer_agent

# Import settings
from config.settings import (
    ENABLE_SEMANTIC_CACHE,
    ENABLE_SELF_HEALING,
    SELF_HEAL_MAX_RETRIES,
    SELF_HEAL_MIN_QUALITY_SCORE,
    ENABLE_VALIDATION
)

logger = logging.getLogger(__name__)


class CompleteRAGPipeline:
    """
    Complete RAG pipeline orchestrating all 10 agents with self-healing.
    """
    
    def __init__(
        self,
        session_id: str = None,
        verbose: bool = True,
        enable_cache: bool = None,
        enable_self_healing: bool = None
    ):
        """
        Initialize RAG Pipeline.
        
        Args:
            session_id: Conversation session ID for memory tracking
            verbose: Enable verbose logging
            enable_cache: Override ENABLE_SEMANTIC_CACHE setting
            enable_self_healing: Override ENABLE_SELF_HEALING setting
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.verbose = verbose
        self.enable_cache = enable_cache if enable_cache is not None else ENABLE_SEMANTIC_CACHE
        self.enable_self_healing = enable_self_healing if enable_self_healing is not None else ENABLE_SELF_HEALING
        
        # Initialize agents
        logger.info("Initializing RAG Pipeline agents...")
        self._initialize_agents()
        
        # Track statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'healing_attempts': 0,
            'healing_successes': 0,
            'total_queries': 0
        }
    
    def _initialize_agents(self):
        """Initialize all agents."""
        self.cache_agent = create_cache_agent(verbose=self.verbose) if self.enable_cache else None
        self.memory_agent = create_memory_agent(verbose=self.verbose)
        self.query_agent = create_query_agent(verbose=self.verbose)
        self.retrieval_agent = create_retrieval_agent(verbose=self.verbose)
        self.reranking_agent = create_reranking_agent(verbose=self.verbose)
        self.answer_agent = create_answer_agent(verbose=self.verbose)
        self.validation_agent = create_validation_agent(verbose=self.verbose) if ENABLE_VALIDATION else None
        self.self_healer_agent = create_self_healer_agent(verbose=self.verbose) if self.enable_self_healing else None
        
        logger.info(f"✅ All agents initialized (session: {self.session_id})")
    
    def run_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run complete RAG pipeline with self-healing.
        
        Args:
            query: User query
            conversation_history: Previous conversation turns
            top_k: Number of chunks to retrieve
            
        Returns:
            {
                'answer': str,
                'sources': list,
                'validation': dict,
                'metadata': dict with stats
            }
        """
        self.stats['total_queries'] += 1
        start_time = datetime.now()
        
        logger.info(f"Processing query: {query}")
        
        try:
            # ================================================================
            # PHASE 1: CACHE CHECK
            # ================================================================
            if self.enable_cache and self.cache_agent:
                cache_result = self._check_cache(query)
                if cache_result['cached']:
                    self.stats['cache_hits'] += 1
                    logger.info("✅ Cache hit - returning cached answer")
                    return cache_result
                else:
                    self.stats['cache_misses'] += 1
            
            # ================================================================
            # PHASE 2: MEMORY & QUERY ENHANCEMENT
            # ================================================================
            enhanced_query, memory_context = self._enhance_query_with_memory(
                query, conversation_history
            )
            
            # ================================================================
            # PHASE 3: RETRIEVAL & RERANKING
            # ================================================================
            retrieved_chunks = self._retrieve_and_rerank(enhanced_query, top_k)
            
            # ================================================================
            # PHASE 4: ANSWER GENERATION
            # ================================================================
            answer_result = self._generate_answer(query, retrieved_chunks, memory_context)
            
            # ================================================================
            # PHASE 5: VALIDATION & SELF-HEALING LOOP
            # ================================================================
            if ENABLE_VALIDATION and self.validation_agent:
                final_result = self._validate_and_heal(
                    answer_result,
                    query,
                    retrieved_chunks,
                    memory_context
                )
            else:
                final_result = answer_result
            
            # ================================================================
            # PHASE 6: CACHE STORAGE
            # ================================================================
            if self.enable_cache and self.cache_agent:
                self._store_in_cache(query, final_result)
            
            # ================================================================
            # PHASE 7: MEMORY UPDATE
            # ================================================================
            self._update_memory(query, final_result)
            
            # Add metadata
            end_time = datetime.now()
            final_result['metadata'] = {
                'session_id': self.session_id,
                'processing_time_ms': (end_time - start_time).total_seconds() * 1000,
                'cache_hit': False,
                'healing_applied': final_result.get('healing_applied', False),
                'retry_count': final_result.get('retry_count', 0),
                'stats': self.stats.copy()
            }
            
            logger.info(f"✅ Query processed successfully in {final_result['metadata']['processing_time_ms']:.0f}ms")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'validation': {'is_valid': False, 'error': str(e)},
                'metadata': {'error': str(e)}
            }
    
    def _check_cache(self, query: str) -> Dict[str, Any]:
        """Check semantic cache for similar queries."""
        task = Task(
            description=f"Check cache for query: '{query}'",
            agent=self.cache_agent,
            expected_output="Cache result with answer if found"
        )
        
        crew = Crew(
            agents=[self.cache_agent],
            tasks=[task],
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        return result if isinstance(result, dict) else {'cached': False}
    
    def _enhance_query_with_memory(
        self,
        query: str,
        conversation_history: Optional[List[Dict]]
    ) -> tuple:
        """Enhance query using memory and query agent."""
        # Memory context
        memory_task = Task(
            description=(
                f"Session: {self.session_id}\n"
                f"Query: {query}\n"
                f"Conversation history: {conversation_history or []}\n"
                "Resolve references and provide memory context"
            ),
            agent=self.memory_agent,
            expected_output="Resolved query with memory context"
        )
        
        # Query enhancement
        query_task = Task(
            description=(
                f"Enhance query: {query}\n"
                "Classify query type, decompose if complex, generate variations"
            ),
            agent=self.query_agent,
            expected_output="Enhanced query with variations and classification"
        )
        
        crew = Crew(
            agents=[self.memory_agent, self.query_agent],
            tasks=[memory_task, query_task],
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        
        # Extract enhanced query and memory context
        enhanced_query = query  # Fallback to original
        memory_context = {}
        
        if isinstance(result, dict):
            enhanced_query = result.get('enhanced_query', query)
            memory_context = result.get('memory_context', {})
        
        return enhanced_query, memory_context
    
    def _retrieve_and_rerank(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve and rerank relevant chunks."""
        # Retrieval
        retrieval_task = Task(
            description=f"Retrieve top {top_k} chunks for: {query}",
            agent=self.retrieval_agent,
            expected_output="Retrieved chunks with scores"
        )
        
        # Reranking
        reranking_task = Task(
            description="Rerank retrieved chunks using cross-encoder or LLM",
            agent=self.reranking_agent,
            expected_output="Reranked chunks optimized for relevance"
        )
        
        crew = Crew(
            agents=[self.retrieval_agent, self.reranking_agent],
            tasks=[retrieval_task, reranking_task],
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        
        # Extract chunks
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and 'chunks' in result:
            return result['chunks']
        else:
            return []

        # ✅ ADD THIS CHECK
        logger.info(f"Retrieved {len(chunks)} chunks for query: {query}")
        if not chunks:
            logger.error("❌ No chunks retrieved! Check retrieval agent and vector store.")
        
        return chunks
    
    def _generate_answer(
        self,
        query: str,
        chunks: List[Dict],
        memory_context: Dict
    ) -> Dict[str, Any]:
        """Generate answer from chunks."""
        task = Task(
            description=(
                f"Generate answer for: {query}\n\n"
                f"Context chunks: {len(chunks)} chunks provided\n"
                f"Memory context: {memory_context}\n\n"
                "Requirements:\n"
                "- Cite every factual claim with [doc:X]\n"
                "- Only use information from provided chunks\n"
                "- Answer format should match question type"
            ),
            agent=self.answer_agent,
            expected_output="Well-cited answer grounded in context"
        )
        
        crew = Crew(
            agents=[self.answer_agent],
            tasks=[task],
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        
        if isinstance(result, str):
            return {
                'answer': result,
                'sources': chunks,
                'validation': None
            }
        elif isinstance(result, dict):
            return result
        else:
            return {
                'answer': str(result),
                'sources': chunks,
                'validation': None
            }
    
    def _validate_and_heal(
        self,
        answer_result: Dict,
        query: str,
        chunks: List[Dict],
        memory_context: Dict
    ) -> Dict[str, Any]:
        """
        Validate answer and apply self-healing if needed.
        
        Implements retry loop:
        1. Validate answer
        2. If fails and healing enabled → heal
        3. Re-validate
        4. Repeat up to SELF_HEAL_MAX_RETRIES times
        5. Return best attempt if all fail
        """
        answer = answer_result.get('answer', '')
        retry_count = 0
        # best_attempt = None
        # ✅ Initialize with current answer as fallback
        best_attempt = {
            'answer': answer,
            'validation': {'is_valid': False, 'quality_score': 0.0},
            'retry_count': 0
        }
        best_score = 0.0
        healing_applied = False
        
        while retry_count <= SELF_HEAL_MAX_RETRIES:
            # Validation
            validation_result = self._validate_answer(answer, query, chunks)
            
            quality_score = validation_result.get('quality_score', 0.0)
            is_valid = validation_result.get('is_valid', False)
            
            logger.info(
                f"Validation attempt {retry_count + 1}: "
                f"is_valid={is_valid}, quality_score={quality_score:.2f}"
            )
            
            # Track best attempt
            if quality_score > best_score:
                best_score = quality_score
                best_attempt = {
                    'answer': answer,
                    'validation': validation_result,
                    'retry_count': retry_count
                }
            
            # Check if validation passes
            if is_valid and quality_score >= SELF_HEAL_MIN_QUALITY_SCORE:
                logger.info(f"✅ Validation passed (score: {quality_score:.2f})")
                return {
                    'answer': answer,
                    'sources': chunks,
                    'validation': validation_result,
                    'healing_applied': healing_applied,
                    'retry_count': retry_count
                }
            
            # Check if we should retry
            # Check if we should retry
            if retry_count >= SELF_HEAL_MAX_RETRIES:
                logger.warning(
                    f"⚠️ Max retries reached. Returning best attempt "
                    f"(score: {best_score:.2f})"
                )
                self.stats['healing_attempts'] += 1
                
                # ✅ Check if best_attempt is None
                if best_attempt is None:
                    logger.error("No valid attempts. Returning original answer.")
                    return {
                        'answer': answer,
                        'sources': chunks,
                        'validation': {'is_valid': False, 'quality_score': 0.0},
                        'healing_applied': healing_applied,
                        'best_attempt': True,
                        'retry_count': retry_count
                    }
                
                return {
                    **best_attempt,
                    'sources': chunks,
                    'healing_applied': healing_applied,
                    'best_attempt': True
                }

            
            # Apply self-healing
            if self.enable_self_healing and self.self_healer_agent:
                logger.info(f"🔧 Applying self-healing (attempt {retry_count + 1})...")
                self.stats['healing_attempts'] += 1
                
                healed_result = self._apply_self_healing(
                    answer,
                    query,
                    chunks,
                    validation_result,
                    memory_context
                )
                
                answer = healed_result.get('answer', answer)
                healing_applied = True
                retry_count += 1
            else:
                # No healing available, return current result
                logger.warning("⚠️ Validation failed but healing disabled")
                return {
                    'answer': answer,
                    'sources': chunks,
                    'validation': validation_result,
                    'healing_applied': False,
                    'retry_count': 0
                }
        
        # Should not reach here, but return best attempt as fallback
        return {
            **best_attempt,
            'sources': chunks,
            'healing_applied': healing_applied
        }
    
    def _validate_answer(
        self,
        answer: str,
        query: str,
        chunks: List[Dict]
    ) -> Dict[str, Any]:
        """Validate answer quality."""
        task = Task(
            description=(
                f"Validate answer:\n"
                f"Query: {query}\n"
                f"Answer: {answer}\n"
                f"Context chunks: {len(chunks)} chunks\n\n"
                "Check: citations, hallucinations, completeness, quality"
            ),
            agent=self.validation_agent,
            expected_output="Validation result with is_valid, quality_score, issues"
        )
        
        crew = Crew(
            agents=[self.validation_agent],
            tasks=[task],
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        
        if isinstance(result, dict):
            return result
        else:
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'error': 'Invalid validation result'
            }
    
    def _apply_self_healing(
        self,
        answer: str,
        query: str,
        chunks: List[Dict],
        validation_result: Dict,
        memory_context: Dict
    ) -> Dict[str, Any]:
        """Apply self-healing to improve answer."""
        task = Task(
            description=(
                f"Self-healing task:\n\n"
                f"VALIDATION RESULT:\n{validation_result}\n\n"
                f"ORIGINAL ANSWER:\n{answer}\n\n"
                f"QUERY:\n{query}\n\n"
                f"CONTEXT CHUNKS:\n{len(chunks)} chunks available\n\n"
                f"MEMORY CONTEXT:\n{memory_context}\n\n"
                "Analyze issues and apply appropriate corrective action:\n"
                "- Missing citations → Regenerate with citation emphasis\n"
                "- Hallucination → Regenerate with stricter grounding\n"
                "- Incomplete → Retrieve more context\n"
                "- Format issue → Match question type\n"
                "- Low confidence → Rephrase query and retry\n\n"
                "Return improved answer."
            ),
            agent=self.self_healer_agent,
            expected_output="Improved answer with healing metadata"
        )
        
        crew = Crew(
            agents=[self.self_healer_agent],
            tasks=[task],
            verbose=self.verbose
        )
        
        result = crew.kickoff()
        
        if isinstance(result, str):
            self.stats['healing_successes'] += 1
            return {'answer': result}
        elif isinstance(result, dict):
            self.stats['healing_successes'] += 1
            return result
        else:
            return {'answer': answer}  # Fallback
    
    def _store_in_cache(self, query: str, result: Dict):
        """Store result in semantic cache."""
        task = Task(
            description=f"Cache query '{query}' with answer: {result.get('answer', '')[:100]}...",
            agent=self.cache_agent,
            expected_output="Cache storage confirmation"
        )
        
        crew = Crew(
            agents=[self.cache_agent],
            tasks=[task],
            verbose=False  # Don't clutter logs
        )
        
        crew.kickoff()
    
    def _update_memory(self, query: str, result: Dict):
        """Update conversation memory."""
        task = Task(
            description=(
                f"Update memory for session {self.session_id}:\n"
                f"Query: {query}\n"
                f"Answer: {result.get('answer', '')}"
            ),
            agent=self.memory_agent,
            expected_output="Memory updated"
        )
        
        crew = Crew(
            agents=[self.memory_agent],
            tasks=[task],
            verbose=False  # Don't clutter logs
        )
        
        crew.kickoff()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.stats.copy()


# ============================================================================
# Convenience Functions
# ============================================================================

def run_rag_query(
    query: str,
    session_id: str = None,
    conversation_history: List[Dict] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run a single RAG query.
    
    Args:
        query: User query
        session_id: Session ID for memory tracking
        conversation_history: Previous conversation turns
        verbose: Enable verbose logging
        
    Returns:
        Result dict with answer, sources, validation, metadata
        
    Example:
        >>> result = run_rag_query("What medications were prescribed?")
        >>> print(result['answer'])
        >>> print(f"Quality score: {result['validation']['quality_score']}")
    """
    pipeline = CompleteRAGPipeline(
        session_id=session_id,
        verbose=verbose
    )
    
    return pipeline.run_query(query, conversation_history)


def create_rag_session(session_id: str = None, verbose: bool = False) -> CompleteRAGPipeline:
    """
    Create a reusable RAG pipeline for a conversation session.
    
    Args:
        session_id: Session ID for memory tracking
        verbose: Enable verbose logging
        
    Returns:
        CompleteRAGPipeline instance
        
    Example:
        >>> pipeline = create_rag_session("user123")
        >>> result1 = pipeline.run_query("What is the diagnosis?")
        >>> result2 = pipeline.run_query("What treatment was given?")  # Uses memory
        >>> stats = pipeline.get_stats()
    """
    return CompleteRAGPipeline(session_id=session_id, verbose=verbose)
