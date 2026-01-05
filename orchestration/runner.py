"""
CLEAN RAG ORCHESTRATOR - Efficient 2-Agent Validation & Healing
Purpose: Orchestrate RAG pipeline with external validation and healing
Architecture:
- Generation → Validation Agent → (HEAL) → Self Healing Agent → Return
- NO inline validation logic
- Clean separation of concerns
Agents:
1. Validation Agent - External scorer (Gemini/Azure)
2. Self Healing Agent - External fixer (8 strategies)
Performance:
- Fast path (ACCEPT): ~800ms (60-70% of queries)
- Healing path: ~3-5s (20-30% of queries)
- Average: ~1.5-2s (vs 25s previously)
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Import agents
from agents.validation_agent import create_validation_agent
from agents.self_healing_agent import create_self_healing_agent

# Import tools
from tools.cache_tools import CacheTools
from tools.retrieval_tools import RetrievalTools
from tools.reranking_tools import get_reranker
from tools.answer_generation_tools import AnswerGenerationTools
from tools.kb_tools import KBTools

# Import settings
from config.settings import (
    ENABLE_SEMANTIC_CACHE,
    ENABLE_SELF_HEALING,
    VALIDATION_SCORE_THRESHOLD,
    HEALING_MAX_RETRIES,
    VALIDATOR_LLM
)

logger = logging.getLogger(__name__)


class CleanRAGOrchestrator:
    """
    Clean orchestrator for RAG pipeline with external validation & healing.
    
    Features:
    - External Validation Agent (no inline logic)
    - External Self Healing Agent (8 strategies)
    - Clean workflow: Generate → Validate → Heal (if needed)
    - Fast path for good answers (800ms)
    - Healing path for medium answers (3-5s)
    """
    
    def __init__(
        self,
        session_id: str = None,
        enable_cache: bool = ENABLE_SEMANTIC_CACHE,
        enable_healing: bool = ENABLE_SELF_HEALING,
        validator_llm: str = VALIDATOR_LLM
    ):
        """
        Initialize orchestrator.
        
        Args:
            session_id: Session ID for tracking
            enable_cache: Enable semantic caching
            enable_healing: Enable self-healing
            validator_llm: "gemini" or "azure" for validation
        """
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_cache = enable_cache
        self.enable_healing = enable_healing
        
        logger.info(f"Initializing CleanRAGOrchestrator (session: {self.session_id})...")
        
        # Initialize tools
        self.cache = CacheTools() if enable_cache else None
        self.retrieval = RetrievalTools()
        self.answer_gen = AnswerGenerationTools()
        self.reranker = get_reranker()
        
        # Initialize agents
        self.validation_agent = create_validation_agent(llm_provider=validator_llm)
        self.healing_agent = create_self_healing_agent() if enable_healing else None
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_accepts": 0,
            "healing_attempts": 0,
            "healing_successes": 0
        }
        
        logger.info(f"✓ Orchestrator ready (validator: {validator_llm}, healing: {enable_healing})")
    
    def process_query(
        self,
        query: str,
        top_k: int = 5,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process query through RAG pipeline.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            conversation_history: Previous turns (optional)
        
        Returns:
            {
                "answer": "final answer text",
                "sources": [...],
                "validation": {...},
                "metadata": {
                    "path": "fast" or "healed",
                    "latency_ms": 1500,
                    "score": 35,
                    ...
                }
            }
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        logger.info(f"Processing query: {query[:50]}...")
        
        try:
            # ============================================================
            # STEP 1: Check Cache
            # ============================================================
            if self.enable_cache and self.cache:
                cached = self.cache.get_cached_answer(query)
                if cached:
                    self.stats["cache_hits"] += 1
                    logger.info(f"✓ Cache HIT (similarity: {cached.get('similarity', 0):.3f})")
                    cached["metadata"] = {
                        "path": "cache",
                        "latency_ms": (time.time() - start_time) * 1000,
                        "cached": True
                    }
                    return cached
                else:
                    self.stats["cache_misses"] += 1
                    logger.info("✗ Cache MISS")
            
            # ============================================================
            # STEP 2: Retrieve & Rerank (ALWAYS)
            # ============================================================
            chunks = self._retrieve_and_rerank(query, top_k)
            
            if not chunks:
                logger.warning("No chunks retrieved!")
                return self._error_response(
                    "No relevant information found",
                    latency_ms=(time.time() - start_time) * 1000
                )
            
            # ============================================================
            # STEP 3: Generate Answer
            # ============================================================
            answer = self._generate_answer(query, chunks, conversation_history)
            
            # ============================================================
            # STEP 4: Validation Agent (External)
            # ============================================================
            validation_result = self.validation_agent.evaluate(
                query=query,
                answer=answer,
                chunks=chunks
            )
            
            score = validation_result.get("score", 0)
            decision = validation_result.get("decision", "HEAL")
            confidence = validation_result.get("confidence", 0.5)
            issues = validation_result.get("issues", [])
            
            logger.info(f"Validation: score={score}/40, decision={decision}, confidence={confidence:.2f}")
            
            # ============================================================
            # STEP 5: Decision - ACCEPT or HEAL
            # ============================================================
            if decision == "ACCEPT":
                # Fast path - answer is good!
                self.stats["validation_accepts"] += 1
                logger.info(f"✓ ACCEPT - Returning answer (latency: {(time.time() - start_time)*1000:.0f}ms)")
                
                result = {
                    "answer": answer,
                    "sources": chunks,
                    "validation": validation_result,
                    "metadata": {
                        "path": "fast_accept",
                        "latency_ms": (time.time() - start_time) * 1000,
                        "score": score,
                        "confidence": confidence,
                        "healing_applied": False
                    }
                }
                
                # Cache the result
                if self.enable_cache and self.cache:
                    self.cache.cache_answer(query, answer, chunks)
                
                return result
            
            # ============================================================
            # STEP 6: Self Healing Agent (External)
            # ============================================================
            if not self.enable_healing or not self.healing_agent:
                logger.warning("Healing disabled, returning original answer")
                return {
                    "answer": answer,
                    "sources": chunks,
                    "validation": validation_result,
                    "metadata": {
                        "path": "validation_only",
                        "latency_ms": (time.time() - start_time) * 1000,
                        "score": score,
                        "healing_applied": False,
                        "healing_disabled": True
                    }
                }
            
            # Apply healing
            logger.info(f"Applying healing for {len(issues)} issues...")
            self.stats["healing_attempts"] += 1
            
            healing_result = self.healing_agent.heal(
                query=query,
                answer=answer,
                chunks=chunks,
                issues=issues,
                validation_score=score
            )
            
            if healing_result.get("success"):
                healed_answer = healing_result.get("healed_answer", answer)
                strategy = healing_result.get("strategy_used", "unknown")
                self.stats["healing_successes"] += 1
                logger.info(f"✓ Healing SUCCESS (strategy: {strategy})")
                
                result = {
                    "answer": healed_answer,
                    "sources": chunks,
                    "validation": validation_result,
                    "metadata": {
                        "path": "healed",
                        "latency_ms": (time.time() - start_time) * 1000,
                        "original_score": score,
                        "healing_applied": True,
                        "strategy_used": strategy,
                        "issues_addressed": len(issues)
                    }
                }
                
                # Cache the healed result
                if self.enable_cache and self.cache:
                    self.cache.cache_answer(query, healed_answer, chunks)
                
                return result
            else:
                logger.warning("Healing FAILED, returning original answer")
                return {
                    "answer": answer,
                    "sources": chunks,
                    "validation": validation_result,
                    "metadata": {
                        "path": "healing_failed",
                        "latency_ms": (time.time() - start_time) * 1000,
                        "score": score,
                        "healing_applied": False,
                        "healing_error": healing_result.get("metadata", {}).get("error")
                    }
                }
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._error_response(
                f"Error processing query: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def _retrieve_and_rerank(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve and rerank chunks (ALWAYS runs)."""
        try:
            # 1. Retrieve more chunks than needed
            retrieval_result = self.retrieval.search(
                query=query,
                top_k=top_k + 2,  # Get extra for reranking
                max_distance=1.5
            )
            
            if not retrieval_result.get("success"):
                logger.error(f"Retrieval failed: {retrieval_result.get('error')}")
                return []
            
            chunks = retrieval_result.get("results", [])
            logger.info(f"Retrieved {len(chunks)} chunks from FAISS")
            
            if not chunks:
                return []
            
            # 2. Rerank with cross-encoder (ALWAYS)
            reranked = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=top_k,
                method="crossencoder"
            )
            
            logger.info(f"Reranked to top {len(reranked)} chunks")
            return reranked
            
        except Exception as e:
            logger.error(f"Retrieve & rerank failed: {e}", exc_info=True)
            return []
    
    def _generate_answer(
        self,
        query: str,
        chunks: List[Dict],
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate answer from chunks."""
        try:
            result = self.answer_gen.generate_answer_with_custom_context(
                query=query,
                context_chunks=chunks,
                conversation_history=conversation_history or [],
                temperature=0.3
            )
            
            answer = result.get("answer", "")
            logger.info(f"Answer generated ({len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"
    
    def _error_response(self, error_msg: str, latency_ms: float) -> Dict[str, Any]:
        """Format error response."""
        return {
            "answer": f"Error: {error_msg}",
            "sources": [],
            "validation": {
                "score": 0,
                "decision": "ERROR",
                "issues": [error_msg]
            },
            "metadata": {
                "path": "error",
                "latency_ms": latency_ms,
                "error": error_msg
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        
        # Calculate rates
        if stats["total_queries"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_queries"]
            stats["validation_accept_rate"] = stats["validation_accepts"] / stats["total_queries"]
            if stats["healing_attempts"] > 0:
                stats["healing_success_rate"] = stats["healing_successes"] / stats["healing_attempts"]
        
        return stats


# Convenience function
def run_rag_query(
    query: str,
    session_id: str = None,
    validator_llm: str = VALIDATOR_LLM
) -> Dict[str, Any]:
    """
    Run a single RAG query (convenience function).
    
    Args:
        query: User question
        session_id: Session ID (optional)
        validator_llm: "gemini" or "azure"
    
    Returns:
        Result dict with answer, validation, metadata
    
    Example:
        result = run_rag_query("Who is Frank Ward?", validator_llm="gemini")
        print(result["answer"])
        print(f"Score: {result['validation']['score']}/40")
        print(f"Latency: {result['metadata']['latency_ms']:.0f}ms")
    """
    orchestrator = CleanRAGOrchestrator(
        session_id=session_id,
        validator_llm=validator_llm
    )
    return orchestrator.process_query(query)


if __name__ == "__main__":
    """
    Interactive CLI for RAG system.
    
    Usage:
        python orchestration/main_runner_varun.py
    
    Commands:
        - Type your question
        - Type "quit", "exit", or "q" to exit
        - Type "stats" to see pipeline statistics
        - Type "clear" to clear cache
    """
    import sys
    
    # Configure logging for CLI (suppress verbose output)
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(levelname)s: %(message)s'
    )
    
    # Suppress noisy libraries
    logging.getLogger('crewai').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    
    print("="*70)
    print("CLEAN RAG SYSTEM - Interactive QA Interface")
    print("="*70)
    print("Type your question or 'quit' to exit")
    print("Commands: 'stats' (show statistics), 'clear' (clear cache)")
    print()
    
    # Initialize orchestrator
    print("Initializing RAG pipeline...")
    try:
        orchestrator = CleanRAGOrchestrator(
            session_id="cli_session",
            enable_cache=True,
            enable_healing=True,
            validator_llm=VALIDATOR_LLM  # From config
        )
        print(f"✓ Pipeline ready! (Validator: {VALIDATOR_LLM})")
        print()
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print("\nPlease check:")
        print("1. Knowledge base is built (run KB setup)")
        print("2. API keys are configured (Azure/Gemini)")
        print("3. Dependencies are installed")
        sys.exit(1)
    
    # Main interactive loop
    while True:
        try:
            # Get user input
            query = input("\n📝 Your question: ").strip()
            
            if not query:
                continue
            
            # Handle commands
    
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            # ===== ADD THIS HELP MENU =====
            if query.lower() in ['help', '?', 'h']:
                print("\n" + "="*70)
                print("AVAILABLE COMMANDS")
                print("="*70)
                print("\n📖 Query Commands:")
                print("  <your question>     Ask a question")
                print("  quit / exit / q     Exit the system")
                print()
                print("📊 Information Commands:")
                print("  help / h / ?        Show this help menu")
                print("  stats               Show pipeline statistics")
                print("  status              Show system status")
                print()
                print("🔧 Management Commands:")
                print("  clear               Clear semantic cache")
                print("  sync                Sync knowledge base with input folder")
                print("  rebuild             Rebuild knowledge base from scratch")
                print()
                print("⚙️  Toggle Commands:")
                print("  verbose on/off      Toggle verbose logging")
                print("  healing on/off      Toggle self-healing")
                print("  cache on/off        Toggle semantic cache")
                print()
                print("💡 Examples:")
                print("  Who is Frank Ward?")
                print("  What procedures did he undergo?")
                print("  stats")
                print("="*70)
                continue

            if query.lower() == 'status':
                print("\n" + "="*70)
                print("SYSTEM STATUS")
                print("="*70)
                print(f"✓ Validator: {VALIDATOR_LLM}")
                print(f"✓ Cache: {'Enabled' if orchestrator.enable_cache else 'Disabled'}")
                print(f"✓ Healing: {'Enabled' if orchestrator.enable_healing else 'Disabled'}")
                print(f"✓ Session ID: {orchestrator.session_id}")
                continue

            if query.lower() == 'sync':
                print("\n🔄 Syncing knowledge base...")
                try:
                    from tools.kb_tools import KBTools
                    kb = KBTools()
                    result = kb.sync_knowledge_base()
                    if result.get("success"):
                        print(f"✓ Sync complete: {result.get('message', 'Done')}")
                    else:
                        print(f"✗ Sync failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"✗ Sync error: {e}")
                continue

            if query.lower() == 'rebuild':
                confirm = input("⚠️  This will rebuild the entire KB. Continue? (yes/no): ")
                if confirm.lower() == 'yes':
                    print("\n🔨 Rebuilding knowledge base...")
                    try:
                        from tools.kb_tools import KBTools
                        kb = KBTools()
                        result = kb.build_knowledge_base(force_rebuild=True)
                        if result.get("success"):
                            print(f"✓ Rebuild complete!")
                        else:
                            print(f"✗ Rebuild failed: {result.get('error')}")
                    except Exception as e:
                        print(f"✗ Rebuild error: {e}")
                else:
                    print("Cancelled.")
                continue

            if query.lower().startswith('verbose '):
                setting = query.split()[1].lower()
                if setting == 'on':
                    logging.getLogger().setLevel(logging.INFO)
                    print("✓ Verbose logging enabled")
                elif setting == 'off':
                    logging.getLogger().setLevel(logging.WARNING)
                    print("✓ Verbose logging disabled")
                continue

            if query.lower().startswith('healing '):
                setting = query.split()[1].lower()
                if setting == 'on':
                    orchestrator.enable_healing = True
                    print("✓ Self-healing enabled")
                elif setting == 'off':
                    orchestrator.enable_healing = False
                    print("✓ Self-healing disabled")
                continue

            if query.lower().startswith('cache '):
                setting = query.split()[1].lower()
                if setting == 'on':
                    orchestrator.enable_cache = True
                    print("✓ Semantic cache enabled")
                elif setting == 'off':
                    orchestrator.enable_cache = False
                    print("✓ Semantic cache disabled")
                continue







            if query.lower() == 'stats':
                print("\n" + "="*70)
                print("PIPELINE STATISTICS")
                print("="*70)
                stats = orchestrator.get_stats()
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2%}" if "rate" in key else f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
                continue
            
            if query.lower() == 'clear':
                if orchestrator.cache:
                    orchestrator.cache.clear_cache()
                    print("✓ Cache cleared")
                else:
                    print("✗ Cache is disabled")
                continue
            
            # Process query
            print(f"\n{'─'*70}")
            print("Processing query...")
            start = time.time()
            
            result = orchestrator.process_query(query)
            
            elapsed = time.time() - start
            
            # Display answer
            print(f"{'─'*70}")
            print("ANSWER:")
            print(f"{'─'*70}")
            print(result["answer"])
            
            # Display metadata
            print(f"\n{'─'*70}")
            print("METADATA:")
            print(f"{'─'*70}")
            metadata = result.get("metadata", {})
            validation = result.get("validation", {})
            
            print(f"⏱️  Latency: {metadata.get('latency_ms', 0):.0f}ms")
            print(f"📊 Score: {validation.get('score', 0)}/40 ({validation.get('confidence', 0)*100:.0f}% confidence)")
            print(f"✓  Decision: {validation.get('decision', 'UNKNOWN')}")
            print(f"🛤️  Path: {metadata.get('path', 'unknown')}")
            
            if metadata.get('healing_applied'):
                print(f"🔧 Healing: {metadata.get('strategy_used', 'unknown')} ({metadata.get('issues_addressed', 0)} issues)")
            
            # Display sources
            sources = result.get("sources", [])
            if sources:
                print(f"\n📚 Sources: {len(sources)} chunks used")
            
            # Display issues if any
            issues = validation.get('issues', [])
            if issues and metadata.get('path') != 'fast_accept':
                print(f"\n⚠️  Issues detected:")
                for i, issue in enumerate(issues[:3], 1):  # Show max 3 issues
                    print(f"   {i}. {issue}")
                if len(issues) > 3:
                    print(f"   ... and {len(issues)-3} more")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            logger.error(f"CLI error: {e}", exc_info=True)
            continue
    
    # Show final stats
    print("\n" + "="*70)
    print("SESSION STATISTICS")
    print("="*70)
    stats = orchestrator.get_stats()
    print(f"Total queries: {stats.get('total_queries', 0)}")
    print(f"Cache hits: {stats.get('cache_hits', 0)}")
    print(f"Validation accepts: {stats.get('validation_accepts', 0)}")
    print(f"Healing attempts: {stats.get('healing_attempts', 0)}")
    print(f"Healing successes: {stats.get('healing_successes', 0)}")
    if stats.get('total_queries', 0) > 0:
        accept_rate = stats.get('validation_accepts', 0) / stats['total_queries'] * 100
        print(f"Accept rate: {accept_rate:.1f}%")
