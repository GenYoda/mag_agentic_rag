<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Agentic Medical RAG System - Complete Context \& Implementation Guide

## Document Purpose

This document contains all context, decisions, architecture, and plans needed to continue implementing the Agentic Medical RAG system. Feed this entire document to a new AI chat session to resume development seamlessly.

***

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current State](#current-state)
3. [System Architecture](#system-architecture)
4. [Complete Data Flows](#complete-data-flows)
5. [Agent Definitions](#agent-definitions)
6. [Tool Specifications](#tool-specifications)
7. [Critical Prompts (MUST PRESERVE)](#critical-prompts-must-preserve)
8. [Implementation Phases](#implementation-phases)
9. [Project Structure](#project-structure)
10. [Dependencies \& Environment](#dependencies--environment)
11. [Key Decisions \& Constraints](#key-decisions--constraints)
12. [Next Immediate Steps](#next-immediate-steps)

***

## 1. Project Overview

### 1.1 What We're Building

Converting an existing medical document RAG chatbot into a multi-agent system using CrewAI that:

- Preserves existing answer quality (CRITICAL: no regression)
- Adds semantic caching for performance
- Implements hash-based PDF tracking with deduplication
- Adds self-healing validation to detect and fix hallucinations
- Implements enhanced conversation memory with entity tracking
- Separates PDF extraction into dedicated agent
- Provides confidence scoring for all answers


### 1.2 Technology Stack

- **Framework**: CrewAI (multi-agent orchestration)
- **LLM**: Azure OpenAI (GPT-4 for chat, text-embedding-3-small for embeddings)
- **Document Processing**: Azure Document Intelligence (handles handwritten text)
- **Vector Store**: FAISS (CPU version)
- **Language**: Python 3.12
- **UI**: Streamlit (deferred to later phase)


### 1.3 Current System (Legacy)

- **KB Manager**: Builds FAISS index from PDFs via Azure Document Intelligence
- **Retriever**: Vector search with optional enhancements (query enhancement, distance filtering, reranking)
- **LLM Generator**: Azure OpenAI with specialized medical prompts (MUST PRESERVE EXACTLY)
- **Orchestrator**: Linear pipeline (retrieve → generate)
- **UI**: Streamlit chat interface


### 1.4 Key Requirements

1. **Zero answer quality regression** - preserve existing prompts exactly
2. **Keep Azure Document Intelligence** - it handles handwritten medical documents
3. **Deduplication at two levels**: PDF-level (content signature) and chunk-level (chunk signature)
4. **Latency control** - use semantic cache and conditional enhancements (HyDE only for complex queries)
5. **Backward compatibility** - keep classic mode as fallback

***

## 2. Current State

### 2.1 What's Been Completed

✅ Project structure defined
✅ Requirements.txt finalized (Python 3.12 compatible)
✅ Environment setup (.env template created)
✅ Complete architecture designed
✅ Data flows documented
✅ Agent roster defined (10 agents)
✅ Tool specifications outlined
✅ Implementation plan (8 phases)
✅ Main CLI runner created (`main.py`)
✅ Crew skeleton created (`crew/medical_rag_crew.py`)

### 2.2 What's NOT Done Yet

❌ No actual code implementation (only skeletons)
❌ Config/settings module not created
❌ Azure client factory not created
❌ No tools implemented
❌ No agents implemented
❌ No core modules (cache, memory, deduplication) implemented
❌ No tasks.py implementation
❌ No tests

### 2.3 What User Has

- Legacy working code in these files:
    - `retriever.py` (RetrieverAgent with FAISS)
    - `llm_generator.py` (LLMGeneratorAgent with medical prompts - PRESERVE EXACTLY)
    - `orchestrator.py` (RAGOrchestrator with classic pipeline)
    - `kb_manager.py` (KnowledgeBaseManager with DocIntel extraction)
    - `reranker.py` (3 reranking methods: simple, detailed, pairwise)
    - `query_enhancer.py` (Query enhancement with keywords/rephrase)
    - `retrieval_enhancer.py` (Distance filtering)
    - `conversation_memory.py` (Basic sliding window memory)
    - `config.py` (Old configuration)
    - `chat_bot.py` (Streamlit UI)
- These files have been moved to `legacy/` folder for reference

***

## 3. System Architecture

### 3.1 Agent Roster (10 Agents)

| \# | Agent | Role | Primary Responsibilities | Tools Used |
| :-- | :-- | :-- | :-- | :-- |
| 1 | **Cache Agent** | Semantic Cache Specialist | Check cache before operations, add validated answers | `semantic_search_cache`, `add_to_cache`, `get_cache_stats` |
| 2 | **Memory Agent** | Conversation Context Manager | Track entities, resolve coreferences, maintain multi-layer memory | `get_context_for_query`, `add_exchange`, `resolve_coreference`, `extract_entities` |
| 3 | **Text Extractor Agent** | Document Intelligence Specialist | Extract text from PDFs using Azure Document Intelligence | `extract_pdf`, `batch_extract_pdfs`, `calculate_content_signature` |
| 4 | **KB Agent** | Knowledge Base Engineer | Build/maintain index, deduplication, autosync with hash tracking | `check_pdf_status`, `build_index`, `append_to_index`, `deduplicate_chunks`, `autosync` |
| 5 | **Query Enhancement Agent** | Query Strategist | Enhance queries (decompose, HyDE, variations) - conditional based on query type | `decompose_complex_query`, `generate_hypothetical_answer`, `parallel_query_variations`, `add_medical_keywords` |
| 6 | **Retrieval Agent** | Contextual Retriever | FAISS search with distance filtering and multi-strategy fusion | `retrieve`, `multi_strategy_retrieve`, `filter_by_distance`, `adaptive_distance_filter` |
| 7 | **Reranking Agent** | Relevance Optimizer | Rerank chunks using cross-encoder or LLM | `rerank_with_crossencoder`, `rerank_with_llm`, `calibrate_scores` |
| 8 | **Answer Agent** | Medical Assistant | Generate answers using PRESERVED medical prompts | `generate_answer`, `format_context`, `estimate_tokens` |
| 9 | **Validation Agent** | Quality Assurance Specialist | Validate grounding, detect hallucinations, self-healing | `validate_citation_grounding`, `validate_citation_accuracy`, `detect_hallucination`, `self_heal_answer`, `validate_and_heal` |
| 10 | **Orchestrator/Supervisor** | Workflow Coordinator | Route queries through optimal agent pipeline | N/A (uses other agents) |

### 3.2 Multi-Layer Memory Design

**Layer 1: Short-term** (current session, sliding window, last 5 exchanges)
**Layer 2: Long-term** (summarized past conversations stored persistently)
**Layer 3: Entity memory** (tracks medications, diagnoses, dates, people, tests mentioned)
**Layer 4: Semantic memory** (FAISS index of past conversations, searchable by similarity)

### 3.3 Deduplication Strategy

**PDF-level deduplication:**

- Track both `file_hash` (MD5 of bytes) and `content_signature` (SHA256 of normalized text)
- If `content_signature` exists under different filename → mark as duplicate, skip indexing

**Chunk-level deduplication:**

- Calculate `chunk_signature` = SHA256(normalized_chunk_text)
- Maintain set of seen signatures during indexing
- Skip chunks with duplicate signatures

**Tracker structure:**

```json
{
  "filename.pdf": {
    "file_hash": "md5_hash",
    "content_signature": "sha256_hash",
    "last_indexed": "ISO_timestamp",
    "path": "full_path",
    "total_pages": 10,
    "total_chunks": 45,
    "is_canonical": true
  }
}
```


### 3.4 Semantic Cache Design

**Cache entry:**

```json
{
  "question": "What medication was prescribed?",
  "question_embedding": [...],
  "answer": "Metformin 500mg...",
  "confidence": 0.94,
  "confidence_band": "HIGH",
  "sources": [...],
  "timestamp": "ISO_timestamp",
  "kb_version": "hash_of_kb_state",
  "metadata": {
    "validation_passed": true,
    "self_healing_iterations": 0
  }
}
```

**Cache invalidation triggers:**

- TTL exceeded (default 24 hours)
- KB version changed (detected via KB version hash)
- Manual cache clear

**Cache search:**

- Similarity threshold: 0.95 (configurable)
- Returns cached answer if match found and valid
- Otherwise returns cache miss

***

## 4. Complete Data Flows

### 4.1 Flow A: KB Initialization \& Sync

```
┌──────────────────────────────────────┐
│ SYSTEM STARTUP / USER TRIGGERS SYNC │
└──────┬───────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ KB AGENT: Check KB Status              │
│  • FAISS index exists?                 │
│  • Get current KB version              │
└──────┬─────────────────────────────────┘
       │
       ├─[KB NOT EXISTS]──────┐
       │                       │
       └─[KB EXISTS]           │
       │                       │
       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ KB AGENT:       │    │ KB AGENT:       │
│ Scan input      │    │ Build from      │
│ Find new/mod    │    │ scratch         │
└──────┬──────────┘    └────────┬────────┘
       │                         │
       ├─[NO CHANGES]───┐        │
       │                 │        │
       └─[CHANGES FOUND] │        │
       │                 │        │
       ▼                 │        ▼
┌──────────────────────────────────────┐
│ For each PDF to process:             │
└──────┬───────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────┐
│ KB AGENT: check_pdf_status()          │
│  • Calculate file_hash (MD5)          │
│  • Check if in tracker                │
│  • Compare hash if exists             │
└──────┬────────────────────────────────┘
       │
       ├─[NEW]────────────────┐
       ├─[MODIFIED]───────────┤
       ├─[DUPLICATE]─────────┐│
       └─[UNCHANGED]────────┐││
                             │││
       ┌─────────────────────┘││
       │                      ││
       ▼                      ▼▼
┌────────────────┐    ┌──────────────┐
│ Process        │    │ SKIP         │
│ NEW/MODIFIED   │    │ (Duplicate)  │
└──────┬─────────┘    └──────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ TEXT EXTRACTOR AGENT: extract_pdf()     │
│  • Call Azure Document Intelligence     │
│  • Extract text per page                │
│  • Calculate content_signature          │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ KB AGENT: Check content_signature       │
└──────┬──────────────────────────────────┘
       │
       ├─[SIGNATURE EXISTS]───────┐
       │                           │
       └─[UNIQUE]                  │
       │                           ▼
       ▼                    ┌─────────────┐
┌──────────────────┐       │ DUPLICATE   │
│ KB AGENT:        │       │ Mark alias  │
│ chunk_text()     │       │ SKIP index  │
│ Track metadata   │       └─────────────┘
│ Calculate        │
│ chunk_signature  │
└──────┬───────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ KB AGENT: deduplicate_chunks()         │
│  • Check chunk_signature vs existing   │
│  • Keep only unique                    │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ RETRIEVAL AGENT: get_embeddings_batch()│
│  • Azure OpenAI Embedding API          │
│  • Batch size: 100                     │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ KB AGENT: Update FAISS Index           │
│  • Create or append to index           │
│  • Save: index, chunks, metadata       │
│  • Update tracker                      │
│  • Calculate KB version hash           │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ RETURN: Sync Report                    │
│  {pdfs_processed, new, modified,       │
│   duplicate, chunks_added,             │
│   chunks_deduplicated, kb_version}     │
└────────────────────────────────────────┘
       │
       ▼
   [KB READY FOR QUERIES]
```


### 4.2 Flow B: Query Answering

```
┌──────────────┐
│  USER QUERY  │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────┐
│ ORCHESTRATOR: Check KB loaded          │
│  • If not → trigger load               │
└──────┬─────────────────────────────────┘
       │
       ├─[KB NOT READY]──────┐
       │                      │
       └─[KB READY]           ▼
       │              ┌──────────────┐
       │              │ ERROR: Load  │
       │              │ KB first     │
       │              └──────────────┘
       ▼
┌────────────────────────────────────────┐
│ 1. MEMORY AGENT: get_context_for_query │
│  • Get recent exchanges                │
│  • Extract entities                    │
│  • Resolve coreferences                │
│  • Search semantic memory              │
└──────┬─────────────────────────────────┘
       │ [resolved_query + context]
       ▼
┌────────────────────────────────────────┐
│ 2. CACHE AGENT: semantic_search_cache()│
│  • Embed resolved_query                │
│  • Search cache index (>0.95)          │
│  • Check KB version + TTL              │
└──────┬─────────────────────────────────┘
       │
       ├─[CACHE HIT]────────────────┐
       │                             │
       └─[CACHE MISS]                │
       │                             │
       │                             ▼
       │                      ┌─────────────┐
       │                      │ Return      │
       │                      │ cached      │
       │                      │ answer      │
       │                      │ [SKIP 3-11] │
       │                      └──────┬──────┘
       ▼                             │
┌─────────────────────────────────┐ │
│ 3. QUERY CLASSIFICATION (opt)   │ │
│  • Classify query type          │ │
│  • Determine complexity         │ │
└──────┬──────────────────────────┘ │
       │                             │
       ▼                             │
┌─────────────────────────────────────┐│
│ 4. QUERY ENHANCEMENT AGENT          ││
│  • If complex: decompose            ││
│  • If high-level: HyDE (optional)   ││
│  • Generate variations              ││
│  • Add medical keywords             ││
└──────┬──────────────────────────────┘│
       │ [enhanced_queries]            │
       ▼                               │
┌─────────────────────────────────────┐│
│ 5. RETRIEVAL AGENT                  ││
│  • Multi-strategy retrieval         ││
│  • FAISS search (top_k + buffer)    ││
│  • Distance filtering               ││
│  • Reciprocal rank fusion           ││
└──────┬──────────────────────────────┘│
       │ [chunks + distances]          │
       ▼                               │
┌─────────────────────────────────────┐│
│ 6. RERANKING AGENT                  ││
│  • Cross-encoder OR LLM reranking   ││
│  • Score relevance                  ││
│  • Calibrate to confidence          ││
└──────┬──────────────────────────────┘│
       │ [reranked top_k chunks]       │
       ▼                               │
┌─────────────────────────────────────┐│
│ 7. ANSWER AGENT                     ││
│  • Format context (PRESERVED)       ││
│  • Build prompts (PRESERVED)        ││
│  • Call Azure OpenAI                ││
│  • Parse response                   ││
└──────┬──────────────────────────────┘│
       │ [raw_answer + sources]        │
       ▼                               │
┌─────────────────────────────────────┐│
│ 8. VALIDATION AGENT                 ││
│  • Check citation grounding         ││
│  • Verify citation accuracy         ││
│  • Detect hallucinations            ││
│  • Calculate confidence             ││
└──────┬──────────────────────────────┘│
       │                               │
       ├─[PASS]───────────────┐        │
       │                       │        │
       └─[FAIL]                │        │
       │                       │        │
       ▼                       │        │
┌──────────────────┐          │        │
│ 9. SELF-HEALING  │          │        │
│  • Identify      │          │        │
│    issues        │          │        │
│  • Regenerate    │          │        │
│  • Re-validate   │          │        │
│  • Max 2 iter    │          │        │
└──────┬───────────┘          │        │
       │ [corrected_answer]   │        │
       └──────────────────────┘        │
       │                               │
       ▼                               │
┌─────────────────────────────────────┐│
│ 10. MEMORY AGENT: add_exchange()    ││
│  • Store Q&A                        ││
│  • Extract entities                 ││
│  • Add to semantic memory           ││
│  • Summarize if buffer full         ││
└──────┬──────────────────────────────┘│
       │                               │
       ▼                               │
┌─────────────────────────────────────┐│
│ 11. CACHE AGENT: add_to_cache()     ││
│  • Embed query                      ││
│  • Store with KB version            ││
└──────┬──────────────────────────────┘│
       │                               │
       └───────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ RETURN TO USER                         │
│  {answer, confidence, confidence_band, │
│   sources, validation_passed,          │
│   self_healing_iterations,             │
│   resolved_query, entities_used,       │
│   cache_hit, metadata}                 │
└────────────────────────────────────────┘
```


### 4.3 Decision Logic for Query Enhancement

**Query Classification determines which enhancements to use:**


| Query Type | Example | Use HyDE? | Use Variations? | Use Keywords? |
| :-- | :-- | :-- | :-- | :-- |
| Field-level lookup | "What is patient's name?" | ❌ No | ❌ No | ✅ Light |
| Simple factual | "What was the diagnosis?" | ❌ No | ❌ No | ✅ Yes |
| High-level summary | "How was diabetes managed over time?" | ✅ Yes | ✅ Yes | ✅ Yes |
| Multi-step reasoning | "Compare current vs previous treatment" | ✅ Yes | ✅ Yes | ✅ Yes |

**Why HyDE is conditional:**

- Internal medical PDFs have structured fields
- Simple queries ("patient name", "policy number") already align well with document text
- HyDE generates hypothetical answers that can mislead for identifier queries
- Only use HyDE for semantic, high-level, or fuzzy queries

***

## 5. Agent Definitions

### 5.1 Cache Agent

```python
from crewai import Agent
from tools.cache_tools import CacheTools

cache_agent = Agent(
    role="Semantic Cache Specialist",
    goal="Identify and return previously answered similar questions to save time and cost",
    backstory="Expert at semantic similarity and maintaining a high-performance Q&A cache",
    tools=[
        CacheTools.semantic_search,
        CacheTools.add_to_cache,
        CacheTools.get_stats
    ],
    verbose=True
)
```


### 5.2 Memory Agent

```python
from crewai import Agent
from tools.memory_tools import MemoryTools

memory_agent = Agent(
    role="Conversation Memory Specialist",
    goal="Maintain conversation context, resolve references, and provide relevant history",
    backstory="Expert at tracking conversation flow, entity mentions, and contextual understanding",
    tools=[
        MemoryTools.get_context_for_query,
        MemoryTools.add_exchange,
        MemoryTools.resolve_coreference,
        MemoryTools.extract_entities,
        MemoryTools.search_past_conversations,
        MemoryTools.clear_memory
    ],
    verbose=True
)
```


### 5.3 Text Extractor Agent

```python
from crewai import Agent
from tools.extraction_tools import ExtractionTools

extractor_agent = Agent(
    role="Medical Document Intelligence Specialist",
    goal="Extract text from complex medical PDFs including handwritten content using Azure Document Intelligence",
    backstory="Specialist in OCR and document layout analysis for medical documents",
    tools=[
        ExtractionTools.extract_pdf,
        ExtractionTools.batch_extract,
        ExtractionTools.calculate_signature
    ],
    verbose=True
)
```


### 5.4 KB Agent

```python
from crewai import Agent
from tools.kb_tools import KBTools

kb_agent = Agent(
    role="Knowledge Base Engineer",
    goal="Build and maintain a deduplicated, hash-tracked knowledge base",
    backstory="Expert at document indexing, deduplication, and incremental updates",
    tools=[
        KBTools.check_pdf_status,
        KBTools.build_index,
        KBTools.append_to_index,
        KBTools.deduplicate_chunks,
        KBTools.autosync,
        KBTools.get_stats
    ],
    verbose=True,
    allow_delegation=True  # Can delegate extraction to Text Extractor
)
```


### 5.5 Query Enhancement Agent

```python
from crewai import Agent
from tools.query_tools import QueryTools

query_agent = Agent(
    role="Query Enhancement Specialist",
    goal="Optimize queries for better retrieval through decomposition, HyDE, and domain expansion",
    backstory="Expert in query understanding, semantic expansion, and medical terminology",
    tools=[
        QueryTools.classify_query,
        QueryTools.decompose_complex_query,
        QueryTools.generate_hypothetical_answer,
        QueryTools.parallel_query_variations,
        QueryTools.add_medical_keywords
    ],
    verbose=True
)
```


### 5.6 Retrieval Agent

```python
from crewai import Agent
from tools.retrieval_tools import RetrievalTools

retrieval_agent = Agent(
    role="Contextual Retriever",
    goal="Find the most relevant medical document chunks with advanced retrieval techniques",
    backstory="Expert in vector search, query enhancement, and semantic reranking",
    tools=[
        RetrievalTools.retrieve,
        RetrievalTools.multi_strategy_retrieve,
        RetrievalTools.filter_by_distance,
        RetrievalTools.adaptive_distance_filter,
        RetrievalTools.search_by_source
    ],
    verbose=True
)
```


### 5.7 Reranking Agent

```python
from crewai import Agent
from tools.reranking_tools import RerankingTools

reranking_agent = Agent(
    role="Relevance Optimizer",
    goal="Rerank retrieved chunks to prioritize most relevant content",
    backstory="Expert at relevance scoring and semantic similarity",
    tools=[
        RerankingTools.rerank_with_crossencoder,
        RerankingTools.rerank_with_llm,
        RerankingTools.calibrate_scores
    ],
    verbose=True
)
```


### 5.8 Answer Agent

```python
from crewai import Agent
from tools.answer_tools import AnswerTools

answer_agent = Agent(
    role="Medical Assistant",
    goal="Generate concise, accurate, well-cited medical answers",
    backstory="Medical documentation expert trained to provide precise answers with proper citations",
    tools=[
        AnswerTools.generate_answer,
        AnswerTools.format_context,
        AnswerTools.estimate_tokens
    ],
    verbose=True
)
```


### 5.9 Validation Agent

```python
from crewai import Agent
from tools.validation_tools import ValidationTools

validation_agent = Agent(
    role="Quality Assurance Specialist",
    goal="Validate answer quality, detect hallucinations, ensure citation accuracy, and self-correct when needed",
    backstory="Expert at fact-checking, citation validation, and self-healing workflows",
    tools=[
        ValidationTools.validate_citation_grounding,
        ValidationTools.validate_citation_accuracy,
        ValidationTools.detect_hallucination,
        ValidationTools.self_heal_answer,
        ValidationTools.calculate_confidence,
        ValidationTools.validate_and_heal
    ],
    verbose=True,
    allow_delegation=True  # Can ask Answer Agent to regenerate
)
```


***

## 6. Tool Specifications

### 6.1 Cache Tools

```python
from crewai_tools import tool

@tool("Semantic Search Cache")
def semantic_search_cache(query: str, kb_version: str) -> dict:
    """
    Search semantic Q&A cache for similar previously answered questions.
    
    Args:
        query: User's question
        kb_version: Current KB version hash (for invalidation)
    
    Returns:
        Cached answer dict if similarity > 0.95 and valid, else None
    """
    # Implementation to be done

@tool("Add to Cache")
def add_to_cache(query: str, answer: str, metadata: dict, kb_version: str) -> bool:
    """Add validated answer to semantic cache."""
    # Implementation to be done

@tool("Get Cache Stats")
def get_cache_stats() -> dict:
    """Return cache statistics."""
    # Implementation to be done
```


### 6.2 Memory Tools

```python
@tool("Get Context for Query")
def get_context_for_query(query: str, max_history: int = 3) -> dict:
    """
    Get conversation context and resolve references.
    
    Returns:
        {
            'recent_exchanges': [...],
            'resolved_query': str,
            'relevant_entities': [...],
            'relevant_past_context': str,
            'original_query': str
        }
    """
    # Implementation to be done

@tool("Add Exchange")
def add_exchange(query: str, answer: str, metadata: dict) -> None:
    """Store Q&A exchange in multi-layer memory."""
    # Implementation to be done

@tool("Resolve Coreference")
def resolve_coreference(query: str, recent_exchanges: list) -> str:
    """Resolve pronouns/references using entity memory."""
    # Implementation to be done

@tool("Extract Entities")
def extract_entities(text: str) -> dict:
    """Extract medical entities using LLM."""
    # Implementation to be done
```


### 6.3 Extraction Tools

```python
@tool("Extract PDF")
def extract_pdf(pdf_path: str) -> dict:
    """
    Extract text from PDF using Azure Document Intelligence.
    
    Returns:
        {
            'full_text': str,
            'page_texts': dict,
            'file_hash': str,
            'content_signature': str,
            'metadata': {...}
        }
    """
    # Implementation to be done

@tool("Calculate Content Signature")
def calculate_content_signature(text: str) -> str:
    """Calculate SHA256 hash of normalized text for deduplication."""
    # Implementation to be done
```


### 6.4 KB Tools

```python
@tool("Check PDF Status")
def check_pdf_status(pdf_path: str) -> dict:
    """
    Check if PDF is new, modified, or duplicate.
    
    Returns:
        {
            'status': 'new|modified|duplicate_content|unchanged',
            'file_hash': str,
            'content_signature': str,
            'recommendation': str
        }
    """
    # Implementation to be done

@tool("Build Index")
def build_index_from_folder(folder_path: str) -> dict:
    """Build FAISS index from scratch with deduplication."""
    # Implementation to be done

@tool("Deduplicate Chunks")
def deduplicate_chunks(chunks: list, existing_signatures: set) -> list:
    """Remove duplicate chunks using content signatures."""
    # Implementation to be done
```


### 6.5 Query Tools

```python
@tool("Classify Query")
def classify_query(query: str) -> dict:
    """
    Classify query type to route to best strategy.
    
    Returns:
        {
            'type': 'factual|yes_no|list|comparison|definition',
            'complexity': 'simple|complex',
            'requires_hyde': bool,
            'requires_variations': bool
        }
    """
    # Implementation to be done

@tool("Decompose Complex Query")
def decompose_complex_query(query: str) -> list:
    """Break complex multi-part questions into atomic sub-queries."""
    # Implementation to be done

@tool("Generate Hypothetical Answer")
def generate_hypothetical_answer(query: str) -> str:
    """HyDE: Generate hypothetical answer for better retrieval."""
    # Implementation to be done
```


### 6.6 Retrieval Tools

```python
@tool("Retrieve")
def retrieve(query: str, top_k: int = 5, return_distances: bool = True) -> list:
    """Basic FAISS vector search."""
    # Implementation to be done

@tool("Multi-Strategy Retrieve")
def multi_strategy_retrieve(queries: list, top_k_each: int = 5) -> list:
    """Retrieve with multiple queries, then fuse using reciprocal rank."""
    # Implementation to be done

@tool("Filter by Distance")
def filter_by_distance(chunks: list, threshold: float = 1.5, min_chunks: int = 3) -> list:
    """Filter chunks by distance threshold."""
    # Implementation to be done
```


### 6.7 Reranking Tools

```python
@tool("Rerank with Cross-Encoder")
def rerank_with_crossencoder(query: str, chunks: list, top_k: int = 5) -> list:
    """Fast reranking using Hugging Face cross-encoder model."""
    # Implementation to be done

@tool("Rerank with LLM")
def rerank_with_llm(query: str, chunks: list, method: str = 'simple', top_k: int = 5) -> list:
    """LLM-based reranking (simple/detailed/pairwise from legacy reranker.py)."""
    # Implementation to be done
```


### 6.8 Answer Tools

```python
@tool("Generate Answer")
def generate_answer(query: str, chunks: list, conversation_history: str = None) -> dict:
    """
    Generate answer using PRESERVED medical prompts.
    
    CRITICAL: Uses exact system + user prompts from legacy llm_generator.py
    
    Returns:
        {
            'answer': str,
            'sources': list,
            'context': str
        }
    """
    # Implementation to be done

@tool("Format Context")
def format_context(chunks: list) -> str:
    """Format chunks into context string (PRESERVE existing format from llm_generator.py)."""
    # Implementation to be done
```


### 6.9 Validation Tools

```python
@tool("Validate Citation Grounding")
def validate_citation_grounding(answer: str, source_chunks: list) -> dict:
    """
    Check if all claims in answer are grounded in source chunks.
    
    Returns:
        {
            'grounded': bool,
            'grounding_score': float,
            'ungrounded_claims': list,
            'issues': list
        }
    """
    # Implementation to be done

@tool("Detect Hallucination")
def detect_hallucination(answer: str, source_chunks: list, query: str) -> dict:
    """Use LLM to detect information in answer NOT supported by sources."""
    # Implementation to be done

@tool("Self-Heal Answer")
def self_heal_answer(query: str, answer: str, chunks: list, issues: list) -> str:
    """Regenerate answer with explicit instruction to fix validation issues."""
    # Implementation to be done

@tool("Validate and Heal")
def validate_and_heal(query: str, answer: str, chunks: list, max_iterations: int = 2) -> dict:
    """
    Main validation workflow with self-healing loop.
    
    Returns:
        {
            'answer': str,
            'validated': bool,
            'confidence': float,
            'confidence_band': str,
            'iterations': int,
            'issues_found': list,
            'warning': str|None
        }
    """
    # Implementation to be done
```


***

## 7. Critical Prompts (MUST PRESERVE)

### 7.1 System Prompt (from llm_generator.py)

**DO NOT MODIFY - THIS CONTROLS ANSWER QUALITY**

```python
SYSTEM_PROMPT = """You are an expert medical document assistant that provides CONCISE, ACCURATE answers.

Answer Style Rules:
1. MATCH answer format to question type:
   - Yes/No questions → Yes/No + brief reason (1 sentence)
   - "What is X?" → Direct answer first, then context if needed
   - "How many?" → Number first, then brief list
   - Lists → Bullet points only

2. BE CONCISE - no unnecessary elaboration

3. Start with the direct answer IMMEDIATELY

4. Add supporting details ONLY if directly relevant

5. Cite sources using the file name from context, not "Document X"

6. Use conversation history to understand context and resolve references like "it", "that medication", "the patient"

7. Only say "I don't have enough information" if context contains NO relevant information

Answer Examples:
Q: Was medication prescribed?
A: Yes. Metformin 500mg twice daily. [Source: Document 1]

Q: What medication?
A: Metformin 500mg twice daily for diabetes management. [Source: Document 1]

Q: Glucose level?
A: 150 mg/dL (elevated, normal range 70-100 mg/dL). [Source: Document 1]
"""
```


### 7.2 User Prompt Template (from llm_generator.py)

**DO NOT MODIFY**

```python
USER_PROMPT_TEMPLATE = """Based on the following documents, answer the question CONCISELY.

CONTEXT DOCUMENTS:
{formatted_context}

QUESTION:
{query}

INSTRUCTIONS:
- Match answer length to question complexity
- Start with direct answer, add details only if needed
- Use information from ALL provided documents
- If this is a follow-up question, use the previous conversation context
- Cite which documents you used
- Prioritize documents with higher relevance scores

ANSWER:
"""

# With conversation history:
USER_PROMPT_WITH_HISTORY_TEMPLATE = """Based on the following documents, answer the question CONCISELY.

CONTEXT DOCUMENTS:
{formatted_context}

PREVIOUS CONVERSATION:
{conversation_history}

QUESTION:
{query}

INSTRUCTIONS:
- Match answer length to question complexity
- Start with direct answer, add details only if needed
- Use information from ALL provided documents
- If this is a follow-up question, use the previous conversation context
- Cite which documents you used
- Prioritize documents with higher relevance scores

ANSWER:
"""
```


### 7.3 Context Formatting (from llm_generator.py)

**PRESERVE THIS EXACT FORMAT**

```python
def format_context(retrieved_chunks):
    """
    Format retrieved chunks into context string.
    
    MUST PRESERVE THIS EXACT FORMAT - IT'S WHAT THE PROMPTS EXPECT
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        text = chunk.get('text', chunk.get('chunk', ''))
        metadata = chunk.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        page_numbers = metadata.get('page_numbers', metadata.get('page', 'N/A'))
        
        # Handle page display
        if isinstance(page_numbers, list):
            if len(page_numbers) > 1:
                page_display = f"Pages {page_numbers[0]}-{page_numbers[-1]}"
            elif len(page_numbers) == 1:
                page_display = f"Page {page_numbers[0]}"
            else:
                page_display = "Page N/A"
        else:
            page_display = f"Page {page_numbers}"
        
        context_parts.append(f"Document {i} ({source}, {page_display}):\n{text}")
    
    return "\n\n".join(context_parts)
```


### 7.4 LLM Configuration (from llm_generator.py)

**PRESERVE THESE VALUES**

```python
LLM_CONFIG = {
    'model': os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-4'),
    'temperature': 0.3,  # From original
    'max_tokens': 400    # From original
}
```


***

## 8. Implementation Phases

### Phase 1: Foundation (Week 1-2) - START HERE

**Files to create:**

1. `config/settings.py` - Pydantic settings replacing old config.py
2. `utils/azure_clients.py` - Azure OpenAI \& DocIntel client factory
3. `utils/prompt_templates.py` - Store PRESERVED prompts
4. `utils/file_utils.py` - File hashing, I/O utilities
5. `core/deduplication.py` - Content signature calculation

**Deliverables:**

- Settings system with environment variable loading
- Azure clients working and tested
- Prompts accessible as constants
- File utilities tested
- Backward compatibility maintained


### Phase 2: Core Services (Week 3)

**Files to create:**
6. `core/pdf_processor.py` - Wrap Azure Document Intelligence
7. `core/embeddings.py` - Embedding generation utilities
8. `core/entity_memory.py` - Entity tracking
9. `core/enhanced_memory.py` - Multi-layer conversation memory
10. `core/semantic_cache.py` - Semantic cache with FAISS

**Deliverables:**

- PDF extraction working
- Embeddings generated in batches
- Entity extraction and coreference working
- Memory tracking conversations
- Cache saving/loading/searching


### Phase 3: Tools Layer (Week 4-5)

**Files to create:**
11-19. All tool files in `tools/` directory

**Deliverables:**

- All tools implemented and tested individually
- Tools wrapped with `@tool` decorator for CrewAI
- Backward compatibility preserved


### Phase 4: Agents Layer (Week 6)

**Files to create:**
20-28. All agent files in `agents/` directory

**Deliverables:**

- All 9 agents defined
- Agents linked to appropriate tools
- Agent roles and backstories finalized


### Phase 5: Crew Orchestration (Week 7)

**Files to create:**
29. `crew/tasks.py` - Task definitions
30. `crew/workflows.py` - Workflow logic
31. `crew/medical_rag_crew.py` - Main crew (already has skeleton)

**Deliverables:**

- All tasks defined
- Workflows working
- Full crew executing end-to-end


### Phase 6: UI (Week 8) - DEFERRED

**Files to create:**

- Streamlit components

**Status:** Deferred - focus on CLI first

### Phase 7: Testing \& Optimization (Week 9)

**Deliverables:**

- Test suite
- Benchmarks
- Performance tuning

***

## 9. Project Structure

```
medical-agentic-rag/
│
├── .env                          # Environment variables
├── .gitignore
├── requirements.txt              # Python 3.12 compatible
├── README.md
├── AGENTIC_IMPLEMENTATION_PLAN.md
│
├── main.py                       # CLI runner (CREATED - skeleton exists)
│
├── config/
│   ├── __init__.py
│   └── settings.py               # TO CREATE - Phase 1
│
├── agents/                       # TO CREATE - Phase 4
│   ├── __init__.py
│   ├── cache_agent.py
│   ├── memory_agent.py
│   ├── extractor_agent.py
│   ├── kb_agent.py
│   ├── query_agent.py
│   ├── retrieval_agent.py
│   ├── reranking_agent.py
│   ├── answer_agent.py
│   └── validation_agent.py
│
├── tools/                        # TO CREATE - Phase 3
│   ├── __init__.py
│   ├── cache_tools.py
│   ├── memory_tools.py
│   ├── extraction_tools.py
│   ├── kb_tools.py
│   ├── query_tools.py
│   ├── retrieval_tools.py
│   ├── reranking_tools.py
│   ├── answer_tools.py
│   └── validation_tools.py
│
├── core/                         # TO CREATE - Phase 2
│   ├── __init__.py
│   ├── enhanced_memory.py
│   ├── entity_memory.py
│   ├── semantic_cache.py
│   ├── pdf_processor.py
│   ├── embeddings.py
│   └── deduplication.py
│
├── crew/                         # TO CREATE - Phase 5
│   ├── __init__.py
│   ├── medical_rag_crew.py      # Skeleton exists
│   ├── tasks.py
│   └── workflows.py
│
├── utils/                        # TO CREATE - Phase 1
│   ├── __init__.py
│   ├── azure_clients.py
│   ├── file_utils.py
│   ├── prompt_templates.py
│   └── logging_config.py
│
├── legacy/                       # Reference only (DO NOT MODIFY)
│   ├── __init__.py
│   ├── retriever.py
│   ├── llm_generator.py
│   ├── orchestrator.py
│   ├── kb_manager.py
│   ├── reranker.py
│   ├── query_enhancer.py
│   ├── retrieval_enhancer.py
│   └── conversation_memory.py
│
├── data/                         # Created by system
│   ├── input/                   # User puts PDFs here
│   ├── knowledge_base/          # FAISS index, chunks, metadata
│   ├── cache/                   # Semantic cache storage
│   └── memory/                  # Conversation memory
│
├── tests/                        # TO CREATE - Phase 7
│   ├── __init__.py
│   ├── test_tools/
│   ├── test_agents/
│   └── test_integration/
│
└── scripts/
    ├── setup_kb.py
    └── clear_cache.py
```


***

## 10. Dependencies \& Environment

### 10.1 requirements.txt (Final - Python 3.12 Compatible)

```txt
# Core
python-dotenv

# Azure
azure-ai-formrecognizer
azure-identity

# Vector & ML
faiss-cpu
numpy>=1.26.0
sentence-transformers

# CrewAI (will install compatible openai, pydantic, langchain)
crewai
crewai-tools

# UI & Utils
pandas
streamlit
tqdm
loguru
python-dateutil

# Testing
pytest
pytest-asyncio
pytest-mock
```


### 10.2 .env Template

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Paths
INDEX_PATH=./data/knowledge_base
INPUT_FOLDER=./data/input
CACHE_PATH=./data/cache
MEMORY_PATH=./data/memory

# KB Settings
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100

# Retrieval
DEFAULT_TOP_K=5
DISTANCE_THRESHOLD=1.5

# LLM
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=400

# Cache
CACHE_SIMILARITY_THRESHOLD=0.95
CACHE_TTL_HOURS=24

# Memory
MAX_MEMORY_EXCHANGES=5

# Validation
VALIDATION_MAX_ITERATIONS=2
VALIDATION_THRESHOLD=0.85

# Feature Flags
ENABLE_AGENTIC_MODE=true
ENABLE_SEMANTIC_CACHE=true
ENABLE_QUERY_ENHANCEMENT=true
ENABLE_RERANKING=true
ENABLE_VALIDATION=true
ENABLE_AUTO_SYNC=true

# Logging
LOG_LEVEL=INFO
```


### 10.3 Python Version

- **Python 3.12** (user confirmed)
- All dependencies verified compatible with 3.12

***

## 11. Key Decisions \& Constraints

### 11.1 Must Preserve

✅ Exact system prompt from `llm_generator.py`
✅ Exact user prompt template from `llm_generator.py`
✅ Context formatting function from `llm_generator.py`
✅ LLM temperature (0.3) and max_tokens (400)
✅ Azure Document Intelligence for PDF extraction
✅ FAISS as vector store

### 11.2 Must Implement

✅ Content signature (SHA256) for PDF deduplication
✅ Chunk signature (SHA256) for chunk deduplication
✅ Semantic cache with FAISS
✅ Multi-layer conversation memory
✅ Entity tracking and coreference resolution
✅ Self-healing validation with max 2 iterations
✅ Confidence scoring (multi-factor)

### 11.3 Conditional Features

✅ HyDE - only for high-level/complex queries
✅ Query variations - only for complex queries
✅ Cross-encoder reranking - optional (faster than LLM)
✅ LLM reranking - optional (more accurate but slower)

### 11.4 Performance Targets

- Cache hit rate: > 40% after 1 week of usage
- Validation pass rate: > 95%
- Latency overhead: < 1 second vs classic mode
- Answer quality: ≥ baseline (no regression)


### 11.5 Design Patterns

- **Agent delegation**: KB Agent can delegate to Text Extractor Agent
- **Sequential process**: Use `Process.sequential` for predictable flow
- **Tool-based**: All functionality via tools (testable, modular)
- **Fail-safe**: Always have fallback to classic mode if agentic fails

***

## 12. Next Immediate Steps

### Step 1: Create Foundation Files (Phase 1)

**File 1: `config/settings.py`**

- Create Pydantic Settings class
- Load from .env file
- Replace legacy `config.py`
- Expose all configuration as properties
- Add validation

**File 2: `utils/azure_clients.py`**

- Create singleton Azure OpenAI client
- Create singleton Azure Document Intelligence client
- Handle authentication
- Error handling and retries

**File 3: `utils/prompt_templates.py`**

- Store SYSTEM_PROMPT as constant
- Store USER_PROMPT_TEMPLATE as constant
- Store USER_PROMPT_WITH_HISTORY_TEMPLATE as constant
- Store format_context function
- Store LLM_CONFIG dict

**File 4: `utils/file_utils.py`**

- Implement MD5 file hashing
- Implement SHA256 text hashing
- Implement text normalization for signatures
- File I/O helpers

**File 5: `core/deduplication.py`**

- Implement content signature calculation
- Implement chunk signature calculation
- Implement signature comparison logic


### Step 2: Test Foundation

- Write unit tests for each module
- Verify Azure clients connect
- Verify prompts load correctly
- Verify hashing works


### Step 3: Move to Phase 2 (Core Services)

- Only after Phase 1 is complete and tested

***

## How to Resume in New Chat

**Feed this entire document to the AI and say:**

> "I'm continuing the Agentic Medical RAG project. I've completed the setup phase and need to implement Phase 1 starting with `config/settings.py`. Please review the context document and let me know you understand the project, then we'll start implementing Phase 1, File 1: `config/settings.py`."

**The AI should:**

1. Acknowledge understanding of the project
2. Confirm it has read all critical prompts
3. Understand the architecture
4. Know what NOT to change (preserved prompts)
5. Be ready to start coding Phase 1

***

## Important Reminders

1. **NEVER modify the medical assistant prompts** - they control answer quality
2. **Always use Azure Document Intelligence** - it handles handwritten text
3. **Deduplication at two levels** - PDF and chunk
4. **HyDE is conditional** - not for simple/identifier queries
5. **Semantic cache checks KB version** - invalidate on KB changes
6. **Validation has max 2 iterations** - prevent infinite loops
7. **Test incrementally** - each phase before moving to next
8. **Keep backward compatibility** - classic mode as fallback

***

**END OF CONTEXT DOCUMENT**

This document contains everything needed to continue implementation. All critical information, decisions, architecture, and next steps are documented.

