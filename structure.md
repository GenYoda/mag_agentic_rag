medical-agentic-rag/
│
├── .env                          # Environment variables (Azure keys, configs)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── AGENTIC_IMPLEMENTATION_PLAN.md  # The plan document we created
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Central configuration (replaces old config.py)
│
├── agents/                       # CrewAI agent definitions
│   ├── __init__.py
│   ├── cache_agent.py           # Semantic cache specialist
│   ├── memory_agent.py          # Conversation context manager
│   ├── extractor_agent.py       # Document intelligence specialist
│   ├── kb_agent.py              # Knowledge base engineer
│   ├── query_agent.py           # Query enhancement specialist
│   ├── retrieval_agent.py       # Contextual retriever
│   ├── reranking_agent.py       # Relevance optimizer
│   ├── answer_agent.py          # Medical assistant (preserves prompts)
│   └── validation_agent.py      # Quality assurance specialist
│
├── tools/                        # Agent tools (functions with @tool decorator)
│   ├── __init__.py
│   ├── cache_tools.py           # Semantic cache operations
│   ├── memory_tools.py          # Conversation memory operations
│   ├── extraction_tools.py      # PDF extraction via DocIntel
│   ├── kb_tools.py              # KB building, dedup, autosync
│   ├── query_tools.py           # Query enhancement, decomposition, HyDE
│   ├── retrieval_tools.py       # FAISS search, fusion, filtering
│   ├── reranking_tools.py       # Cross-encoder, LLM reranking
│   ├── answer_tools.py          # Answer generation (preserved prompts)
│   └── validation_tools.py      # Grounding, hallucination, self-healing
│
├── core/                         # Core business logic (non-agent modules)
│   ├── __init__.py
│   ├── enhanced_memory.py       # EnhancedConversationMemory class
│   ├── entity_memory.py         # Entity tracking and coreference
│   ├── semantic_cache.py        # SemanticCache class with FAISS
│   ├── pdf_processor.py         # DocIntel wrapper
│   ├── embeddings.py            # Embedding generation utilities
│   └── deduplication.py         # Content signature & chunk dedup logic
│
├── crew/                         # CrewAI orchestration
│   ├── __init__.py
│   ├── medical_rag_crew.py      # Main Crew definition
│   ├── tasks.py                 # Task definitions
│   └── workflows.py             # Workflow orchestrators (KB sync, query)
│
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── azure_clients.py         # Azure OpenAI & DocIntel client factory
│   ├── file_utils.py            # File I/O, hashing
│   ├── prompt_templates.py      # Preserved prompts from llm_generator.py
│   └── logging_config.py        # Logging setup
│
├── legacy/                       # Preserve old working code for reference
│   ├── __init__.py
│   ├── retriever.py             # Original (read-only)
│   ├── llm_generator.py         # Original (read-only)
│   ├── orchestrator.py          # Original (read-only)
│   ├── kb_manager.py            # Original (read-only)
│   ├── reranker.py              # Original (read-only)
│   ├── query_enhancer.py        # Original (read-only)
│   ├── retrieval_enhancer.py    # Original (read-only)
│   └── conversation_memory.py   # Original (read-only)
│
├── data/                         # Data directories (git-ignored)
│   ├── input/                   # PDF input folder
│   ├── knowledge_base/          # FAISS index, chunks, metadata
│   │   ├── faiss.index
│   │   ├── chunks.pkl
│   │   ├── metadata.json
│   │   ├── config.json
│   │   └── document_tracker.json
│   ├── cache/                   # Semantic cache storage
│   │   ├── cache_index.faiss
│   │   └── cache_data.json
│   └── memory/                  # Conversation memory persistence
│       └── conversation_history.json
│
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   ├── test_tools/
│   ├── test_agents/
│   ├── test_core/
│   └── test_integration/
│
├── ui/                           # Streamlit interface
│   ├── __init__.py
│   ├── app.py                   # Main Streamlit app
│   ├── components/              # UI components
│   │   ├── __init__.py
│   │   ├── sidebar.py
│   │   ├── chat_area.py
│   │   ├── stats_panel.py
│   │   └── debug_panel.py
│   └── styles/
│       └── custom.css
│
└── scripts/                      # Utility scripts
    ├── setup_kb.py              # Initial KB build script
    ├── clear_cache.py           # Cache management
    └── run_tests.py             # Test runner
