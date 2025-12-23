# test_phase2_performance.py
import time
from orchestration.main_runner import create_rag_session

queries = [
    "What is the case number?",  # Simple
    "Who is the plaintiff?",      # Simple
    "What medications were prescribed and when?",  # Medium
]

pipeline = create_rag_session()

for query in queries:
    start = time.time()
    result = pipeline.run_query(query)
    elapsed = time.time() - start
    
    print(f"\nQuery: {query}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Healing: {result['metadata'].get('healing_applied', False)}")
    print(f"Retries: {result['metadata'].get('retry_count', 0)}")
    
    # Check performance goals
    if elapsed < 5:
        print("✅ EXCELLENT performance")
    elif elapsed < 8:
        print("✅ GOOD performance")
    else:
        print("⚠️ SLOW - needs optimization")
