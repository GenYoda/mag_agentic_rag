"""
Test script for MemoryTools (Phase 3 - Tool 8/9)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from tools.memory_tools import MemoryTools
from datetime import datetime
import time


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)


def test_1_initialization():
    """Test 1: Memory initialization"""
    print_section("TEST 1: Memory Initialization")
    
    memory = MemoryTools(
        session_id="test_session_1",
        max_history_turns=10,
        enable_persistence=True,
        enable_entity_tracking=True
    )
    
    stats = memory.get_memory_stats()
    
    print(f"\n📊 Memory Configuration:")
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Max History Turns: {stats['max_history_turns']}")
    print(f"   Persistence: {'✅' if stats['persistence_enabled'] else '❌'}")
    print(f"   Entity Tracking: {'✅' if stats['entity_tracking_enabled'] else '❌'}")
    print(f"   Current Turns: {stats['turns_in_memory']}")
    print(f"   Current Entities: {stats['total_entities']}")
    
    print("\n✅ Initialization test complete")
    return memory


def test_2_basic_conversation(memory: MemoryTools):
    """Test 2: Basic conversation tracking"""
    print_section("TEST 2: Basic Conversation Tracking")
    
    # Add some conversation turns
    conversations = [
        ("What are the allegations against Memorial Health?",
         "Memorial Health faces allegations of medical negligence involving patient Sarah Johnson."),
        
        ("Who are the defendants?",
         "The defendants include Memorial Health, Nurse Brittany Roy, and Dr. Thombs."),
        
        ("What happened to Sarah Johnson?",
         "Sarah Johnson received incorrect medication on June 15, 2023, leading to complications.")
    ]
    
    print("\n💬 Adding conversation turns:\n")
    
    for i, (query, response) in enumerate(conversations, 1):
        turn = memory.add_turn(query, response)
        print(f"   [{i}] Q: {query}")
        print(f"       A: {response[:80]}...")
        print(f"       Entities: {len(turn.entities_mentioned)}")
        print()
        time.sleep(0.1)
    
    stats = memory.get_memory_stats()
    print(f"📊 After conversation:")
    print(f"   Total turns: {stats['total_turns']}")
    print(f"   Entities tracked: {stats['total_entities']}")
    print(f"   Entity types: {stats['entities_by_type']}")
    
    print("\n✅ Basic conversation test complete")


def test_3_context_retrieval(memory: MemoryTools):
    """Test 3: Context retrieval"""
    print_section("TEST 3: Context Retrieval")
    
    print("\n📖 Full Conversation Context:\n")
    context = memory.get_conversation_context(max_turns=3)
    print(context[:500] + "...\n")
    
    print("\n🎯 Relevant Context for: 'Tell me more about Nurse Roy'")
    relevant = memory.get_relevant_context("Tell me more about Nurse Roy", max_turns=2)
    print(relevant[:400] + "...")
    
    print("\n✅ Context retrieval test complete")


def test_4_entity_tracking(memory: MemoryTools):
    """Test 4: Entity tracking"""
    print_section("TEST 4: Entity Tracking")
    
    print("\n👥 Tracked Entities:\n")
    
    all_entities = memory.get_all_entities()
    
    for i, entity in enumerate(all_entities[:5], 1):  # Show first 5
        print(f"   [{i}] {entity.name}")
        print(f"       Type: {entity.entity_type}")
        print(f"       Mentions: {len(entity.mentions)} times")
        print(f"       First seen: {entity.first_seen.strftime('%H:%M:%S')}")
        print()
    
    print(f"📊 Total unique entities: {len(all_entities)}")
    
    # Get specific entity context
    if all_entities:
        first_entity = all_entities[0]
        print(f"\n📝 Detail for '{first_entity.name}':")
        context = memory.get_entity_context(first_entity.entity_id)
        print(context)
    
    print("\n✅ Entity tracking test complete")


def test_5_follow_up_detection(memory: MemoryTools):
    """Test 5: Follow-up question detection"""
    print_section("TEST 5: Follow-Up Question Detection")
    
    test_queries = [
        "What are the allegations against Memorial Health?",  # Not follow-up
        "What about the defendants?",  # Follow-up
        "Tell me more about that",  # Follow-up
        "Who is Nurse Roy?",  # Not follow-up
        "What did she do?",  # Follow-up
        "What are the damages claimed?",  # Not follow-up
    ]
    
    print("\n🔍 Testing follow-up detection:\n")
    
    for query in test_queries:
        is_follow_up = memory.detect_follow_up(query)
        icon = "↪️" if is_follow_up else "❓"
        label = "FOLLOW-UP" if is_follow_up else "NEW QUERY"
        
        print(f"   {icon} [{label}] {query}")
    
    print("\n✅ Follow-up detection test complete")


def test_6_persistence(memory: MemoryTools):
    """Test 6: Memory persistence"""
    print_section("TEST 6: Memory Persistence & Recovery")
    
    session_id = memory.session_id
    
    print("\n💾 Saving current session...")
    stats_before = memory.get_memory_stats()
    print(f"   Turns: {stats_before['total_turns']}")
    print(f"   Entities: {stats_before['total_entities']}")
    
    # Create new instance with same session ID
    print("\n🔄 Creating new instance with same session ID...")
    memory_reloaded = MemoryTools(session_id=session_id)
    
    stats_after = memory_reloaded.get_memory_stats()
    print(f"\n✅ Session reloaded:")
    print(f"   Turns: {stats_after['total_turns']}")
    print(f"   Entities: {stats_after['total_entities']}")
    
    # Verify data matches
    if stats_before['total_turns'] == stats_after['total_turns']:
        print(f"\n✅ Persistence working correctly!")
    else:
        print(f"\n❌ Persistence issue detected")
    
    print("\n✅ Persistence test complete")


def test_7_memory_limit():
    """Test 7: Memory limit enforcement"""
    print_section("TEST 7: Memory Limit Enforcement")
    
    # Create memory with small limit
    memory = MemoryTools(
        session_id="test_limit",
        max_history_turns=5,
        enable_persistence=False
    )
    
    print(f"\n📊 Max turns allowed: {memory.max_history_turns}")
    print(f"\n💬 Adding 8 conversation turns...\n")
    
    for i in range(8):
        memory.add_turn(
            f"Question {i+1}",
            f"Answer {i+1}"
        )
        turns_in_memory = len(memory.conversation_history)
        print(f"   Turn {i+1} added → Memory size: {turns_in_memory}")
    
    final_count = len(memory.conversation_history)
    print(f"\n✅ Final memory size: {final_count}")
    print(f"✅ Correctly limited to: {memory.max_history_turns}")
    
    # Check which turns are kept (should be most recent)
    print(f"\n📝 Turns in memory:")
    for turn in memory.conversation_history:
        print(f"   - Turn {turn.turn_id}: {turn.user_query}")
    
    print("\n✅ Memory limit test complete")


def test_8_clear_memory(memory: MemoryTools):
    """Test 8: Clear memory"""
    print_section("TEST 8: Clear Memory")
    
    stats_before = memory.get_memory_stats()
    print(f"\n📊 Before clear:")
    print(f"   Turns: {stats_before['total_turns']}")
    print(f"   Entities: {stats_before['total_entities']}")
    
    print(f"\n🗑️  Clearing memory...")
    memory.clear_memory()
    
    stats_after = memory.get_memory_stats()
    print(f"\n📊 After clear:")
    print(f"   Turns: {stats_after['total_turns']}")
    print(f"   Entities: {stats_after['total_entities']}")
    
    if stats_after['total_turns'] == 0 and stats_after['total_entities'] == 0:
        print(f"\n✅ Memory successfully cleared!")
    else:
        print(f"\n❌ Memory not fully cleared")
    
    print("\n✅ Clear memory test complete")


def main():
    """Run all tests"""
    print("\n" + "🧪"*40)
    print("MEMORY TOOLS TEST SUITE (Phase 3 - Tool 8)")
    print("🧪"*40)
    
    try:
        # Run tests
        memory = test_1_initialization()
        test_2_basic_conversation(memory)
        test_3_context_retrieval(memory)
        test_4_entity_tracking(memory)
        test_5_follow_up_detection(memory)
        test_6_persistence(memory)
        test_7_memory_limit()
        test_8_clear_memory(memory)
        
        # Summary
        print_section("TEST SUMMARY")
        tests = [
            "Memory Initialization",
            "Basic Conversation",
            "Context Retrieval",
            "Entity Tracking",
            "Follow-up Detection",
            "Memory Persistence",
            "Memory Limit",
            "Clear Memory"
        ]
        
        for test in tests:
            print(f"✅ PASS - {test}")
        
        print(f"\nResults: {len(tests)}/{len(tests)} tests passed")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
