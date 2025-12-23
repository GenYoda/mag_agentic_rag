"""
Reset Knowledge Base - Clear all KB data for fresh start
"""

import sys
from pathlib import Path

sys.path.append('.')

from tools.kb_tools import KBTools

print("="*70)
print("RESET KNOWLEDGE BASE")
print("="*70)
print("\n⚠️  WARNING: This will delete:")
print("   - FAISS index")
print("   - All chunks")
print("   - All metadata")
print("   - PDF tracker (all hashes)")
print("\nYou will need to rebuild the index from scratch.")

response = input("\n❓ Are you sure? (yes/no): ").strip().lower()

if response != 'yes':
    print("\n❌ Reset cancelled")
    sys.exit(0)

print("\n🔄 Resetting KB...")

kb = KBTools()
result = kb.reset_kb()

if result.get('success'):
    print(f"\n✅ {result.get('message')}")
    print(f"\n📁 Files deleted:")
    for file in result.get('files_deleted', []):
        print(f"   - {file}")
    print("\n💡 Run your main_runner.py to rebuild the index")
else:
    print(f"\n❌ Reset failed: {result.get('error')}")
    sys.exit(1)

print("="*70)
