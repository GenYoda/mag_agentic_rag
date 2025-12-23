# rebuild_kb.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tools.kb_tools import KBTools

print("\n🔨 Rebuilding Knowledge Base...")
print("="*80)

kb = KBTools()

result = kb.build_index(
    input_folder="data/input",
    force=True,  # Force rebuild
    recursive=True
)

if result['success']:
    print(f"\n✅ SUCCESS!")
    print(f"   PDFs Processed: {result['pdfs_processed']}")
    print(f"   Chunks Indexed: {result['chunks_indexed']}")
    print(f"   Processing Time: {result['processing_time']}s")
    print(f"   KB Version: {result['kb_version']}")
    
    if result['errors']:
        print(f"\n⚠️  Errors: {len(result['errors'])}")
        for pdf, error in result['errors'].items():
            print(f"   {Path(pdf).name}: {error}")
else:
    print(f"\n❌ FAILED: {result.get('error')}")
