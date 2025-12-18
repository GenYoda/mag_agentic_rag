# save_kb_chunks.py
"""
Save all KB chunks to a text file for inspection.
"""
from tools.kb_tools import KBTools
from pathlib import Path

# Load KB
kb = KBTools()
result = kb.load_index()

if not result['success']:
    print(f"❌ Failed to load KB: {result.get('error')}")
    exit(1)

# Create output file
output_file = Path("kb_chunks_export.txt")

print(f"\n📄 Exporting {len(kb.chunks)} chunks to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    # Write header
    f.write("="*80 + "\n")
    f.write("KNOWLEDGE BASE CHUNKS EXPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total Chunks: {len(kb.chunks)}\n")
    f.write(f"KB Version: {kb.metadata.get('kb_version', 'unknown')}\n")
    f.write(f"Created: {kb.metadata.get('created_at', 'unknown')}\n")
    f.write("\n" + "="*80 + "\n\n")
    
    # Write each chunk
    for i, chunk in enumerate(kb.chunks, 1):
        f.write(f"CHUNK {i}\n")
        f.write("-" * 80 + "\n")
        
        # Metadata
        metadata = chunk.get('metadata', {})
        f.write(f"Source: {metadata.get('source', 'Unknown')}\n")
        f.write(f"Page: {metadata.get('page_numbers', 'N/A')}\n")
        f.write(f"Chunk Index: {chunk.get('chunk_index', 'N/A')}\n")
        f.write(f"File Hash: {metadata.get('file_hash', 'N/A')[:16]}...\n")
        
        # Text content
        f.write("\nText:\n")
        f.write(chunk['text'])
        f.write("\n\n" + "="*80 + "\n\n")

print(f"✅ Exported to {output_file.absolute()}")
print(f"📊 Total chunks: {len(kb.chunks)}")
print(f"\n💡 Upload this file and I'll suggest better test queries!")
