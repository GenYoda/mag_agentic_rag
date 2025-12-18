"""Quick network connectivity test for Azure Document Intelligence"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.azure_clients import test_document_intelligence_connection

print("Testing Azure Document Intelligence connection...\n")

result = test_document_intelligence_connection()

if result['success']:
    print("✅ Connection successful!")
    print(f"   Endpoint: {result['endpoint']}")
else:
    print("❌ Connection failed!")
    print(f"   Error: {result['error']}")
    print("\n💡 Troubleshooting:")
    print("   1. Connect to VPN if required")
    print("   2. Check internet connection")
    print("   3. Verify AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT in .env")
