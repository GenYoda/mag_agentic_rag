"""
================================================================================
AZURE OPENAI CONNECTION TEST
================================================================================
Quick diagnostic script to test Azure OpenAI connectivity
Tests: DNS, Network, API credentials, Simple completion
================================================================================
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_test(name):
    print(f"\n→ {name}")

def print_pass(msg="PASS"):
    print(f"  ✅ {msg}")

def print_fail(msg="FAIL"):
    print(f"  ❌ {msg}")

# ============================================================================
# Test 1: Check Environment Variables
# ============================================================================

def test_env_vars():
    print_header("TEST 1: ENVIRONMENT VARIABLES")
    
    from config.settings import azure_settings
    
    print_test("Checking Azure OpenAI configuration...")
    
    endpoint = azure_settings.azure_openai_endpoint
    key = azure_settings.azure_openai_key
    deployment = azure_settings.azure_openai_chat_deployment
    api_version = azure_settings.azure_openai_api_version
    
    print(f"  Endpoint: {endpoint}")
    print(f"  Deployment: {deployment}")
    print(f"  API Version: {api_version}")
    print(f"  Key exists: {key is not None and len(key) > 0}")
    
    if not endpoint:
        print_fail("AZURE_OPENAI_ENDPOINT not set!")
        return False
    
    if not key:
        print_fail("AZURE_OPENAI_KEY not set!")
        return False
    
    if not deployment:
        print_fail("AZURE_OPENAI_CHAT_DEPLOYMENT not set!")
        return False
    
    print_pass("All environment variables configured")
    return True

# ============================================================================
# Test 2: DNS Resolution
# ============================================================================

def test_dns_resolution():
    print_header("TEST 2: DNS RESOLUTION")
    
    import socket
    from config.settings import azure_settings
    
    # Extract hostname from endpoint
    endpoint = azure_settings.azure_openai_endpoint
    hostname = endpoint.replace('https://', '').replace('http://', '').rstrip('/')
    
    print_test(f"Resolving DNS for: {hostname}")
    
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"  IP Address: {ip_address}")
        print_pass("DNS resolution successful")
        return True
    except socket.gaierror as e:
        print_fail(f"DNS resolution failed: {e}")
        print("\n  Possible causes:")
        print("  - No internet connection")
        print("  - Corporate firewall blocking DNS")
        print("  - VPN required but not connected")
        print("  - Incorrect endpoint URL")
        return False
    except Exception as e:
        print_fail(f"DNS test error: {e}")
        return False

# ============================================================================
# Test 3: Network Connectivity (Ping)
# ============================================================================

def test_network_connectivity():
    print_header("TEST 3: NETWORK CONNECTIVITY")
    
    import socket
    from config.settings import azure_settings
    
    endpoint = azure_settings.azure_openai_endpoint
    hostname = endpoint.replace('https://', '').replace('http://', '').rstrip('/')
    
    print_test(f"Testing HTTPS connection to: {hostname}")
    
    try:
        # Try to connect to port 443 (HTTPS)
        sock = socket.create_connection((hostname, 443), timeout=10)
        sock.close()
        print_pass("Network connectivity successful")
        return True
    except socket.timeout:
        print_fail("Connection timeout - host unreachable")
        print("  Check: VPN, firewall, internet connection")
        return False
    except socket.gaierror:
        print_fail("DNS resolution failed (already detected)")
        return False
    except Exception as e:
        print_fail(f"Connection failed: {e}")
        return False

# ============================================================================
# Test 4: Azure OpenAI API Connection
# ============================================================================

def test_azure_openai_api():
    print_header("TEST 4: AZURE OPENAI API")
    
    from config.settings import azure_settings
    
    print_test("Testing Azure OpenAI API with simple completion...")
    
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=azure_settings.azure_openai_key,
            api_version=azure_settings.azure_openai_api_version,
            azure_endpoint=azure_settings.azure_openai_endpoint
        )
        
        print("  Sending test request...")
        
        response = client.chat.completions.create(
            model=azure_settings.azure_openai_chat_deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API test successful' and nothing else."}
            ],
            temperature=0,
            max_tokens=50
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"  Response: {answer}")
        
        print_pass("Azure OpenAI API working!")
        return True
        
    except Exception as e:
        print_fail(f"API call failed: {e}")
        
        error_str = str(e).lower()
        
        if 'failed to resolve' in error_str or 'getaddrinfo failed' in error_str:
            print("\n  🔴 DNS RESOLUTION ERROR")
            print("  This is a NETWORK issue, not a code issue.")
            print("\n  Solutions:")
            print("  1. Connect to VPN if required")
            print("  2. Check internet connection")
            print("  3. Check corporate firewall settings")
            print("  4. Try from a different network")
            
        elif 'unauthorized' in error_str or '401' in error_str:
            print("\n  🔴 AUTHENTICATION ERROR")
            print("  Solutions:")
            print("  1. Check AZURE_OPENAI_KEY is correct")
            print("  2. Verify key hasn't expired")
            
        elif 'not found' in error_str or '404' in error_str:
            print("\n  🔴 DEPLOYMENT NOT FOUND")
            print("  Solutions:")
            print("  1. Check deployment name is correct")
            print("  2. Verify deployment exists in Azure portal")
            
        return False

# ============================================================================
# Test 5: Internet Connectivity (Fallback)
# ============================================================================

def test_internet():
    print_header("TEST 5: INTERNET CONNECTIVITY (FALLBACK)")
    
    import socket
    
    print_test("Testing connection to google.com...")
    
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        print_pass("Internet connection working")
        return True
    except Exception as e:
        print_fail(f"No internet connection: {e}")
        print("\n  🔴 NO INTERNET CONNECTION")
        print("  You are completely offline or behind a strict firewall.")
        return False

# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    print("\n" + "=" * 70)
    print("  AZURE OPENAI CONNECTION DIAGNOSTIC")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Environment variables
    results['env_vars'] = test_env_vars()
    if not results['env_vars']:
        print("\n⚠️  Fix environment variables before proceeding")
        return
    
    # Test 2: DNS resolution
    results['dns'] = test_dns_resolution()
    
    # Test 3: Network connectivity
    if results['dns']:
        results['network'] = test_network_connectivity()
    else:
        results['network'] = False
        print_header("TEST 3: NETWORK CONNECTIVITY")
        print("  ⏭️  Skipped (DNS failed)")
    
    # Test 4: Azure API
    if results['network']:
        results['api'] = test_azure_openai_api()
    else:
        results['api'] = False
        print_header("TEST 4: AZURE OPENAI API")
        print("  ⏭️  Skipped (Network failed)")
    
    # Test 5: General internet (fallback)
    if not results['dns']:
        results['internet'] = test_internet()
    
    # Summary
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    print(f"\n  Environment Variables: {'✅ PASS' if results.get('env_vars') else '❌ FAIL'}")
    print(f"  DNS Resolution:        {'✅ PASS' if results.get('dns') else '❌ FAIL'}")
    print(f"  Network Connectivity:  {'✅ PASS' if results.get('network') else '❌ FAIL'}")
    print(f"  Azure OpenAI API:      {'✅ PASS' if results.get('api') else '❌ FAIL'}")
    
    if results.get('api'):
        print("\n🎉 ALL TESTS PASSED - Azure OpenAI is working!")
        print("   Your RAG system should work now.")
    else:
        print("\n⚠️  TESTS FAILED - See errors above")
        
        if not results.get('dns'):
            print("\n📌 PRIMARY ISSUE: DNS RESOLUTION")
            print("   → This is a NETWORK problem, not a code problem")
            print("   → You need to connect to VPN or fix network access")
        elif not results.get('network'):
            print("\n📌 PRIMARY ISSUE: NETWORK CONNECTIVITY")
            print("   → Azure endpoint is blocked")
            print("   → Check firewall/VPN settings")
        elif not results.get('api'):
            print("\n📌 PRIMARY ISSUE: API AUTHENTICATION/CONFIGURATION")
            print("   → Check credentials and deployment name")
    
    print("=" * 70 + "\n")

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
