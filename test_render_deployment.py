"""
Test script for deployed Render backend
"""
import httpx
import json

# Replace this with your actual Render URL
RENDER_URL = "https://gigsama-backend.onrender.com"  # ← UPDATE THIS!

def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("Testing Root Endpoint (GET /)")
    print("="*60)
    
    response = httpx.get(f"{RENDER_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint (GET /health)")
    print("="*60)
    
    response = httpx.get(f"{RENDER_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_stats():
    """Test stats endpoint"""
    print("\n" + "="*60)
    print("Testing Stats Endpoint (GET /stats)")
    print("="*60)
    
    response = httpx.get(f"{RENDER_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_chat_deterministic():
    """Test deterministic chat (tool calling)"""
    print("\n" + "="*60)
    print("Testing Deterministic Chat (POST /chat)")
    print("="*60)
    
    payload = {
        "system": "You are a recruitment assistant. Convert requests to tool calls.",
        "messages": [
            {
                "role": "user",
                "content": "Find frontend developers in FinTech in Kenya"
            }
        ]
    }
    
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = httpx.post(
        f"{RENDER_URL}/chat",
        json=payload,
        timeout=60.0  # Longer timeout for inference
    )
    
    print(f"\nStatus: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_chat_creative():
    """Test creative chat (conversation)"""
    print("\n" + "="*60)
    print("Testing Creative Chat (POST /chat)")
    print("="*60)
    
    payload = {
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": "Explain what a backend developer does"
            }
        ]
    }
    
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = httpx.post(
        f"{RENDER_URL}/chat",
        json=payload,
        timeout=60.0
    )
    
    print(f"\nStatus: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_debug_routing():
    """Test routing debug endpoint"""
    print("\n" + "="*60)
    print("Testing Debug Routing (POST /debug/routing)")
    print("="*60)
    
    params = {
        "system": "You are a helpful assistant",
        "user_message": "Find developers in Kenya"
    }
    
    response = httpx.post(f"{RENDER_URL}/debug/routing", params=params)
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("RENDER BACKEND DEPLOYMENT TEST")
    print("🚀" * 30)
    print(f"\nTesting: {RENDER_URL}")
    
    if RENDER_URL == "https://gigsama-backend.onrender.com":
        print("\n⚠️  WARNING: Please update RENDER_URL with your actual Render URL!")
        print("You can find it in your Render dashboard.\n")
        exit(1)
    
    results = {}
    
    try:
        results['root'] = test_root()
    except Exception as e:
        print(f"❌ Root test failed: {str(e)}")
        results['root'] = False
    
    try:
        results['health'] = test_health()
    except Exception as e:
        print(f"❌ Health test failed: {str(e)}")
        results['health'] = False
    
    try:
        results['stats'] = test_stats()
    except Exception as e:
        print(f"❌ Stats test failed: {str(e)}")
        results['stats'] = False
    
    try:
        results['debug_routing'] = test_debug_routing()
    except Exception as e:
        print(f"❌ Debug routing test failed: {str(e)}")
        results['debug_routing'] = False
    
    try:
        results['chat_deterministic'] = test_chat_deterministic()
    except Exception as e:
        print(f"❌ Deterministic chat test failed: {str(e)}")
        results['chat_deterministic'] = False
    
    try:
        results['chat_creative'] = test_chat_creative()
    except Exception as e:
        print(f"❌ Creative chat test failed: {str(e)}")
        results['chat_creative'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your deployment is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
