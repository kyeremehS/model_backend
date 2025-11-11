"""
End-to-End Test for Dual Backend + Modal Inference Pipeline
Tests the full flow from backend to Modal inference service
"""

import requests
import json
import time

print("="*80)
print("🧪 END-TO-END INFERENCE TEST")
print("="*80)

# Configuration
BACKEND_URL = "http://localhost:8001"
MODAL_URL = "https://affum3331--gigsama-backend-fastapi-app.modal.run"

print("\n📋 Test Configuration:")
print(f"   Backend URL: {BACKEND_URL}")
print(f"   Modal URL: {MODAL_URL}")

# Test 1: Check Modal Health
print("\n" + "="*80)
print("TEST 1: Modal Inference Service Health Check")
print("="*80)

try:
    response = requests.get(f"{MODAL_URL}/health", timeout=10)
    response.raise_for_status()
    health_data = response.json()
    
    print("✅ Modal service is healthy!")
    print(f"   Status: {health_data.get('status')}")
    print(f"   Service: {health_data.get('service')}")
    print(f"   Model: {health_data.get('model')}")
    print(f"   GPU: {health_data.get('gpu')}")
    
except Exception as e:
    print(f"❌ Modal service health check failed: {str(e)}")
    exit(1)

# Test 2: Direct Modal Inference
print("\n" + "="*80)
print("TEST 2: Direct Modal Inference")
print("="*80)

test_prompt = "What is artificial intelligence? Answer in one sentence."
print(f"📝 Test prompt: {test_prompt}")

try:
    start_time = time.time()
    
    response = requests.post(
        f"{MODAL_URL}/inference",
        json={
            "prompt": test_prompt,
            "temperature": 0.0,
            "max_tokens": 100,
            "top_p": 0.9
        },
        timeout=60
    )
    response.raise_for_status()
    result = response.json()
    
    latency = (time.time() - start_time) * 1000
    
    if result.get("error"):
        print(f"❌ Modal inference failed: {result.get('message')}")
        exit(1)
    
    print("✅ Modal inference successful!")
    print(f"   Response: {result.get('text')[:200]}...")
    print(f"   Tokens generated: {result.get('tokens')}")
    print(f"   Finish reason: {result.get('finish_reason')}")
    print(f"   Latency: {latency:.2f}ms")
    
except Exception as e:
    print(f"❌ Direct Modal inference failed: {str(e)}")
    exit(1)

# Test 3: Backend Health (if running)
print("\n" + "="*80)
print("TEST 3: Backend Service Health Check (Optional)")
print("="*80)

try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    response.raise_for_status()
    health_data = response.json()
    
    print("✅ Backend service is running!")
    print(f"   Status: {health_data.get('status')}")
    print(f"   Modal URL: {health_data.get('modal_url')}")
    
    # Test 4: End-to-End Chat Request
    print("\n" + "="*80)
    print("TEST 4: End-to-End Chat Request via Backend")
    print("="*80)
    
    chat_request = {
        "system": "You are a helpful assistant. Extract job parameters as JSON.",
        "messages": [
            {
                "role": "user",
                "content": "I want to hire Backend developers in Kenya with HealthTech experience"
            }
        ]
    }
    
    print(f"📝 Test message: {chat_request['messages'][0]['content']}")
    
    start_time = time.time()
    
    response = requests.post(
        f"{BACKEND_URL}/chat",
        json=chat_request,
        timeout=60
    )
    response.raise_for_status()
    result = response.json()
    
    latency = (time.time() - start_time) * 1000
    
    print("✅ End-to-end chat successful!")
    print(f"   Response: {result.get('text')[:200]}...")
    print(f"   Input tokens: {result.get('tokens_input')}")
    print(f"   Output tokens: {result.get('tokens_output')}")
    print(f"   Finish reason: {result.get('finish_reason')}")
    print(f"   Backend used: {result.get('backend_used')}")
    print(f"   Routing reason: {result.get('routing_reason')}")
    print(f"   Total latency: {latency:.2f}ms")
    print(f"   Request ID: {result.get('request_id')}")
    
except requests.exceptions.ConnectionError:
    print("⚠️  Backend service is not running")
    print("   To start backend: python dual_backend.py")
    print("   Then run this test again")
    
except Exception as e:
    print(f"❌ Backend test failed: {str(e)}")

# Summary
print("\n" + "="*80)
print("📊 TEST SUMMARY")
print("="*80)
print("✅ Modal inference service: OPERATIONAL")
print("✅ Direct inference: WORKING")
print("ℹ️  Backend service: Check above for status")
print("\n🎉 Core infrastructure is fully functional!")
print("="*80)
