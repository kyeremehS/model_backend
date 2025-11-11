"""
Simple test script for /chat endpoint
"""
import httpx
import json

# Your Render URL
RENDER_URL = "https://gigsama-backend.onrender.com"

def test_chat(user_message: str, system_prompt: str = "You are a helpful assistant."):
    """Send a message to the chat endpoint and print the response"""
    
    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}")
    
    payload = {
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_message
            }
        ]
    }
    
    try:
        response = httpx.post(
            f"{RENDER_URL}/chat",
            json=payload,
            timeout=60.0
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n Response ({data['backend_used']} backend):")
            print(f"\n{data['text']}\n")
            print(f"  Latency: {data['latency_ms']:.0f}ms")
            print(f" Tokens: {data['tokens_input']} in, {data['tokens_output']} out")
        else:
            print(f"\n Error {response.status_code}:")
            print(response.text)
    
    except Exception as e:
        print(f"\n Error: {str(e)}")

if __name__ == "__main__":
    print("\n Testing Chat Endpoint")
    print(f"URL: {RENDER_URL}/chat\n")
    
    # Test 1: Simple conversation
    test_chat("Hello, how are you?")
    
    # Test 2: Tool calling
    test_chat(
        "Find backend developers in FinTech in Nairobi",
        "You are a recruitment assistant. Convert requests to tool calls."
    )
    
    # Test 3: Another conversation
    test_chat("What is Python?")
    
    print("\n" + "="*60)
    print("✅ All tests complete!")
    print("="*60 + "\n")
