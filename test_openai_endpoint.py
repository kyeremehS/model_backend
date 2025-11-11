"""
Test the OpenAI-compatible endpoint
"""
import httpx
import json

RENDER_URL = "https://gigsama-backend.onrender.com"

def test_openai_chat():
    """Test OpenAI-compatible endpoint with simple chat"""
    print("\n" + "="*60)
    print("Testing OpenAI-compatible endpoint: /v1/chat/completions")
    print("="*60)
    
    payload = {
        "model": "Qwen3-1.7B",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    print(f"\nRequest:\n{json.dumps(payload, indent=2)}")
    
    try:
        response = httpx.post(
            f"{RENDER_URL}/v1/chat/completions",
            json=payload,
            timeout=60.0
        )
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Response (OpenAI format):")
            print(json.dumps(data, indent=2))
            
            # Extract the actual message
            message = data["choices"][0]["message"]["content"]
            print(f"\n💬 Message: {message}")
            
            # Show usage
            usage = data["usage"]
            print(f"\n📊 Usage: {usage['prompt_tokens']} in + {usage['completion_tokens']} out = {usage['total_tokens']} total")
        else:
            print(f"\n❌ Error:")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

def test_openai_with_tools():
    """Test OpenAI-compatible endpoint with tool calling"""
    print("\n" + "="*60)
    print("Testing OpenAI endpoint with tools")
    print("="*60)
    
    payload = {
        "model": "Qwen3-1.7B",
        "messages": [
            {"role": "user", "content": "Find backend developers in FinTech in Nairobi"}
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_candidates",
                    "description": "Search for candidates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "industry": {"type": "string"},
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }
    
    print(f"\nRequest:\n{json.dumps(payload, indent=2)}")
    
    try:
        response = httpx.post(
            f"{RENDER_URL}/v1/chat/completions",
            json=payload,
            timeout=60.0
        )
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Response (OpenAI format):")
            print(json.dumps(data, indent=2))
        else:
            print(f"\n❌ Error:")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\n🚀 Testing OpenAI-Compatible Endpoint")
    print(f"URL: {RENDER_URL}/v1/chat/completions\n")
    
    # Test 1: Simple chat
    test_openai_chat()
    
    # Test 2: With tools
    test_openai_with_tools()
    
    print("\n" + "="*60)
    print("✅ Tests complete!")
    print("="*60 + "\n")
