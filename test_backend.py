import requests

response = requests.post(
    "http://localhost:8001/chat",
    json={
        "system": "You are a helpful assistant. Extract job parameters as JSON.",
        "messages": [
            {"role": "user", "content": "I want to hire Backend developers in Kenya with HealthTech experience"}
        ]
    }
)

print(response.json())