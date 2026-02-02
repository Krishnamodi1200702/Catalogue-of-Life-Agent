"""Quick single test"""
import requests
import json

query = "Search for Escherichia coli in Catalogue of Life"

response = requests.post(
    "http://localhost:8989/chat",
    json={"type": "user_text_message", "value": query}
)

data = response.json()

print("\n" + "="*80)
print(f"QUERY: {query}")
print("="*80 + "\n")

for msg in data:
    if msg["type"] == "ai_text_message":
        print(f"\n AI RESPONSE:\n{msg['value']}\n")
    
    elif msg["type"] == "ai_artifact_message":
        print(f"\n ARTIFACT:")
        print(f"   Description: {msg['value']['description']}")
        print(f"   Type: {msg['value']['mimetype']}")
        print(f"   Metadata: {json.dumps(msg['value']['meta'], indent=2)}\n")

print("="*80)