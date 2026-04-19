"""
Simple HTTP client for testing the Catalogue of Life agent.
Extracts response from iChatBio history array.
"""

import requests
import uuid
from typing import Optional


def call_agent(query: str, timeout: int = 30) -> Optional[str]:
    """
    Call the COL agent via JSON-RPC.
    
    Args:
        query: User query
        timeout: Request timeout in seconds
        
    Returns:
        Response text or None
    """
    url = "http://127.0.0.1:9999/"
    message_id = str(uuid.uuid4())
    
    payload = {
        "jsonrpc": "2.0",
        "id": "test-request",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "data",
                        "data": {
                            "entrypoint": {
                                "id": "run"
                            },
                            "parameters": {}
                        }
                    }
                ]
            }
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # Extract from result.history array
        if "result" in data and "history" in data["result"]:
            history = data["result"]["history"]
            
            # Find the last agent message with direct_response type
            for message in reversed(history):
                if message.get("role") == "agent":
                    parts = message.get("parts", [])
                    for part in parts:
                        # Look for direct_response metadata
                        metadata = part.get("metadata", {})
                        if metadata.get("ichatbio_type") == "direct_response":
                            if part.get("kind") == "text":
                                return part.get("text")
        
        return None
        
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except Exception as e:
        return None