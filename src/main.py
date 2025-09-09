import sys
import os

try:
    from ichatbio.server import run_agent_server
    print("ichatbio imported successfully")
except ImportError as e:
    print(f"Failed to import ichatbio: {e}")
    sys.exit(1)

try:
    from src.col_agent import CatalogueOfLifeAgent
    print("col_agent imported successfully")
except ImportError as e:
    print(f"Failed to import col_agent: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Checking OpenAI API key...")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment")
        sys.exit(1)
    print("OPENAI_API_KEY found")
    
    try:
        print("Initializing agent...")
        agent = CatalogueOfLifeAgent()
        print("Agent initialized")
        
        print("Starting server on 0.0.0.0:29197...")
        run_agent_server(agent, host="0.0.0.0", port=29197)
    except Exception as e:
        print(f"Server failed to start: {e}")
        import traceback
        traceback.print_exc()