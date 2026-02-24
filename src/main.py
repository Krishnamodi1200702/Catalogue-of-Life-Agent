import sys
import os
from starlette.applications import Starlette
from ichatbio.server import build_agent_app


try:
    from src.col_agent import CatalogueOfLifeAgent
    print("col_agent imported successfully")
except ImportError as e:
    print(f"Failed to import col_agent: {e}")
    sys.exit(1)


def create_app() -> Starlette:
    print("Checking OpenAI API key...")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment")
        sys.exit(1)
    print("OPENAI_API_KEY found")
    
    agent = CatalogueOfLifeAgent()
    app = build_agent_app(agent)
    return app