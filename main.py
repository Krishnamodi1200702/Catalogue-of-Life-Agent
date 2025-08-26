from ichatbio.server import run_agent_server
from col_agent_sdk import CatalogueOfLifeAgent

if __name__ == "__main__":
    agent = CatalogueOfLifeAgent()
    run_agent_server(agent, host="0.0.0.0", port=9999)