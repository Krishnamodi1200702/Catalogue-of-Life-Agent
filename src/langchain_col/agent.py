"""
Catalogue of Life Agent - LangChain Architecture
================================================

This agent runs as a LangChain tool-calling agent, providing access to the 
Catalogue of Life database through the ChecklistBank API. The LLM decides 
which tools to call based on user queries.

Author: Krishna Modi
Version: 3.1.0-langchain
License: MIT
"""

import logging
from typing_extensions import override
import langchain.agents
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from starlette.applications import Starlette
from dotenv import load_dotenv

load_dotenv()

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint

from .context import current_context
from .tools import (
    search_species,
    get_taxon_details,
    get_synonyms,
    get_vernacular_names,
    get_classification,
    get_taxon_children,
    get_distribution,
    get_references,
    abort,
    finish,
    reset_tool_tracker,
)

logger = logging.getLogger(__name__)


class CatalogueOfLifeLangChainAgent(IChatBioAgent):
    """
    LangChain-based agent for accessing Catalogue of Life data.
    
    The agent uses an LLM to understand user queries and orchestrate
    calls to various ChecklistBank API endpoints through LangChain tools.
    """
    
    def __init__(self):
        self.langchain_agent = langchain.agents.create_agent(
            model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            tools=[
                search_species,
                get_taxon_details,
                get_synonyms,
                get_vernacular_names,
                get_classification,
                get_taxon_children,
                get_distribution,
                get_references,
                abort,
                finish,
            ],
                        system_prompt=(
                "You are a biodiversity assistant with access to the Catalogue of Life database.\n\n"
                
                "=== CRITICAL: EXECUTION RULES ===\n"
                "1. Call ONE tool to gather information\n"
                "2. Tool returns \"\" (empty string) = success, data is in artifacts\n"
                "3. IMMEDIATELY call finish() with a summary\n"
                "MAXIMUM: 2 tool calls total (1 data tool + finish), NEVER more.\n\n"
                
                "=== TOOL BEHAVIOR ===\n"
                "- All data tools create artifacts and return \"\" (empty string)\n"
                "- Empty string = success, NOT failure\n"
                "- Do NOT call the same tool twice\n"
                "- Do NOT call multiple tools for the same query\n"
                "- After ANY tool returns \"\", call finish IMMEDIATELY\n\n"
                
                "=== EXAMPLES ===\n\n"
                "User: \"Tell me about Panthera leo\"\n"
                "Step 1: Call get_taxon_details(\"Panthera leo\")\n"
                "Step 2: Tool returns \"\" (success)\n"
                "Step 3: Call finish(\"Panthera leo (Lion) is a large cat species in the family Felidae. Complete taxonomic details and distribution data are available in the artifact.\")\n"
                "DONE - 2 tool calls total.\n\n"
                
                "User: \"Where is the black rat found?\"\n"
                "Step 1: Call get_distribution(\"Rattus rattus\")\n"
                "Step 2: Tool returns \"\" (success)\n"
                "Step 3: Call finish(\"The black rat (Rattus rattus) is found across multiple regions. Complete distribution data is available in the artifact.\")\n"
                "DONE - 2 tool calls total.\n\n"
                
                "User: \"What family is Panthera leo in?\"\n"
                "Step 1: Call get_classification(\"Panthera leo\")\n"
                "Step 2: Tool returns \"\" (success)\n"
                "Step 3: Call finish(\"Panthera leo belongs to the family Felidae. Complete classification hierarchy is available in the artifact.\")\n"
                "DONE - 2 tool calls total.\n\n"
                
                "User: \"Find xyz123invalid\"\n"
                "Step 1: Call search_species(\"xyz123invalid\")\n"
                "Step 2: Tool returns \"\" but creates no artifact (species not found)\n"
                "Step 3: Call abort(\"Species 'xyz123invalid' not found in the Catalogue of Life database. Please check the spelling or try a different name.\")\n"
                "DONE - 2 tool calls total.\n\n"
                
                "=== FINISH MESSAGE FORMAT ===\n"
                "Keep responses concise (2-3 sentences):\n"
                "1. Direct answer to the question\n"
                "2. Mention that complete data is in artifacts\n"
                "Example: \"Panthera leo (Lion) is found in Africa and parts of India. Complete distribution data with regional details is available in the artifact.\"\n\n"
                
                "=== CRITICAL RULES ===\n"
                "1. NEVER call more than ONE data tool per request\n"
                "2. After ANY tool returns \"\", call finish IMMEDIATELY\n"
                "3. NEVER generate species data from memory - always use tools\n"
                "4. If tool returns \"\" but you're unsure, call finish anyway - artifacts have the data\n"
                "5. Empty string (\"\") = SUCCESS, not failure\n\n"
                
                "=== FORBIDDEN BEHAVIORS ===\n"
                "❌ Calling 2+ data tools (search + details, details + distribution, etc.)\n"
                "❌ Calling tools after you already have an answer\n"
                "❌ Responding without calling finish() or abort()\n"
                "❌ Treating \"\" as an error (it means success)\n"
                "❌ Exploring with multiple tools \"to be thorough\"\n\n"
                
                "Remember: ONE data tool → finish. That's it. Tools return \"\" = success."
            ),
        )
    
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Catalogue of Life Agent",
            description=(
                "Access the Catalogue of Life database for species information, "
                "taxonomic classifications, synonyms, vernacular names, distribution data, "
                "and bibliographic references. Powered by LangChain for intelligent query understanding."
            ),
            icon=None,
            documentation_url="https://api.checklistbank.org",
            url=None,
            entrypoints=[
                AgentEntrypoint(
                    id="run",
                    description="Ask questions about species, taxonomy, and biodiversity data using natural language.",
                    parameters=None,
                )
            ],
        )
    
    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: BaseModel,
    ):
        """
        Execute the LangChain agent with proper control flow.
        
        Tools handle all responses via finish() and abort().
        This method only orchestrates execution.
        """
        reset_tool_tracker()
        token = current_context.set(context)
        
        try:
            await self.langchain_agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": request}
                    ]
                },
                config={"recursion_limit": 10}
            )
            
        except Exception as e:
            logger.exception(f"Error running LangChain agent: {e}")
            await context.reply(
                "An error occurred while processing your request. Please try again."
            )
        
        finally:
            current_context.reset(token)


def create_app() -> Starlette:
    """Create the Starlette app for this agent."""
    agent = CatalogueOfLifeLangChainAgent()
    app = build_agent_app(agent)
    return app