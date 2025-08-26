import os
import json
from typing import Optional, List, override
from urllib.parse import urlencode

import dotenv
import instructor
import requests
from openai import AsyncOpenAI
from pydantic import BaseModel, HttpUrl, Field

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

# Load environment variables
dotenv.load_dotenv()

# Data model to hold search parameters extracted from the query
class CoLQueryParams(BaseModel):
    q: str = Field(..., description="Scientific or common name keyword")
    rank: Optional[str] = Field(None, description="Taxonomic rank")
    limit: Optional[int] = Field(10, description="Number of results to return")

# Model for each species entry
class NameInfo(BaseModel):
    scientificName: str
    rank: str
    link: Optional[str] = None
    acceptedName: Optional[str] = None
    classification: Optional[List[str]] = None

# Response structure after processing all results
class ColResponse(BaseModel):
    results: List[NameInfo]
    query_url: str
    total: int

# Parameters for the agent entrypoint
class ColAgentParameters(BaseModel):
    format: str = Field(default="detailed", description="Response format: 'detailed' or 'summary'")

class CatalogueOfLifeAgent(IChatBioAgent):
    
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Catalogue of Life Agent",
            description="Search and retrieve taxonomic information from the Catalogue of Life database using natural language queries.",
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="search_taxonomy",
                    description="Search for species, genera, or taxonomic information",
                    parameters=ColAgentParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[ColAgentParameters]):
        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            process: IChatBioAgentProcess
            
            try:
                await process.log("Starting taxonomic search process")
                
                # Set up OpenAI client with instructor
                openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                instructor_client = instructor.patch(openai_client)
                
                await process.log("Extracting search parameters from user query using GPT")
                
                # Extract structured params from user message using GPT
                query_instructions = """
                You are a helpful assistant extracting Catalogue of Life API parameters from user queries.
                Extract:
                - q: a keyword like a species or genus name
                - rank: taxonomic rank (like 'species', 'genus', etc.) if mentioned
                - limit: max number of results (default 10)
                """
                
                query_params: CoLQueryParams = await instructor_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": query_instructions},
                        {"role": "user", "content": request}
                    ],
                    response_model=CoLQueryParams,
                    temperature=0,
                    max_retries=3
                )
                
                await process.log(f"Extracted parameters: q='{query_params.q}', rank='{query_params.rank}', limit={query_params.limit}")
                
                # Build API request
                col_base_url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
                api_params = {
                    "q": query_params.q,
                    "limit": query_params.limit or 10
                }
                
                if query_params.rank:
                    api_params["rank"] = query_params.rank
                
                await process.log(f"Querying Catalogue of Life API: {col_base_url}")
                
                # Make the API request
                response = requests.get(col_base_url, params=api_params)
                query_url = response.url
                
                await process.log(f"API request completed with status code: {response.status_code}")
                
                if response.status_code != 200:
                    await process.log(f"API request failed: {response.status_code}")
                    await context.reply(f"Error accessing Catalogue of Life: HTTP {response.status_code}")
                    return
                
                # Process the response
                results = []
                total = 0
                
                data = response.json()
                total = data.get("total", 0)
                
                await process.log(f"Found {total} total matches in Catalogue of Life")
                
                if total == 0:
                    await context.reply(f"No results found for '{query_params.q}' in the Catalogue of Life database.")
                    return
                
                # Process each result
                await process.log("Processing search results")
                
                for item in data.get("result", []):
                    try:
                        usage = item.get("usage", {})
                        name_info = usage.get("name", {})
                        classification = [entry.get("name") for entry in item.get("classification", []) if entry.get("name")]

                        result = NameInfo(
                            scientificName=name_info.get("scientificName", "Unknown"),
                            rank=name_info.get("rank", "Unknown"),
                            link=f"https://www.catalogueoflife.org/data/taxon/{usage.get('id')}" if usage.get('id') else None,
                            acceptedName=usage.get("accepted", {}).get("name", {}).get("scientificName") if usage.get("accepted") else None,
                            classification=classification if classification else None
                        )
                        results.append(result)
                    except Exception as e:
                        await process.log(f"Skipping invalid result: {str(e)}")
                        continue
                
                # Create structured response
                structured_response = ColResponse(
                    results=results,
                    query_url=query_url,
                    total=total
                )
                
                await process.log(f"Successfully processed {len(results)} results")
                
                # Create artifact with the results
                await process.log("Creating results artifact")
                
                # Convert to dict for JSON serialization
                artifact_data = {
                    "query_info": {
                        "search_term": query_params.q,
                        "rank_filter": query_params.rank,
                        "query_url": query_url
                    },
                    "results": []
                }
                
                for result in structured_response.results:
                    result_dict = {
                        "scientificName": result.scientificName,
                        "rank": result.rank,
                        "link": result.link,
                        "acceptedName": result.acceptedName,
                        "classification": result.classification
                    }
                    artifact_data["results"].append(result_dict)
                
                artifact_data["total"] = structured_response.total
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Catalogue of Life search results for '{query_params.q}'",
                    content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                    uris=[query_url],
                    metadata={
                        "search_term": query_params.q,
                        "total_results": total,
                        "processed_results": len(results),
                        "source": "Catalogue of Life"
                    }
                )
                
                await process.log("Artifact created successfully")
                
                # Generate human-readable summary
                summary = self._generate_summary(structured_response, query_params)
                
                await process.log("Search process completed successfully")
                
            except Exception as e:
                await process.log(f"Error during search: {str(e)}")
                await context.reply(f"An error occurred while searching: {str(e)}")
                return
        
        # Send final response to user
        await context.reply(summary)
    
    def _generate_summary(self, response: ColResponse, params: CoLQueryParams) -> str:
        """Generate a human-readable summary of the results."""
        if not response.results:
            return f"No results found for '{params.q}' in the Catalogue of Life database."
        
        summary = f"Found {response.total} matches for '{params.q}' in Catalogue of Life.\n\n"
        
        # Show top results
        summary += "**Top Results:**\n"
        for i, result in enumerate(response.results[:5], 1):  # Show top 5
            accepted_info = f" (Accepted name: {result.acceptedName})" if result.acceptedName else ""
            classification_info = f" | Classification: {' > '.join(result.classification[:3])}" if result.classification else ""
            
            summary += f"{i}. **{result.scientificName}** ({result.rank}){accepted_info}\n"
            if result.link:
                summary += f"   🔗 [View on CoL]({result.link})\n"
            if classification_info:
                summary += f"   📊{classification_info}\n"
            summary += "\n"
        
        if len(response.results) > 5:
            summary += f"... and {len(response.results) - 5} more results.\n\n"
        
        summary += f"**Total found:** {response.total} records\n"
        summary += "**Source:** Catalogue of Life ChecklistBank API\n\n"
        summary += "📁 **Detailed results** are available in the generated artifact above, including full classifications and metadata."
        
        return summary


if __name__ == "__main__":
    from ichatbio.server import run_agent_server
    
    agent = CatalogueOfLifeAgent()
    print("Starting Catalogue of Life Agent...")
    print("Agent card available at: http://localhost:9999/.well-known/agent.json")
    print("Press Ctrl+C to stop the server")
    
    run_agent_server(agent, host="0.0.0.0", port=9999)