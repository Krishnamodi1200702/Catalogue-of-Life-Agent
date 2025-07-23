import os
import urllib.parse
from typing import Optional, Literal, override
from urllib.parse import urlencode
import json

import dotenv
import instructor
import pydantic
import requests
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

dotenv.load_dotenv()

TaxonomicResponseFormat = Literal["json", "summary"]

class GetTaxonomicDataParameters(BaseModel):
    format: TaxonomicResponseFormat = "json"

class TaxonModel(BaseModel):
    """API parameters for Catalogue of Life."""
    search_term: Optional[str] = Field(None,
                                      description="Scientific name, common name, or taxonomic keyword to search for.",
                                      examples=["Ursus maritimus", "polar bear", "Felidae"])
    rank: Optional[str] = Field(None, description="Taxonomic rank filter (species, genus, family, etc.)")
    limit: Optional[int] = Field(20, description="Maximum number of results to return. Default is 20.")

    @pydantic.field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        if v <= 0:
            raise ValueError("limit must be positive")
        return v

    def to_url(self, format: TaxonomicResponseFormat):
        url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
        params = {}

        if format == "summary":
            params |= {"format": "summary"}

        if self.search_term:
            params |= {"q": self.search_term}

        if self.rank:
            params |= {"rank": self.rank}

        if self.limit:
            params |= {"limit": self.limit}

        if params:
            url += "?" + urlencode(params)

        return url

class CatalogueOfLifeAgent(IChatBioAgent):
    
    @override
    def get_agent_card(self) -> AgentCard:
        # Railway URL detection
        railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
        if railway_domain:
            agent_url = f"https://{railway_domain}"
        else:
            port = int(os.environ.get("PORT", 3000))
            agent_url = f"http://localhost:{port}"
            
        return AgentCard(
            name="Catalogue of Life",
            description="Retrieves comprehensive taxonomic and species data from the Catalogue of Life database.",
            icon=None,
            url=agent_url,
            entrypoints=[
                AgentEntrypoint(
                    id="get_taxonomic_data",
                    description="Returns taxonomic information for species, genera, or higher taxa",
                    parameters=GetTaxonomicDataParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        # Start a process to log the agent's taxonomic search actions
        async with context.begin_process(summary="Searching for taxonomic data") as process:
            process: IChatBioAgentProcess
            
            try:
                # Log the initial step
                await process.log("Analyzing user query and extracting taxonomic search parameters")
                
                # Set up OpenAI client for parameter extraction with Navigator AI
                openai_client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL")  # Navigator AI endpoint
                )
                instructor_client = instructor.patch(openai_client)

                # Extract structured query parameters from user request
                await process.log("Using Navigator AI to convert natural language to Catalogue of Life API parameters")
                
                # Try gpt-4o first, fallback to gpt-3.5-turbo if needed
                try:
                    taxon: TaxonModel = await instructor_client.chat.completions.create(
                        model="gpt-4o",  # Use supported Navigator AI model
                        response_model=TaxonModel,
                        messages=[
                            {"role": "system",
                             "content": "You translate user requests into Catalogue of Life API parameters for taxonomic searches. Focus on extracting scientific names, common names, or taxonomic groups."},
                            {"role": "user", "content": request}
                        ],
                        max_retries=3
                    )
                except Exception as model_error:
                    await process.log(f"gpt-4o failed, trying gpt-3.5-turbo: {model_error}")
                    taxon: TaxonModel = await instructor_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        response_model=TaxonModel,
                        messages=[
                            {"role": "system",
                             "content": "You translate user requests into Catalogue of Life API parameters for taxonomic searches. Focus on extracting scientific names, common names, or taxonomic groups."},
                            {"role": "user", "content": request}
                        ],
                        max_retries=3
                    )

                # Build the API URL
                api_url = taxon.to_url(params.format if params else "json")
                
                await process.log(f"Built API query URL with parameters: {taxon.model_dump(exclude_none=True)}")
                
                # Log the API call
                await process.log(f"Querying Catalogue of Life database at: {api_url}")
                
                # Execute the API request
                response = requests.get(api_url)
                
                if response.status_code != 200:
                    await process.log(f"API request failed with status {response.status_code}")
                    await context.reply(f"Error accessing Catalogue of Life database: HTTP {response.status_code}")
                    return

                # Parse and log results
                raw_data = response.json()
                total_results = raw_data.get("total", 0)
                
                await process.log(f"Successfully retrieved {total_results} matching taxa from database")
                
                if total_results == 0:
                    await process.log("No taxonomic data found for the specified query")
                    await context.reply("No taxonomic data found for the specified query in Catalogue of Life.")
                    return

                # Process and structure the data
                await process.log("Processing and structuring taxonomic data")
                
                structured_data = self._process_taxonomic_data(raw_data, taxon, params)
                
                # Create artifact with the taxonomic data
                await process.log("Creating taxonomic data artifact")
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Taxonomic data for \"{taxon.search_term}\" from Catalogue of Life" if taxon.search_term else "Taxonomic search results",
                    uris=[api_url],
                    metadata={
                        "api_query_url": api_url,
                        "total_taxa_found": total_results,
                        "search_term": taxon.search_term,
                        "taxonomic_rank": taxon.rank,
                        "source": "Catalogue of Life ChecklistBank API",
                        "ai_service": "Navigator AI (UF)"
                    }
                )
                
                # Generate summary for the user
                await process.log("Generating human-readable summary with Navigator AI")
                
                summary_text = await self._generate_summary_with_ai(instructor_client, structured_data, taxon, total_results)
                
                await process.log("Taxonomic search completed successfully")

            except InstructorRetryException as e:
                await process.log("Failed to process taxonomic query with Navigator AI")
                await context.reply("Sorry, I couldn't process your taxonomic query. Please try rephrasing your request.")
                return
            except Exception as e:
                await process.log(f"Unexpected error during taxonomic search: {str(e)}")
                await context.reply(f"An error occurred while retrieving taxonomic data: {str(e)}")
                return

        # Reply to the user with the summary
        await context.reply(summary_text)

    def _process_taxonomic_data(self, raw_data: dict, query: TaxonModel, params: Optional[BaseModel]) -> dict:
        """Transform raw Catalogue of Life API response into structured taxonomic data."""
        processed_results = []
        
        for item in raw_data.get("result", []):
            usage = item.get("usage", {})
            name_info = usage.get("name", {})
            
            # Extract core taxonomic information
            taxon_entry = {
                "scientific_name": name_info.get("scientificName"),
                "taxonomic_rank": name_info.get("rank"),
                "authorship": name_info.get("authorship"),
                "nomenclatural_status": usage.get("status"),
                "col_id": usage.get("id")
            }
            
            # Add accepted name if this is a synonym
            if usage.get("accepted"):
                accepted_info = usage.get("accepted", {}).get("name", {})
                taxon_entry["accepted_name"] = accepted_info.get("scientificName")
                taxon_entry["is_synonym"] = True
            else:
                taxon_entry["is_synonym"] = False
            
            # Build taxonomic classification hierarchy
            if params and hasattr(params, 'format') and params.format == "json":
                classification = {}
                for tax_entry in item.get("classification", []):
                    rank = tax_entry.get("rank", "").lower()
                    name = tax_entry.get("name")
                    if rank and name:
                        classification[rank] = name
                taxon_entry["classification"] = classification
            
            # Add Catalogue of Life URL for reference
            if usage.get("id"):
                taxon_entry["col_url"] = f"https://www.catalogueoflife.org/data/taxon/{usage.get('id')}"
            
            processed_results.append(taxon_entry)
        
        return {
            "query_info": {
                "search_term": query.search_term,
                "target_rank": query.rank,
                "api_endpoint": "Catalogue of Life ChecklistBank API",
                "timestamp": raw_data.get("timestamp")
            },
            "result_summary": {
                "total_found": raw_data.get("total", 0),
                "returned_count": len(processed_results),
                "search_completed": True
            },
            "taxonomic_data": processed_results
        }

    async def _generate_summary_with_ai(self, instructor_client, data: dict, query: TaxonModel, total_results: int) -> str:
        """Generate an AI-powered summary of the taxonomic search results using Navigator AI."""
        try:
            results = data.get("taxonomic_data", [])
            search_term = query.search_term or "your query"
            
            if not results:
                return f"No taxonomic entries found for '{search_term}' in the Catalogue of Life database."
            
            # Create a prompt for AI summary generation
            summary_prompt = f"""
            Create a concise, informative summary of these Catalogue of Life taxonomic search results for '{search_term}'. 
            Include key findings, taxonomic classifications, and any notable patterns. Keep it under 150 words and make it accessible to scientists.
            
            Search Results Data:
            Total found: {total_results}
            Primary results: {json.dumps(results[:3], indent=2)}
            """
            
            # Try gpt-4o first, fallback to gpt-3.5-turbo
            try:
                response = await instructor_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a taxonomic expert summarizing Catalogue of Life search results. Be concise, accurate, and scientific."},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
            except Exception:
                response = await instructor_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a taxonomic expert summarizing Catalogue of Life search results. Be concise, accurate, and scientific."},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to basic summary if AI fails
            return self._generate_basic_summary(data, query, total_results)

    def _generate_basic_summary(self, data: dict, query: TaxonModel, total_results: int) -> str:
        """Generate a basic summary without AI as fallback."""
        results = data.get("taxonomic_data", [])
        search_term = query.search_term or "your query"
        
        if not results:
            return f"No taxonomic entries found for '{search_term}' in the Catalogue of Life database."
        
        summary = f"Found {total_results} taxonomic records for '{search_term}' in Catalogue of Life.\n\n"
        
        # Explain the primary result
        primary_result = results[0]
        scientific_name = primary_result.get("scientific_name")
        rank = primary_result.get("taxonomic_rank")
        
        summary += f"Primary result: {scientific_name} (rank: {rank})\n"
        
        # Explain nomenclatural status
        if primary_result.get("is_synonym"):
            accepted_name = primary_result.get("accepted_name")
            summary += f"This name is currently considered a synonym of {accepted_name}.\n"
        else:
            summary += "This is an accepted taxonomic name.\n"
        
        # Additional context for multiple results
        if len(results) > 1:
            summary += f"\nThe search returned {len(results)} related taxonomic entries, which may include different taxonomic ranks, synonyms, or homonyms.\n"
        
        summary += "\nThe generated artifact contains detailed taxonomic data including full classifications, nomenclatural information, and direct links to Catalogue of Life entries."
        
        return summary


if __name__ == "__main__":
    from ichatbio.server import run_agent_server
    
    # Railway port detection
    port = int(os.environ.get("PORT", 3000))
    
    agent = CatalogueOfLifeAgent()
    print("Starting Catalogue of Life Agent with Navigator AI...")
    print(f"Using AI endpoint: {os.getenv('OPENAI_BASE_URL', 'default OpenAI')}")
    print(f"Agent card: http://localhost:{port}/.well-known/agent.json")
    print("Press Ctrl+C to stop")
    
    # Accept external connections
    run_agent_server(agent, host="0.0.0.0", port=port)