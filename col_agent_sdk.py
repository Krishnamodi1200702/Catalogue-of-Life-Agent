import os
import json
from typing import Optional, Literal, override
from urllib.parse import urlencode

import dotenv
import instructor
import pydantic
import requests
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

dotenv.load_dotenv()

TaxonomicResponseFormat = Literal["json", "summary"]

class GetTaxonomicDataParameters(BaseModel):
    format: TaxonomicResponseFormat = Field(default="json", description="Response format: 'json' for detailed data or 'summary' for brief overview")

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
        if v is not None and v <= 0:
            raise ValueError("limit must be positive")
        return v

    def to_url(self, format: TaxonomicResponseFormat = "json"):
        url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
        params = {}

        if self.search_term:
            params["q"] = self.search_term

        if self.rank:
            params["rank"] = self.rank

        if self.limit:
            params["limit"] = self.limit
        
        # Add format parameter if needed (though CoL API doesn't use this, keeping for consistency)
        if format == "summary":
            params["format"] = "summary"

        if params:
            url += "?" + urlencode(params)

        return url

class CatalogueOfLifeAgent(IChatBioAgent):
    
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Catalogue of Life Taxonomic Search",
            description="Retrieves comprehensive taxonomic and species data from the Catalogue of Life database using natural language queries.",
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="get_taxonomic_data",
                    description="Search and retrieve taxonomic information for species, genera, families, or other taxonomic groups",
                    parameters=GetTaxonomicDataParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[GetTaxonomicDataParameters]):
        # Validate entrypoint
        if entrypoint != "get_taxonomic_data":
            await context.reply(f"Unknown entrypoint: {entrypoint}")
            return

        # Start a process to log the agent's taxonomic search actions
        async with context.begin_process(summary="Searching Catalogue of Life database") as process:
            process: IChatBioAgentProcess
            
            try:
                # Log the initial step
                await process.log("Initializing taxonomic search process")
                await process.log(f"User query: '{request}'")
                
                # Set up OpenAI client for parameter extraction
                openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                instructor_client = instructor.patch(openai_client)

                # Extract structured query parameters from user request
                await process.log("Converting natural language query to structured API parameters")
                
                taxon: TaxonModel = await instructor_client.chat.completions.create(
                    model="gpt-4o",
                    response_model=TaxonModel,
                    messages=[
                        {"role": "system",
                         "content": "Extract taxonomic search parameters from user requests. Focus on scientific names, common names, taxonomic ranks, and any limits mentioned. If no specific limit is mentioned, use 20 as default."},
                        {"role": "user", "content": request}
                    ],
                    max_retries=3,
                    temperature=0
                )

                # Build the API URL
                format_param = params.format if params else "json"
                api_url = taxon.to_url(format_param)
                
                extracted_params = taxon.model_dump(exclude_none=True)
                await process.log(f"Extracted search parameters: {extracted_params}")
                
                # Log the API call
                await process.log(f"Querying Catalogue of Life ChecklistBank API")
                await process.log(f"API URL: {api_url}")
                
                # Execute the API request
                response = requests.get(api_url, timeout=30)
                
                if response.status_code != 200:
                    await process.log(f"API request failed with status {response.status_code}")
                    await process.log(f"Response content: {response.text[:200]}...")
                    await context.reply(f"Error accessing Catalogue of Life database: HTTP {response.status_code}")
                    return

                # Parse and log results
                raw_data = response.json()
                total_results = raw_data.get("total", 0)
                returned_results = len(raw_data.get("result", []))
                
                await process.log(f"API request successful - found {total_results} total matches")
                await process.log(f"Returning {returned_results} results in this response")
                
                if total_results == 0:
                    await process.log("No taxonomic data found for the specified query")
                    await context.reply(f"No taxonomic data found for '{taxon.search_term or request}' in the Catalogue of Life database.")
                    return

                # Process and structure the data
                await process.log("Processing and structuring taxonomic data")
                
                structured_data = self._process_taxonomic_data(raw_data, taxon, params)
                
                # Create artifact with the taxonomic data
                await process.log("Creating taxonomic data artifact")
                
                artifact_content = json.dumps(structured_data, indent=2).encode('utf-8')
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Taxonomic data for \"{taxon.search_term or 'search query'}\" from Catalogue of Life",
                    content=artifact_content,
                    uris=[api_url],
                    metadata={
                        "api_query_url": api_url,
                        "total_taxa_found": total_results,
                        "returned_count": returned_results,
                        "search_term": taxon.search_term,
                        "taxonomic_rank": taxon.rank,
                        "source": "Catalogue of Life ChecklistBank API",
                        "dataset": "3LR (Catalogue of Life Checklist)"
                    }
                )
                
                # Generate summary for the user
                await process.log("Generating human-readable summary")
                
                summary_text = self._generate_summary(structured_data, taxon, total_results)
                
                await process.log("Taxonomic search completed successfully")

            except InstructorRetryException as e:
                await process.log(f"Failed to process query with GPT after retries: {str(e)}")
                await context.reply("Sorry, I couldn't process your taxonomic query. Please try rephrasing your request with more specific taxonomic terms.")
                return
            except requests.exceptions.Timeout:
                await process.log("API request timed out")
                await context.reply("The request to Catalogue of Life timed out. Please try again.")
                return
            except requests.exceptions.RequestException as e:
                await process.log(f"Network error during API request: {str(e)}")
                await context.reply("Network error while accessing Catalogue of Life. Please try again later.")
                return
            except Exception as e:
                await process.log(f"Unexpected error during taxonomic search: {str(e)}")
                await context.reply(f"An unexpected error occurred while retrieving taxonomic data: {str(e)}")
                return

        # Reply to the user with the summary
        await context.reply(summary_text)

    def _process_taxonomic_data(self, raw_data: dict, query: TaxonModel, params: Optional[GetTaxonomicDataParameters]) -> dict:
        """
        Transform raw Catalogue of Life API response into structured taxonomic data.
        """
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
                "col_id": usage.get("id"),
                "dataset_key": usage.get("datasetKey")
            }
            
            # Add accepted name if this is a synonym
            if usage.get("accepted"):
                accepted_info = usage.get("accepted", {}).get("name", {})
                taxon_entry["accepted_name"] = accepted_info.get("scientificName")
                taxon_entry["accepted_id"] = usage.get("accepted", {}).get("id")
                taxon_entry["is_synonym"] = True
            else:
                taxon_entry["is_synonym"] = False
            
            # Build taxonomic classification hierarchy
            classification = {}
            classification_list = []
            for tax_entry in item.get("classification", []):
                rank = tax_entry.get("rank", "").lower()
                name = tax_entry.get("name")
                if rank and name:
                    classification[rank] = name
                    classification_list.append(f"{rank.title()}: {name}")
            
            taxon_entry["classification"] = classification
            taxon_entry["classification_path"] = classification_list
            
            # Add Catalogue of Life URL for reference
            if usage.get("id"):
                taxon_entry["col_url"] = f"https://www.catalogueoflife.org/data/taxon/{usage.get('id')}"
            
            # Add any additional useful information
            if name_info.get("code"):
                taxon_entry["nomenclatural_code"] = name_info.get("code")
            
            processed_results.append(taxon_entry)
        
        return {
            "query_info": {
                "search_term": query.search_term,
                "target_rank": query.rank,
                "result_limit": query.limit,
                "api_endpoint": "Catalogue of Life ChecklistBank API",
                "dataset": "3LR",
                "query_url": query.to_url(params.format if params else "json")
            },
            "result_summary": {
                "total_found": raw_data.get("total", 0),
                "returned_count": len(processed_results),
                "has_more_results": raw_data.get("total", 0) > len(processed_results),
                "search_completed": True
            },
            "taxonomic_data": processed_results
        }

    def _generate_summary(self, data: dict, query: TaxonModel, total_results: int) -> str:
        """
        Generate a human-readable summary of the taxonomic search results.
        """
        results = data.get("taxonomic_data", [])
        search_term = query.search_term or "your query"
        
        if not results:
            return f"No taxonomic entries found for '{search_term}' in the Catalogue of Life database."
        
        summary = f"## Taxonomic Search Results\n\n"
        summary += f"Found **{total_results:,} taxonomic records** for '*{search_term}*' in Catalogue of Life.\n\n"
        
        # Show details about the primary result
        primary_result = results[0]
        scientific_name = primary_result.get("scientific_name")
        rank = primary_result.get("taxonomic_rank")
        authorship = primary_result.get("authorship")
        
        summary += f"### Primary Result\n"
        summary += f"**{scientific_name}**"
        if authorship:
            summary += f" {authorship}"
        summary += f" ({rank})\n\n"
        
        # Explain nomenclatural status
        if primary_result.get("is_synonym"):
            accepted_name = primary_result.get("accepted_name")
            summary += f"⚠️ This name is currently considered a **synonym** of *{accepted_name}*.\n\n"
        else:
            summary += f"✅ This is an **accepted taxonomic name**.\n\n"
        
        # Show classification if available
        if primary_result.get("classification"):
            classification = primary_result["classification"]
            summary += f"### Taxonomic Classification\n"
            for rank_name in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                if rank_name in classification:
                    summary += f"- **{rank_name.title()}:** {classification[rank_name]}\n"
            summary += "\n"
        
        # Show additional results summary
        if len(results) > 1:
            summary += f"### Additional Results\n"
            summary += f"The search returned **{len(results)} records** in this response"
            if total_results > len(results):
                summary += f" (showing first {len(results)} of {total_results:,} total matches)"
            summary += ".\n\n"
            
            # Show a few more results briefly
            for i, result in enumerate(results[1:min(4, len(results))], 2):
                name = result.get("scientific_name")
                rank = result.get("taxonomic_rank")
                status = " (synonym)" if result.get("is_synonym") else ""
                summary += f"{i}. *{name}* ({rank}){status}\n"
            
            if len(results) > 4:
                summary += f"... and {len(results) - 4} more results.\n"
            summary += "\n"
        
        # Add source and artifact information
        summary += f"### Data Source\n"
        summary += f"**Source:** Catalogue of Life ChecklistBank API (Dataset 3LR)\n"
        summary += f"**Total Matches:** {total_results:,} records\n\n"
        summary += f"📄 **Complete data** with full classifications, nomenclatural details, and direct links to Catalogue of Life entries is available in the generated JSON artifact above."
        
        return summary


if __name__ == "__main__":
    from ichatbio.server import run_agent_server
    
    agent = CatalogueOfLifeAgent()
    print("Starting Catalogue of Life Agent (Enhanced SDK Version)...")
    print("Agent card available at: http://localhost:3000/.well-known/agent.json")
    print("Press Ctrl+C to stop the server")
    
    run_agent_server(agent, host="0.0.0.0", port=9999)