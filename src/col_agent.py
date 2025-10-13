import os
import json
import traceback
from typing import Optional, List
from typing_extensions import override
from urllib.parse import urlencode

import dotenv
import instructor
import requests
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

# Load environment variables
dotenv.load_dotenv()

print("DEBUG: Loading environment variables...")
print(f"DEBUG: OPENAI_API_KEY loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
print(f"DEBUG: OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'Not set')}")

# STEP 3: Create Pydantic data models for the API
class SearchParameters(BaseModel):
    """User-facing parameters for the search entrypoint"""
    query: str = Field(
        description="What to search for in Catalogue of Life",
        examples=["tiger", "Panthera leo", "oak tree"]
    )

# STEP 3 & 4: Data model with validation for GPT to extract search terms
class CoLQueryParams(BaseModel):
    """Validated parameters for COL API queries"""
    search_term: str = Field(
        ..., 
        description="Scientific or common name to search for",
        min_length=2,
        max_length=100,
        examples=["Panthera leo", "tiger", "Homo sapiens"]
    )
    limit: Optional[int] = Field(
        5, 
        description="Number of results (max 20)",
        ge=1,
        le=20
    )

# STEP 1: Explore their data - Response models for API data
class TaxonClassification(BaseModel):
    """Model for taxonomic classification entries"""
    rank: str
    name: str

class CoLResult(BaseModel):
    """Model for individual search results from COL API"""
    scientificName: str
    rank: str
    status: str
    classification: Optional[List[TaxonClassification]] = []

class CatalogueOfLifeAgent(IChatBioAgent):
    
    def __init__(self):
        print("DEBUG: Initializing CatalogueOfLifeAgent...")
        super().__init__()
        
        # Initialize OpenAI client for STEP 5: LLM parameter generation
        try:
            self.openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            self.instructor_client = instructor.patch(self.openai_client)
            print("DEBUG: OpenAI client initialized successfully")
        except Exception as e:
            print(f"DEBUG: Failed to initialize OpenAI client: {e}")
            raise
    
    @override
    def get_agent_card(self) -> AgentCard:
        print("DEBUG: Creating agent card...")
        
        card = AgentCard(
            name="Catalogue of Life Agent",
            description="Search for species and taxonomic information from the Catalogue of Life database",
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="search",
                    description="Search for species or taxonomic information",
                    parameters=SearchParameters
                )
            ]
        )
        
        print(f"DEBUG: Agent card created - Name: {card.name}")
        print(f"DEBUG: Agent URL: {card.url}")
        print(f"DEBUG: Entrypoints: {[ep.id for ep in card.entrypoints]}")
        
        return card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: SearchParameters):
        """
        Main agent workflow implementing all 9 steps:
        1. Explore data (via response models)
        2. Explore API (COL REST API)
        3. Create Pydantic models (CoLQueryParams, CoLResult)
        4. Add validation (min_length, ge, le constraints)
        5. LLM generates parameters (instructor + GPT)
        6. Transform to API URLs (build query URLs)
        7. Execute API queries (requests.get)
        8. Extract/derive summary (parse results, build taxonomy)
        9. Generate verbal summary (detailed reply with suggestions)
        """
        print(f"\nDEBUG: Agent.run() called!")
        print(f"DEBUG: Entrypoint: {entrypoint}")
        print(f"DEBUG: Request: {request}")
        print(f"DEBUG: Params: {params}")
        
        # Validate entrypoint
        if entrypoint != "search":
            error_msg = f"Unknown entrypoint: {entrypoint}. Expected 'search'"
            print(f"DEBUG: {error_msg}")
            await context.reply(error_msg)
            return

        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            process: IChatBioAgentProcess
            
            try:
                # ============================================================
                # STEP 5: Have LLM generate parameters using response model
                # ============================================================
                print("DEBUG: Calling GPT to extract search parameters...")
                
                query_instructions = """
                You are helping search the Catalogue of Life database, which uses scientific (Latin) names.
                
                Your task: Convert the user's query into the BEST search term.
                
                CRITICAL RULES:
                1. If the user gives a common name, YOU MUST convert it to the scientific name:
                   - "lion" → "Panthera leo"
                   - "tiger" → "Panthera tigris" 
                   - "elephant" → "Loxodonta africana" OR "Elephas maximus"
                   - "human" → "Homo sapiens"
                   - "dog" → "Canis lupus familiaris"
                   - "cat" → "Felis catus"
                   - "polar bear" → "Ursus maritimus"
                   - "bald eagle" → "Haliaeetus leucocephalus"
                   
                2. If the user already gives a scientific name, keep it as-is:
                   - "Panthera leo" → "Panthera leo"
                   - "Homo sapiens" → "Homo sapiens"
                   
                3. If the user gives a genus name, keep it:
                   - "Panthera" → "Panthera"
                   - "Homo" → "Homo"
                   
                4. If you don't know the scientific name for a common name, use the common name
                
                Return ONLY the search term, nothing else.
                """
                
                try:
                    query_params: CoLQueryParams = await self.instructor_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": query_instructions},
                            {"role": "user", "content": f"Query: {request}\nParams: {params.query}"}
                        ],
                        response_model=CoLQueryParams,
                        temperature=0,
                        max_retries=2
                    )
                    
                    print(f"DEBUG: GPT extracted - search_term: '{query_params.search_term}', limit: {query_params.limit}")
                    
                except Exception as gpt_error:
                    print(f"DEBUG: GPT extraction failed: {gpt_error}")
                    
                    # Fallback: use the raw query with validation
                    query_params = CoLQueryParams(
                        search_term=params.query or request,
                        limit=5
                    )
                    print(f"DEBUG: Using fallback search term: '{query_params.search_term}'")
                
                # LOG 1: Search initiated
                await process.log(f"Search initiated for: '{query_params.search_term}'")
                print("DEBUG: Starting search process...")
                
                # ============================================================
                # STEP 6: Transform parameters into API query URLs
                # ============================================================
                print("DEBUG: Building COL API URL...")
                
                col_url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
                api_params = {
                    "q": query_params.search_term,
                    "limit": min(query_params.limit or 5, 20)
                    "content": "SCIENTIFIC_NAME"
                }
                
                # Construct full URL for logging
                full_url = f"{col_url}?{urlencode(api_params)}"
                
                print(f"DEBUG: API URL: {col_url}")
                print(f"DEBUG: API Params: {api_params}")
                print(f"DEBUG: Full URL: {full_url}")
                
                # LOG 2: API Query with full URL
                await process.log(
                    "API Query",
                    data={
                        "endpoint": col_url,
                        "parameters": api_params,
                        "full_url": full_url
                    }
                )
                
                # ============================================================
                # STEP 7: Run API query, collect response
                # ============================================================
                print("DEBUG: Executing COL API request...")
                
                try:
                    response = requests.get(col_url, params=api_params, timeout=10)
                    print(f"DEBUG: API Response Status: {response.status_code}")
                    print(f"DEBUG: API Response URL: {response.url}")
                    
                    if response.status_code != 200:
                        error_msg = f"Catalogue of Life API error: HTTP {response.status_code}"
                        print(f"DEBUG: {error_msg}")
                        await process.log(f"API error: {error_msg}")
                        await context.reply(error_msg)
                        return
                    
                    # Parse JSON response
                    try:
                        data = response.json()
                        print(f"DEBUG: Parsed JSON successfully")
                        print(f"DEBUG: Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        
                    except json.JSONDecodeError as json_error:
                        print(f"DEBUG: JSON decode error: {json_error}")
                        await process.log("Failed to parse JSON response")
                        await context.reply("Error: Invalid response from Catalogue of Life API")
                        return
                    
                except requests.RequestException as req_error:
                    print(f"DEBUG: Request error: {req_error}")
                    await process.log(f"Network error: {str(req_error)}")
                    await context.reply("Error: Could not connect to Catalogue of Life API")
                    return
                
                # ============================================================
                # STEP 8: Extract/derive summary material from response
                # ============================================================
                results = data.get("result", [])
                total = data.get("total", 0)
                
                print(f"DEBUG: Found {len(results)} results out of {total} total")
                
                if len(results) == 0:
                    # Provide helpful suggestions when no results
                    await process.log("No results found")
                    no_results_msg = (
                        f"No species found for '{query_params.search_term}' in Catalogue of Life.\n\n"
                        f"**Suggestions:**\n"
                        f"- Try using a scientific name (e.g., 'Panthera tigris' instead of 'tiger')\n"
                        f"- Check spelling\n"
                        f"- Use a broader search term (e.g., genus instead of species)\n"
                        f"- Try common English names for well-known species"
                    )
                    print(f"DEBUG: {no_results_msg}")
                    await context.reply(no_results_msg)
                    return
                
                # Extract and structure data from results
                print("DEBUG: Extracting and formatting results...")
                
                formatted_results = []
                
                for i, item in enumerate(results[:5], 1):
                    try:
                        print(f"DEBUG: Processing result {i}...")
                        
                        # Extract from correct nested structure
                        usage = item.get("usage", {})
                        name_obj = usage.get("name", {})
                        
                        scientific_name = name_obj.get("scientificName", "Unknown")
                        rank = name_obj.get("rank", "Unknown")
                        status = usage.get("status", "Unknown")
                        
                        # Extract classification hierarchy
                        classification = item.get("classification", [])
                        taxonomy = {}
                        
                        desired_ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
                        
                        if classification:
                            for taxon in classification:
                                taxon_rank = taxon.get("rank", "").lower()
                                taxon_name = taxon.get("name", "")
                                if taxon_rank in desired_ranks and taxon_name:
                                    taxonomy[taxon_rank] = taxon_name
                        
                        result_info = {
                            "scientificName": scientific_name,
                            "rank": rank,
                            "status": status,
                            "taxonomy": taxonomy
                        }
                        formatted_results.append(result_info)
                        
                    except Exception as item_error:
                        print(f"DEBUG: Error processing result {i}: {item_error}")
                        continue
                
                # LOG 3: Results summary with top results
                top_results_summary = [
                    {
                        "scientific_name": r["scientificName"],
                        "rank": r["rank"],
                        "status": r["status"]
                    }
                    for r in formatted_results
                ]
                
                await process.log(
                    f"Results: Found {total} matches, showing top {len(formatted_results)}",
                    data={"results": top_results_summary}
                )
                
                # ============================================================
                # STEP 9: Generate verbal summary with actions, outcomes, and suggestions
                # ============================================================
                print("DEBUG: Generating verbal summary...")
                
                # Build concise reply - just the key finding
                # The detailed data is in the artifact
                if len(formatted_results) == 1:
                    main_result = formatted_results[0]
                    result_scientific_name = main_result['scientificName']
                    reply_text = f"Found {result_scientific_name} ({main_result['rank']}, {main_result['status']}). "
                    
                    # Check if result seems unrelated to query (heuristic check)
                    query_lower = query_params.search_term.lower()
                    result_name_lower = result_scientific_name.lower()
                    
                    # If query doesn't appear in result name and they're very different, it might be wrong
                    if (len(query_lower) > 3 and 
                        query_lower not in result_name_lower and 
                        result_name_lower.split()[0].lower() != query_lower):  # Check genus too
                        
                        reply_text += f"\n\nNote: This result may not match your search for '{query_params.search_term}'. "
                        reply_text += "The COL API searches broadly (including author names and references). "
                        reply_text += "For better results, try using the scientific name (e.g., 'Panthera leo' for lion). "
                    
                elif len(formatted_results) > 1:
                    reply_text = f"Found {total} matches for '{query_params.search_term}'. "
                    reply_text += f"Top result: {formatted_results[0]['scientificName']} ({formatted_results[0]['rank']}). "
                else:
                    reply_text = f"Found {total} matches. "
                
                reply_text += "See artifact for complete taxonomic data."
                
                # Create artifact with complete data
                print("DEBUG: Creating artifact...")
                
                artifact_data = {
                    "search_info": {
                        "query": query_params.search_term,
                        "original_request": request,
                        "total_found": total,
                        "showing": len(formatted_results),
                        "api_url": response.url
                    },
                    "results": formatted_results,
                    "raw_response": data
                }
                
                try:
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"COL search results for '{query_params.search_term}' - {len(formatted_results)} of {total} results",
                        content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                        uris=[response.url],
                        metadata={
                            "data_source": "Catalogue of Life",
                            "query": query_params.search_term,
                            "total_found": total,
                            "results_count": len(formatted_results),
                            "api_endpoint": col_url
                        }
                    )
                    print("DEBUG: Artifact created successfully")
                    
                except Exception as artifact_error:
                    print(f"DEBUG: Failed to create artifact: {artifact_error}")
                    await process.log(f"Failed to create artifact: {str(artifact_error)}")
                
                # Send final response
                print("DEBUG: Sending final response to user")
                print(f"DEBUG: Reply text length: {len(reply_text)} characters")
                
                await context.reply(reply_text)
                print("DEBUG: Response sent successfully!")
                
                # Add a confirmation log to verify reply was sent
                await process.log("Response sent to user")
                
            except Exception as e:
                error_msg = f"Unexpected error during search: {str(e)}"
                print(f"DEBUG: MAJOR ERROR: {error_msg}")
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                
                await process.log(f"Error occurred: {error_msg}")
                await context.reply(
                    f"Sorry, an error occurred while searching Catalogue of Life.\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Suggestions:**\n"
                    f"- Try rephrasing your search query\n"
                    f"- Check your internet connection\n"
                    f"- Try again in a few moments"
                )


# Test the agent locally before running the server
def test_agent():
    """Test function to verify agent works before starting server"""
    print("\nTESTING AGENT LOCALLY...")
    
    try:
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        print(f"TEST: Agent card created successfully")
        print(f"TEST: Agent name: {card.name}")
        print(f"TEST: Entrypoints: {[ep.id for ep in card.entrypoints]}")
        return True
    except Exception as e:
        print(f"TEST FAILED: {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    print("Starting Catalogue of Life Agent...")
    
    # Test the agent first
    if not test_agent():
        print("Agent test failed! Fix errors before starting server.")
        exit(1)
    
    print("Agent test passed! Starting server...")
    
    try:
        from ichatbio.server import run_agent_server
        
        agent = CatalogueOfLifeAgent()
        
        print("Server starting on: http://localhost:9999")
        print("Agent card will be at: http://localhost:9999/.well-known/agent.json")
        print("Test the API manually: https://api.checklistbank.org/dataset/3LR/nameusage/search?q=tiger")
        print("Press Ctrl+C to stop")
        
        run_agent_server(agent, host="0.0.0.0", port=9999)
        
    except Exception as e:
        print(f"SERVER ERROR: {e}")
        print(f"TRACEBACK: {traceback.format_exc()}")