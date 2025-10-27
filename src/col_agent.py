import os
import json
import traceback
from typing import Optional, List, Union
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
        description="Search for species by scientific name only. Common names are not supported. This agent requires scientific names (e.g., 'Ursidae' for bears, 'Panthera leo' for lions).",
        examples=["Panthera leo", "Ursidae", "Homo sapiens", "Quercus"]
    )

class TaxonDetailsParameters(BaseModel):
    """Get detailed information for a specific taxon using its Catalogue of Life ID"""
    taxon_id: str = Field(
        description="Catalogue of Life taxon ID (alphanumeric code from search results like '4CGXP', '5K5L5'). This ID must be from a COL search result.",
        examples=["4CGXP", "5K5L5", "9BBLS", "3T", "4CGXS"]
    )

class GetSynonymsParameters(BaseModel):
    """Parameters for getting synonyms of a taxon"""
    query: str = Field(
        description="Get all synonyms for a taxon using its scientific name or taxon ID. Common names are not supported.",
        examples=["Panthera leo", "4CGXP", "Homo sapiens", "4CGXS"]
    )

# STEP 3 & 4: Data model with validation for GPT to extract search terms
class CoLQueryParams(BaseModel):
    """Validated parameters for COL API queries"""
    search_term: str = Field(
        ..., 
        description="Scientific name to search for",
        min_length=2,
        max_length=100,
        examples=["Panthera leo", "Ursidae", "Homo sapiens"]
    )
    limit: Optional[int] = Field(
        1000, 
        description="Number of results (max 1000)",
        ge=1,
        le=1000
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
                    description="Search for species or taxonomic information for scientific names",
                    parameters=SearchParameters
                ),
                AgentEntrypoint(
                    id="get_taxon_details",
                    description="Retrieve complete taxonomic details for a specific taxon using its Catalogue of Life ID. Use this when user provides a taxon ID from search results.",
                    parameters=TaxonDetailsParameters
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description="Common names are not supported, use scientific names for a specific taxon using either its scientific name or taxon ID",
                    parameters=GetSynonymsParameters
                )
            ]
        )
        
        print(f"DEBUG: Agent card created - Name: {card.name}")
        print(f"DEBUG: Agent URL: {card.url}")
        print(f"DEBUG: Entrypoints: {[ep.id for ep in card.entrypoints]}")
        
        return card

    async def _handle_search(self, context: ResponseContext, request: str, params: SearchParameters):
        """Handle the search entrypoint"""
        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            process: IChatBioAgentProcess
            
            try:
                # Use the query directly as scientific name - no GPT conversion
                search_term = params.query
                
                # LOG 1: Search initiated
                await process.log(f"Search initiated for: '{search_term}'")
                print("DEBUG: Starting search process...")
                
                # ============================================================
                # STEP 6: Build COL API request with validated params
                # ============================================================
                col_params = {
                    "q": search_term,
                    "limit": 20,
                    "offset": 0,
                    "facet": "false",
                    "content": "SCIENTIFIC_NAME",
                    "fuzzy": "false",
                    "type": "EXACT"
                }
                
                base_url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
                url_with_params = f"{base_url}?{urlencode(col_params)}"
                
                # LOG 2: API call details
                await process.log(
                    "Calling Catalogue of Life API",
                    data={
                        "url": base_url,
                        "parameters": col_params
                    }
                )
                print(f"DEBUG: Making API call to: {url_with_params}")
                
                # ============================================================
                # STEP 7: Make API call to COL
                # ============================================================
                try:
                    response = requests.get(url_with_params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    print(f"DEBUG: API response status: {response.status_code}")
                    print(f"DEBUG: Total results: {data.get('total', 0)}")
                    
                except requests.exceptions.RequestException as api_error:
                    print(f"DEBUG: API request failed: {api_error}")
                    await process.log(f"API request failed: {str(api_error)}")
                    await context.reply("Error: Could not connect to Catalogue of Life API")
                    return
                except json.JSONDecodeError:
                    print(f"DEBUG: JSON decode error")
                    await process.log("Failed to parse API response")
                    await context.reply("Error: Invalid response from Catalogue of Life API")
                    return
                
                # LOG 3: API response received
                await process.log(
                    f"API response received: {data.get('total', 0)} results",
                    data={
                        "total_results": data.get('total', 0),
                        "returned_results": len(data.get('result', []))
                    }
                )
                
                # ============================================================
                # STEP 8: Process and format results WITH STREAMING
                # ============================================================
                results = data.get('result', [])
                
                if not results:
                    await process.log("No results found")
                    await context.reply(f"No results found for '{search_term}'. Please ensure you're using a valid scientific name (not a common name).")
                    return
                
                # Process results with STREAMING
                print("DEBUG: Processing and streaming search results...")
                
                # Stream the header
                await context.stream(f"Found {len(results)} result(s) for '{search_term}':\n\n")
                
                formatted_results = []
                for i, item in enumerate(results[:10], 1):  # Limit to first 10 for display
                    try:
                        # Extract from 'usage' object
                        usage = item.get('usage', {})
                        name = usage.get('name', {})
                        
                        scientific_name = name.get('scientificName', 'Unknown')
                        authorship = name.get('authorship', '')
                        rank = name.get('rank', 'Unknown')
                        status = usage.get('status', 'Unknown')
                        
                        # Get classification path
                        classification_obj = usage.get('classification', {})
                        classification_items = []
                        
                        # Build classification hierarchy
                        for rank_name in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                            if rank_name in classification_obj:
                                classification_items.append({
                                    'rank': rank_name.capitalize(),
                                    'name': classification_obj[rank_name]
                                })
                        
                        # Get parent information
                        parent_name = classification_obj.get(
                            'genus' if rank == 'SPECIES' else 'family', 
                            'Unknown'
                        )
                        
                        # Extract ID from the item (not usage)
                        taxon_id = item.get('id', usage.get('id', 'Unknown'))
                        
                        # Format and STREAM each result
                        result_text = f"**{i}. {scientific_name}**"
                        if authorship:
                            result_text += f" {authorship}"
                        result_text += f"\n   - ID: `{taxon_id}`\n"
                        result_text += f"   - Rank: {rank}\n"
                        result_text += f"   - Status: {status}\n"
                        result_text += f"   - Parent: {parent_name}\n"
                        
                        if classification_items:
                            classification_str = " > ".join([c['name'] for c in classification_items])
                            result_text += f"   - Classification: {classification_str}\n"
                        
                        result_text += "\n"
                        
                        # STREAM this result immediately
                        await context.stream(result_text)
                        
                        # Store for artifact
                        formatted_results.append({
                            "id": str(taxon_id),
                            "scientific_name": scientific_name,
                            "authorship": authorship,
                            "rank": rank,
                            "status": status,
                            "classification": classification_items,
                            "parent": parent_name
                        })
                        
                    except Exception as item_error:
                        print(f"DEBUG: Error processing result item: {item_error}")
                        continue
                
                # Stream the footer
                if len(results) > 10:
                    await context.stream(f"\nShowing first 10 of {len(results)} results. ")
                    await context.stream("See artifact for complete data.\n")
                else:
                    await context.stream("\nSee artifact for complete data.\n")
                
                # COMPLETE THE STREAM
                await context.complete()
                
                # LOG 4: Results processed
                result_summary = [
                    {
                        "scientific_name": r["scientific_name"],
                        "rank": r["rank"],
                        "id": r["id"]
                    } 
                    for r in formatted_results[:5]  # Log first 5
                ]
                
                await process.log(
                    f"Processed {len(formatted_results)} results",
                    data={"sample_results": result_summary}
                )
                
                # ============================================================
                # STEP 9: Create artifact with full data
                # ============================================================
                print("DEBUG: Creating artifact...")
                
                artifact_data = {
                    "query": search_term,
                    "total_results": data.get('total', 0),
                    "returned_results": len(results),
                    "results": formatted_results,
                    "raw_response": data
                }
                
                try:
                    col_url = f"https://www.catalogueoflife.org/data/search?q={search_term}&type=EXACT"
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Search results for '{search_term}' - {len(formatted_results)} species",
                        content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                        uris=[col_url],
                        metadata={
                            "data_source": "Catalogue of Life",
                            "query": search_term,
                            "result_count": len(formatted_results)
                        }
                    )
                    
                    print("DEBUG: Artifact created successfully")
                    
                except Exception as artifact_error:
                    print(f"DEBUG: Failed to create artifact: {artifact_error}")
                    await process.log(f"Failed to create artifact: {str(artifact_error)}")
                
                await process.log("Response sent to user")
                print("DEBUG: Response sent successfully!")
                
            except Exception as e:
                error_msg = f"Unexpected error during search: {str(e)}"
                print(f"DEBUG: MAJOR ERROR: {error_msg}")
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                
                await process.log(f"Error occurred: {error_msg}")
                await context.reply(
                    f"Sorry, an error occurred during the search.\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"Please try again with a valid scientific name."
                )

    async def _handle_taxon_details(self, context: ResponseContext, request: str, params: TaxonDetailsParameters):
        """Handle getting detailed information for a specific taxon"""
        async with context.begin_process(summary="Fetching taxon details") as process:
            process: IChatBioAgentProcess
            
            try:
                taxon_id = params.taxon_id
                
                # LOG 1: Request initiated
                await process.log(f"Fetching details for taxon ID: {taxon_id}")
                print(f"DEBUG: Getting details for taxon ID: {taxon_id}")
                
                # ============================================================
                # STEP 2: Build COL API URL for taxon details
                # ============================================================
                col_url = f"https://api.checklistbank.org/dataset/3LR/taxon/{taxon_id}"
                
                # LOG 2: API call
                await process.log(
                    "Calling Catalogue of Life API",
                    data={"url": col_url}
                )
                print(f"DEBUG: Making API call to: {col_url}")
                
                # ============================================================
                # STEP 3: Make API call
                # ============================================================
                try:
                    response = requests.get(col_url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    print(f"DEBUG: API response status: {response.status_code}")
                    
                except requests.exceptions.HTTPError as http_error:
                    if response.status_code == 404:
                        print(f"DEBUG: Taxon not found (404)")
                        await process.log("Taxon not found")
                        await context.reply(f"No taxon found with ID '{taxon_id}'. Please ensure this is a valid Catalogue of Life taxon ID from a search result.")
                        return
                    else:
                        raise http_error
                        
                except requests.exceptions.RequestException as api_error:
                    print(f"DEBUG: API request failed: {api_error}")
                    await process.log(f"API request failed: {str(api_error)}")
                    await context.reply("Error: Could not connect to Catalogue of Life API")
                    return
                except json.JSONDecodeError:
                    print(f"DEBUG: JSON decode error")
                    await process.log("Failed to parse API response")
                    await context.reply("Error: Invalid response from Catalogue of Life API")
                    return
                
                # ============================================================
                # STEP 4: Extract and format taxon information WITH STREAMING
                # ============================================================
                print("DEBUG: Extracting and streaming taxon details...")
                
                # Extract basic information
                name = data.get('name', {})
                scientific_name = name.get('scientificName', 'Unknown')
                authorship = name.get('authorship', '')
                rank = name.get('rank', 'Unknown')
                status = data.get('status', 'Unknown')
                
                # Get classification
                classification = data.get('classification', [])
                
                # Get synonyms count
                synonyms = data.get('synonyms', [])
                synonym_count = len(synonyms) if isinstance(synonyms, list) else 0
                
                # Get additional info
                extinct = data.get('extinct', False)
                temporal_range_start = data.get('temporalRangeStart')
                temporal_range_end = data.get('temporalRangeEnd')
                
                # Stream the response progressively
                await context.stream(f"# Taxon Details: {scientific_name}\n\n")
                
                if authorship:
                    await context.stream(f"**Full Name:** {scientific_name} {authorship}\n\n")
                
                # Basic Information section
                basic_info = "## Basic Information\n\n"
                basic_info += f"- **ID:** `{taxon_id}`\n"
                basic_info += f"- **Rank:** {rank}\n"
                basic_info += f"- **Status:** {status}\n"
                if extinct:
                    basic_info += f"- **Extinct:** Yes\n"
                if temporal_range_start:
                    basic_info += f"- **Temporal Range:** {temporal_range_start}"
                    if temporal_range_end:
                        basic_info += f" - {temporal_range_end}"
                    basic_info += "\n"
                basic_info += "\n"
                
                await context.stream(basic_info)
                
                # Classification section
                if classification:
                    class_text = "## Classification\n\n"
                    for item in classification:
                        class_rank = item.get('rank', 'Unknown')
                        class_name = item.get('name', 'Unknown')
                        class_text += f"- **{class_rank.capitalize()}:** {class_name}\n"
                    class_text += "\n"
                    
                    await context.stream(class_text)
                
                # Links section
                links_text = "## Links\n\n"
                links_text += f"- [View on Catalogue of Life](https://www.catalogueoflife.org/data/taxon/{taxon_id})\n"
                links_text += f"- [View on ChecklistBank](https://www.checklistbank.org/dataset/3LR/taxon/{taxon_id})\n"
                
                if synonym_count > 0:
                    links_text += f"\n## Synonyms\n\nThis taxon has {synonym_count} synonym(s). Use the get_synonyms entrypoint to view them.\n"
                
                links_text += "\nSee artifact for complete data.\n"
                
                await context.stream(links_text)
                
                # Complete the stream
                await context.complete()
                
                # LOG 3: Details retrieved
                await process.log(
                    "Taxon details retrieved",
                    data={
                        "scientific_name": scientific_name,
                        "rank": rank,
                        "status": status
                    }
                )
                
                # ============================================================
                # STEP 5: Create artifact with complete data
                # ============================================================
                print("DEBUG: Creating artifact...")
                
                artifact_data = {
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name,
                    "authorship": authorship,
                    "rank": rank,
                    "status": status,
                    "extinct": extinct,
                    "classification": classification,
                    "temporal_range": {
                        "start": temporal_range_start,
                        "end": temporal_range_end
                    } if temporal_range_start else None,
                    "synonym_count": synonym_count,
                    "raw_data": data
                }
                
                try:
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Complete details for {scientific_name} (ID: {taxon_id})",
                        content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                        uris=[col_url],
                        metadata={
                            "data_source": "Catalogue of Life",
                            "taxon_id": taxon_id,
                            "scientific_name": scientific_name,
                            "rank": rank
                        }
                    )
                    print("DEBUG: Artifact created successfully")
                    
                except Exception as artifact_error:
                    print(f"DEBUG: Failed to create artifact: {artifact_error}")
                    await process.log(f"Failed to create artifact: {str(artifact_error)}")
                
                await process.log("Response sent to user")
                print("DEBUG: Response sent successfully!")
                
            except Exception as e:
                error_msg = f"Unexpected error fetching taxon details: {str(e)}"
                print(f"DEBUG: MAJOR ERROR: {error_msg}")
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                
                await process.log(f"Error occurred: {error_msg}")
                await context.reply(
                    f"Sorry, an error occurred while fetching taxon details.\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"Please ensure you're using a valid taxon ID from a search result."
                )

    async def _handle_get_synonyms(self, context: ResponseContext, request: str, params: GetSynonymsParameters):
        """Handle getting synonyms for a taxon"""
        async with context.begin_process(summary="Fetching synonyms") as process:
            process: IChatBioAgentProcess
            
            try:
                query = params.query
                taxon_id = None
                scientific_name = None
                
                # Determine if query is a taxon ID or scientific name
                if query.isalnum() and len(query) <= 10:  # Likely a taxon ID
                    taxon_id = query
                    await process.log(f"Looking up synonyms for taxon ID: {taxon_id}")
                else:
                    scientific_name = query
                    await process.log(f"Looking up synonyms for: {scientific_name}")
                    
                    # First, search for the taxon to get its ID
                    search_params = {
                        "q": scientific_name,
                        "limit": 1,
                        "content": "SCIENTIFIC_NAME",
                        "type": "EXACT"
                    }
                    
                    search_url = f"https://api.checklistbank.org/dataset/3LR/nameusage/search?{urlencode(search_params)}"
                    
                    try:
                        response = requests.get(search_url, timeout=30)
                        response.raise_for_status()
                        search_data = response.json()
                        
                        if search_data.get('result'):
                            taxon_id = search_data['result'][0].get('id')
                            await process.log(f"Found taxon ID: {taxon_id}")
                        else:
                            await process.log("No matching taxon found")
                            await context.reply(f"No taxon found for '{scientific_name}'. Please check the scientific name.")
                            return
                            
                    except Exception as search_error:
                        print(f"DEBUG: Search failed: {search_error}")
                        await process.log(f"Search failed: {str(search_error)}")
                        await context.reply("Error searching for the taxon. Please try with a taxon ID instead.")
                        return
                
                # Now get synonyms using the taxon details endpoint
                col_url = f"https://api.checklistbank.org/dataset/3LR/taxon/{taxon_id}"
                
                await process.log("Fetching taxon data from API")
                
                try:
                    response = requests.get(col_url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract synonyms from the response
                    synonyms_data = data.get('synonyms', [])
                    
                    if not scientific_name:
                        name_obj = data.get('name', {})
                        scientific_name = name_obj.get('scientificName', f'taxon ID {taxon_id}')
                    
                except Exception as api_error:
                    print(f"DEBUG: API request failed: {api_error}")
                    await process.log(f"API request failed: {str(api_error)}")
                    await context.reply("Error: Could not connect to Catalogue of Life API")
                    return
                
                # Check if there are any synonyms
                if not synonyms_data or len(synonyms_data) == 0:
                    await process.log("No synonyms found")
                    display_name = scientific_name if scientific_name else f"taxon ID '{taxon_id}'"
                    await context.reply(f"No synonyms found for {display_name}. This may be the currently accepted name with no alternative names.")
                    return
                
                # Process and stream synonyms
                print("DEBUG: Processing and streaming synonyms...")
                
                # Stream header
                display_name = scientific_name if scientific_name else f"taxon ID {taxon_id}"
                await context.stream(f"**Found {len(synonyms_data)} synonym(s) for {display_name}:**\n\n")
                
                synonyms_list = []
                
                for i, item in enumerate(synonyms_data, 1):
                    try:
                        name_obj = item.get('name', {})
                        syn_scientific_name = name_obj.get('scientificName', 'Unknown')
                        authorship = name_obj.get('authorship', '')
                        rank = name_obj.get('rank', 'Unknown')
                        status = item.get('status', 'Unknown')
                        
                        # Stream each synonym
                        full_name = f"{syn_scientific_name} {authorship}".strip()
                        synonym_text = f"{i}. *{full_name}*\n"
                        synonym_text += f"   - Rank: {rank}\n"
                        synonym_text += f"   - Status: {status}\n\n"
                        
                        await context.stream(synonym_text)
                        
                        # Store for artifact
                        synonym_info = {
                            "scientificName": syn_scientific_name,
                            "authorship": authorship,
                            "rank": rank,
                            "status": status
                        }
                        synonyms_list.append(synonym_info)
                        
                    except Exception as item_error:
                        print(f"DEBUG: Error processing synonym: {item_error}")
                        continue
                
                # Stream footer
                await context.stream("\nSee artifact for complete synonym data.")
                
                # Complete the stream
                await context.complete()
                
                # LOG: Synonyms retrieved
                synonyms_summary = [
                    {
                        "scientific_name": s["scientificName"],
                        "status": s["status"]
                    }
                    for s in synonyms_list[:10]  # Show first 10 in log
                ]
                
                await process.log(
                    f"Found {len(synonyms_list)} synonyms",
                    data={"synonyms": synonyms_summary}
                )
                
                # Create artifact
                print("DEBUG: Creating artifact...")
                
                artifact_data = {
                    "query": query,
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name,
                    "synonym_count": len(synonyms_list),
                    "synonyms": synonyms_list,
                    "raw_response": data
                }
                
                try:
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Synonyms for {scientific_name or taxon_id} - {len(synonyms_list)} total",
                        content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                        uris=[col_url],
                        metadata={
                            "data_source": "Catalogue of Life",
                            "taxon_id": taxon_id,
                            "scientific_name": scientific_name or "Unknown",
                            "synonym_count": len(synonyms_list)
                        }
                    )
                    print("DEBUG: Artifact created successfully")
                    
                except Exception as artifact_error:
                    print(f"DEBUG: Failed to create artifact: {artifact_error}")
                    await process.log(f"Failed to create artifact: {str(artifact_error)}")
                
                await process.log("Response sent to user")
                print("DEBUG: Response sent successfully!")
                
            except Exception as e:
                error_msg = f"Unexpected error fetching synonyms: {str(e)}"
                print(f"DEBUG: MAJOR ERROR: {error_msg}")
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                
                await process.log(f"Error occurred: {error_msg}")
                await context.reply(
                    f"Sorry, an error occurred while fetching synonyms.\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"Please try again."
                )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Union[SearchParameters, TaxonDetailsParameters, GetSynonymsParameters]):
        """
        Main agent entry point - routes to appropriate handler based on entrypoint
        """
        print(f"\nDEBUG: Agent.run() called!")
        print(f"DEBUG: Entrypoint: {entrypoint}")
        print(f"DEBUG: Request: {request}")
        print(f"DEBUG: Params: {params}")
        
        # Validate entrypoint
        if entrypoint not in ["search", "get_taxon_details", "get_synonyms"]:
            error_msg = f"Unknown entrypoint: {entrypoint}. Expected 'search', 'get_taxon_details', or 'get_synonyms'"
            print(f"DEBUG: {error_msg}")
            await context.reply(error_msg)
            return

        # Route to appropriate handler
        if entrypoint == "search":
            await self._handle_search(context, request, params)
        elif entrypoint == "get_taxon_details":
            await self._handle_taxon_details(context, request, params)
        elif entrypoint == "get_synonyms":
            await self._handle_get_synonyms(context, request, params)


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