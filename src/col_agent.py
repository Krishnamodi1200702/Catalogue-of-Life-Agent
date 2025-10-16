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
        description="What to search for in Catalogue of Life",
        examples=["tiger", "Panthera leo", "oak tree"]
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
        description="Scientific name (e.g., 'Panthera leo') OR taxon ID (e.g., '4CGXP') to get synonyms for",
        examples=["Panthera leo", "4CGXP", "Homo sapiens", "4CGXS"]
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
                ),
                AgentEntrypoint(
                    id="get_taxon_details",
                    description="Retrieve complete taxonomic details for a specific taxon using its Catalogue of Life ID. Use this when user provides a taxon ID from search results.",
                    parameters=TaxonDetailsParameters
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description="Get all synonyms (alternative scientific names) for a specific taxon using either its scientific name or taxon ID",
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
                    "limit": min(query_params.limit or 5, 20),
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
                        
                        # Extract taxon ID
                        taxon_id = item.get("id", "")
                        
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
                            "id": taxon_id,
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
                        "id": r["id"],
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
                # STEP 9: Generate response
                # ============================================================
                print("DEBUG: Generating response...")
                
                if len(formatted_results) == 1:
                    main_result = formatted_results[0]
                    result_scientific_name = main_result['scientificName']
                    reply_text = f"Found {result_scientific_name} ({main_result['rank']}, {main_result['status']}). "
                    
                    query_lower = query_params.search_term.lower()
                    result_name_lower = result_scientific_name.lower()
                    
                    if (len(query_lower) > 3 and 
                        query_lower not in result_name_lower and 
                        result_name_lower.split()[0].lower() != query_lower):
                        
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
                await context.reply(reply_text)
                print("DEBUG: Response sent successfully!")
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

    async def _handle_taxon_details(self, context: ResponseContext, request: str, params: TaxonDetailsParameters):
        """Handle the get_taxon_details entrypoint"""
        async with context.begin_process(summary="Fetching taxon details") as process:
            process: IChatBioAgentProcess
            
            try:
                taxon_id = params.taxon_id.strip()
                
                # LOG 1: Request initiated
                await process.log(f"Fetching details for taxon ID: '{taxon_id}'")
                print(f"DEBUG: Fetching taxon details for ID: {taxon_id}")
                
                # Build API URL
                col_url = f"https://api.checklistbank.org/dataset/3LR/taxon/{taxon_id}"
                
                print(f"DEBUG: API URL: {col_url}")
                
                # LOG 2: API Query
                await process.log(
                    "API Query",
                    data={"endpoint": col_url}
                )
                
                # Execute API request
                print("DEBUG: Executing COL API request...")
                
                try:
                    response = requests.get(col_url, timeout=10)
                    print(f"DEBUG: API Response Status: {response.status_code}")
                    
                    if response.status_code == 404:
                        await process.log("Taxon ID not found")
                        await context.reply(f"Taxon ID '{taxon_id}' not found in Catalogue of Life. Please check the ID and try again.")
                        return
                    
                    if response.status_code != 200:
                        error_msg = f"Catalogue of Life API error: HTTP {response.status_code}"
                        print(f"DEBUG: {error_msg}")
                        await process.log(f"API error: {error_msg}")
                        await context.reply(error_msg)
                        return
                    
                    data = response.json()
                    print(f"DEBUG: Parsed JSON successfully")
                    
                except requests.RequestException as req_error:
                    print(f"DEBUG: Request error: {req_error}")
                    await process.log(f"Network error: {str(req_error)}")
                    await context.reply("Error: Could not connect to Catalogue of Life API")
                    return
                except json.JSONDecodeError:
                    print(f"DEBUG: JSON decode error")
                    await process.log("Failed to parse API response")
                    await context.reply("Error: Invalid response from Catalogue of Life API")
                    return
                
                # Extract taxon information
                print("DEBUG: Extracting taxon details...")
                
                name_obj = data.get("name", {})
                scientific_name = name_obj.get("scientificName", "Unknown")
                authorship = name_obj.get("authorship", "")
                rank = name_obj.get("rank", "Unknown")
                status = data.get("status", "Unknown")
                
                # Extract classification
                classification = data.get("classification", [])
                taxonomy = {}
                
                desired_ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
                
                for taxon in classification:
                    taxon_rank = taxon.get("rank", "").lower()
                    taxon_name = taxon.get("name", "")
                    if taxon_rank in desired_ranks and taxon_name:
                        taxonomy[taxon_rank] = taxon_name
                
                # Extract additional details
                extinct = data.get("extinct", False)
                environments = data.get("environments", [])
                link = data.get("link", "")
                
                # LOG 3: Details retrieved
                await process.log(
                    f"Retrieved details for: {scientific_name}",
                    data={
                        "scientific_name": scientific_name,
                        "rank": rank,
                        "status": status,
                        "extinct": extinct
                    }
                )
                
                # Build reply
                full_name = f"{scientific_name} {authorship}".strip()
                reply_text = f"**{full_name}**\n\n"
                reply_text += f"**Rank:** {rank}\n"
                reply_text += f"**Status:** {status}\n"
                
                if extinct:
                    reply_text += f"**Extinct:** Yes ☠️\n"
                
                if environments:
                    reply_text += f"**Environments:** {', '.join(environments)}\n"
                
                if taxonomy:
                    reply_text += f"\n**Complete Taxonomic Classification:**\n"
                    for rank_name in ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]:
                        if rank_name in taxonomy:
                            reply_text += f"- {rank_name.capitalize()}: {taxonomy[rank_name]}\n"
                
                if link:
                    reply_text += f"\n**External Link:** {link}\n"
                
                reply_text += f"\nSee artifact for complete data including references and additional details."
                
                # Create artifact
                print("DEBUG: Creating artifact...")
                
                artifact_data = {
                    "taxon_info": {
                        "id": taxon_id,
                        "scientific_name": scientific_name,
                        "authorship": authorship,
                        "rank": rank,
                        "status": status,
                        "extinct": extinct,
                        "environments": environments,
                        "link": link
                    },
                    "taxonomy": taxonomy,
                    "raw_response": data
                }
                
                try:
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Complete taxon details for {scientific_name}",
                        content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                        uris=[link] if link else [col_url],
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
                
                # Send response
                print("DEBUG: Sending response to user")
                await context.reply(reply_text)
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
                    f"Please check the taxon ID and try again."
                )

    async def _handle_get_synonyms(self, context: ResponseContext, request: str, params: GetSynonymsParameters):
        """Handle the get_synonyms entrypoint - accepts name or ID"""
        async with context.begin_process(summary="Fetching synonyms") as process:
            process: IChatBioAgentProcess
            
            try:
                query = params.query.strip()
                taxon_id = None
                scientific_name = None
                
                # Determine if input is a taxon ID (short alphanumeric) or scientific name (has spaces/longer)
                is_likely_id = len(query) < 10 and " " not in query and not query[0].isupper()
                
                if is_likely_id:
                    # Treat as taxon ID
                    taxon_id = query
                    await process.log(f"Using taxon ID: '{taxon_id}'")
                    print(f"DEBUG: Input identified as taxon ID: {taxon_id}")
                else:
                    # Treat as scientific name - need to search for it first
                    await process.log(f"Searching for scientific name: '{query}'")
                    print(f"DEBUG: Input identified as scientific name: {query}")
                    
                    # Search for the species to get its taxon ID
                    search_url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
                    search_params = {
                        "q": query,
                        "limit": 1,
                        "content": "SCIENTIFIC_NAME"
                    }
                    
                    try:
                        search_response = requests.get(search_url, params=search_params, timeout=10)
                        
                        if search_response.status_code != 200:
                            await process.log(f"Search failed: HTTP {search_response.status_code}")
                            await context.reply(f"Could not search for '{query}'. Please check the name and try again.")
                            return
                        
                        search_data = search_response.json()
                        results = search_data.get("result", [])
                        
                        if len(results) == 0:
                            await process.log("No matching species found")
                            await context.reply(f"No species found for '{query}'. Please check the spelling or try the scientific name.")
                            return
                        
                        # Get the first result (most relevant)
                        first_result = results[0]
                        taxon_id = first_result.get("id")
                        usage = first_result.get("usage", {})
                        name_obj = usage.get("name", {})
                        scientific_name = name_obj.get("scientificName", query)
                        
                        await process.log(f"Found taxon: {scientific_name} (ID: {taxon_id})")
                        print(f"DEBUG: Found taxon ID: {taxon_id} for name: {scientific_name}")
                        
                    except requests.RequestException as search_error:
                        print(f"DEBUG: Search error: {search_error}")
                        await process.log(f"Search error: {str(search_error)}")
                        await context.reply("Error: Could not search for species")
                        return
                    except json.JSONDecodeError:
                        await process.log("Failed to parse search response")
                        await context.reply("Error: Invalid search response")
                        return
                
                if not taxon_id:
                    await context.reply("Could not determine taxon ID. Please try again.")
                    return
                
                # Now fetch synonyms using the taxon ID
                col_url = f"https://api.checklistbank.org/dataset/3LR/taxon/{taxon_id}/synonyms"
                
                print(f"DEBUG: Fetching synonyms from: {col_url}")
                
                # LOG: API Query
                await process.log(
                    "API Query",
                    data={
                        "endpoint": col_url,
                        "taxon_id": taxon_id
                    }
                )
                
                # Execute API request
                print("DEBUG: Executing COL API request...")
                
                try:
                    response = requests.get(col_url, timeout=10)
                    print(f"DEBUG: API Response Status: {response.status_code}")
                    
                    if response.status_code == 404:
                        await process.log("Taxon ID not found")
                        await context.reply(f"Taxon ID '{taxon_id}' not found. The species may not have synonym data available.")
                        return
                    
                    if response.status_code != 200:
                        error_msg = f"Catalogue of Life API error: HTTP {response.status_code}"
                        print(f"DEBUG: {error_msg}")
                        await process.log(f"API error: {error_msg}")
                        await context.reply(error_msg)
                        return
                    
                    data = response.json()
                    print(f"DEBUG: Parsed JSON successfully")
                    synonyms_data = data.get("heterotypic", [])
                    print(f"DEBUG: Number of synonyms: {len(synonyms_data)}")
                    
                except requests.RequestException as req_error:
                    print(f"DEBUG: Request error: {req_error}")
                    await process.log(f"Network error: {str(req_error)}")
                    await context.reply("Error: Could not connect to Catalogue of Life API")
                    return
                except json.JSONDecodeError:
                    print(f"DEBUG: JSON decode error")
                    await process.log("Failed to parse API response")
                    await context.reply("Error: Invalid response from Catalogue of Life API")
                    return
                
                # Check if there are any synonyms
                if not synonyms_data or len(synonyms_data) == 0:
                    await process.log("No synonyms found")
                    display_name = scientific_name if scientific_name else f"taxon ID '{taxon_id}'"
                    await context.reply(f"No synonyms found for {display_name}. This may be the currently accepted name with no alternative names.")
                    return
                
                # Extract synonym information
                print("DEBUG: Extracting synonym details...")
                
                synonyms_list = []
                
                for item in synonyms_data:
                    try:
                        name_obj = item.get("name", {})
                        syn_scientific_name = name_obj.get("scientificName", "Unknown")
                        authorship = name_obj.get("authorship", "")
                        rank = name_obj.get("rank", "Unknown")
                        status = item.get("status", "Unknown")
                        
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
                
                # Build reply
                display_name = scientific_name if scientific_name else f"taxon ID {taxon_id}"
                reply_text = f"**Found {len(synonyms_list)} synonym(s) for {display_name}:**\n\n"
                
                for i, synonym in enumerate(synonyms_list, 1):
                    full_name = f"{synonym['scientificName']} {synonym['authorship']}".strip()
                    reply_text += f"{i}. *{full_name}*\n"
                    reply_text += f"   - Rank: {synonym['rank']}\n"
                    reply_text += f"   - Status: {synonym['status']}\n\n"
                
                reply_text += f"\nSee artifact for complete synonym data."
                
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
                
                # Send response
                print("DEBUG: Sending response to user")
                await context.reply(reply_text)
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