"""
Catalogue of Life Agent for iChatBio Platform
==============================================

This agent provides access to the Catalogue of Life database through the ChecklistBank API.
It enables biodiversity researchers and educators to query species information, taxonomic
classifications, synonyms, and vernacular names.

API Documentation: https://api.checklistbank.org
Dataset: Catalogue of Life Latest Release (3LR)

Author: Krishna Modi
Version: 2.1.0
License: MIT
"""

import os
import json
import logging
from typing import Optional, Union, Literal
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


# Configuration
dotenv.load_dotenv()

COL_BASE_URL = "https://api.checklistbank.org"
COL_DATASET_KEY = "3LR"
COL_TIMEOUT = 10
MAX_SEARCH_RESULTS = 20
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHILDREN_LIMIT = 20

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Entrypoint Parameter Models

class SearchParameters(BaseModel):
    """
    Parameters for searching species by scientific name.
    The ChecklistBank API requires scientific names, not common names.
    """
    query: str = Field(
        description=(
            "Scientific name to search. Examples: 'Panthera leo', 'Homo sapiens', 'Quercus'. "
            "Can be a species name, genus, family, or any taxonomic rank. "
            "Common names are not supported by this endpoint."
        ),
        examples=["Panthera leo", "Homo sapiens", "Quercus", "Felidae"]
    )
    limit: Optional[int] = Field(
        default=DEFAULT_SEARCH_LIMIT,
        description="Maximum number of results to return",
        ge=1,
        le=MAX_SEARCH_RESULTS
    )


class TaxonDetailsParameters(BaseModel):
    """
    Parameters for retrieving detailed taxonomic information.
    Accepts either a COL taxon ID or a scientific name.
    """
    taxon_id: str = Field(
        description=(
            "Catalogue of Life taxon ID (e.g., '4CGXP') or scientific name (e.g., 'Panthera leo'). "
            "If a scientific name is provided, the agent will search for the taxon ID first."
        ),
        examples=["4CGXP", "Panthera leo", "5K5L5", "Homo sapiens"]
    )


class GetSynonymsParameters(BaseModel):
    """
    Parameters for retrieving taxonomic synonyms.
    Returns alternative scientific names for the same taxon.
    """
    query: str = Field(
        description=(
            "Scientific name or taxon ID to get synonyms for. "
            "Example: 'Panthera leo' or '4CGXP'. "
            "Common names are not supported."
        ),
        examples=["Panthera leo", "4CGXP", "Homo sapiens"]
    )


class GetVernacularNamesParameters(BaseModel):
    """
    Parameters for retrieving vernacular (common) names in multiple languages.
    """
    taxon_id: str = Field(
        description=(
            "Taxon ID or scientific name to get common names for. "
            "Example: '4CGXP' or 'Panthera leo'. "
            "Returns names in all available languages."
        ),
        examples=["4CGXP", "Panthera leo", "5K5L5"]
    )


class GetClassificationParameters(BaseModel):
    """
    Parameters for retrieving complete taxonomic classification lineage.
    Returns the full hierarchy from genus up to domain.
    """
    taxon_id: str = Field(
        description=(
            "Taxon ID or scientific name to get classification for. "
            "Example: '4CGXP' or 'Panthera leo'. "
            "Returns parent lineage from genus to domain."
        ),
        examples=["4CGXP", "Panthera leo", "5K5L5", "Homo sapiens"]
    )


class GetTaxonChildrenParameters(BaseModel):
    """
    Parameters for retrieving child taxa (e.g., all species in a genus).
    Returns immediate children only, not full descendant tree.
    """
    taxon_id: str = Field(
        description=(
            "Taxon ID or scientific name to get children for. "
            "Example: '6DBT' (genus Panthera) or 'Panthera'. "
            "Returns all child taxa (e.g., species in a genus)."
        ),
        examples=["6DBT", "Panthera", "Felidae", "Carnivora"]
    )
    limit: Optional[int] = Field(
        default=DEFAULT_CHILDREN_LIMIT,
        description="Maximum number of children to return",
        ge=1,
        le=100
    )


class CatalogueOfLifeAgent(IChatBioAgent):
    """
    Main agent class for interacting with the Catalogue of Life database.
    
    This agent provides six main entrypoints:
    1. search - Find species by scientific name
    2. get_taxon_details - Retrieve complete taxonomic information
    3. get_synonyms - Get alternative scientific names
    4. get_vernacular_names - Get common names in various languages
    5. get_classification - Get complete taxonomic hierarchy
    6. get_taxon_children - Get child taxa for a given taxon
    """
    
    def __init__(self, dataset_key: str = COL_DATASET_KEY, timeout: int = COL_TIMEOUT):
        """
        Initialize the Catalogue of Life agent.
        
        Args:
            dataset_key: ChecklistBank dataset identifier (default: 3LR for latest release)
            timeout: API request timeout in seconds
        """
        super().__init__()
        self.dataset_key = dataset_key
        self.timeout = timeout
        
        # Initialize OpenAI client for future LLM-enhanced features
        # Currently unused but available for query parsing and parameter extraction
        try:
            self.openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            self.instructor_client = instructor.patch(self.openai_client)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
            self.instructor_client = None
    
    @override
    def get_agent_card(self) -> AgentCard:
        """
        Returns the agent card with entrypoint definitions.
        iChatBio uses this card to determine how to interact with the agent.
        """
        return AgentCard(
            name="Catalogue of Life Agent",
            description=(
                "Access the Catalogue of Life database to search for species information, "
                "taxonomic classifications, synonyms, and common names. "
                "Searches must use scientific names, not common names."
            ),
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="search",
                    description=(
                        "Search for species using scientific names when you need to find taxa or get an overview. "
                        "Returns taxonomic information including classification, rank, and status for multiple matching results. "
                        "Best for: discovering species, exploring taxonomy, finding taxon IDs, getting quick overviews."
                    ),
                    parameters=SearchParameters
                ),
                AgentEntrypoint(
                    id="get_taxon_details",
                    description=(
                        "Retrieve comprehensive details for a specific taxon: full classification, authorship, "
                        "extinction status, habitat information, and direct link to Catalogue of Life page. "
                        "Best for: complete information about a known species, extinction data, habitat details."
                    ),
                    parameters=TaxonDetailsParameters
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description=(
                        "Get all alternative scientific names (synonyms) for a taxon. "
                        "Returns historical names and nomenclatural variants. "
                        "Best for: taxonomic history, alternative names, nomenclature research."
                    ),
                    parameters=GetSynonymsParameters
                ),
                AgentEntrypoint(
                    id="get_vernacular_names",
                    description=(
                        "Get common names in various languages for a taxon. "
                        "Returns vernacular names used in different regions and languages. "
                        "Best for: common names, translations, regional names."
                    ),
                    parameters=GetVernacularNamesParameters
                ),
                AgentEntrypoint(
                    id="get_classification",
                    description=(
                        "Get ONLY the taxonomic classification hierarchy showing parent lineage from genus up to domain. "
                        "Returns a clean hierarchy: genus → family → order → class → phylum → kingdom → domain. "
                        "Best for: answering 'what family/order/class', taxonomic hierarchy questions, parent lineage. "
                        "More focused than get_taxon_details which includes additional information beyond just classification."
                    ),
                    parameters=GetClassificationParameters
                ),
                AgentEntrypoint(
                    id="get_taxon_children",
                    description=(
                        "Get all immediate child taxa of a given taxon (e.g., all species in a genus, all genera in a family). "
                        "Returns names, ranks, and status for up to 100 children. "
                        "Best for: exploring taxonomic groups, listing species within a genus, discovering related taxa."
                    ),
                    parameters=GetTaxonChildrenParameters
                ),
            ]
        )
    
    # Utility Methods
    
    def _is_taxon_id(self, query: str) -> bool:
        """
        Determine if a query string is likely a taxon ID rather than a scientific name.
        
        COL taxon IDs are typically:
        - Short alphanumeric codes (1-10 characters)
        - Contain digits and/or uppercase letters
        - Examples: '4CGXP', '5K5L5', 'RT', 'N'
        
        Scientific names are typically:
        - Start with uppercase letter followed by lowercase
        - Contain spaces for binomial names
        - Examples: 'Panthera leo', 'Homo sapiens', 'Quercus'
        
        Args:
            query: The input string to classify
            
        Returns:
            True if the query appears to be a taxon ID, False otherwise
        """
        query = query.strip()
        
        if " " in query:
            return False
        
        if len(query) <= 2 and query.isupper():
            return True
        
        if query[0].isupper() and len(query) > 6:
            return False
        
        has_digit = any(c.isdigit() for c in query)
        has_upper = any(c.isupper() for c in query)
        
        if has_digit and has_upper:
            return True
        
        if query.isupper() and len(query) <= 10:
            return True
        
        return False
    
    async def _make_api_request(
        self,
        process: IChatBioAgentProcess,
        url: str,
        params: Optional[dict] = None,
        expected_structure: Literal["dict", "list"] = "dict"
    ) -> Optional[Union[dict, list]]:
        """
        Centralized method for making API requests to ChecklistBank.
        Handles error cases and logs all requests consistently.
        
        Args:
            process: The current process context for logging
            url: Full API endpoint URL
            params: Optional query parameters
            expected_structure: Expected response type ('dict' or 'list')
            
        Returns:
            Parsed JSON response or None if request failed
        """
        full_url = f"{url}?{urlencode(params)}" if params else url
        
        await process.log(
            "Executing API request",
            data={"endpoint": url, "parameters": params or {}, "full_url": full_url}
        )
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            logger.info(f"API request to {url} returned status {response.status_code}")
            
            if response.status_code == 404:
                await process.log("Resource not found in database")
                return None
            
            if response.status_code != 200:
                await process.log(f"API error: HTTP {response.status_code}")
                return None
            
            data = response.json()
            
            if expected_structure == "dict" and not isinstance(data, dict):
                await process.log(f"Unexpected response structure: expected dict, got {type(data).__name__}")
                return None
            
            if expected_structure == "list" and not isinstance(data, list):
                await process.log(f"Unexpected response structure: expected list, got {type(data).__name__}")
                return None
            
            return data
            
        except requests.Timeout:
            await process.log(f"Request timeout after {self.timeout} seconds")
            logger.error(f"Timeout on request to {url}")
            return None
            
        except requests.RequestException as e:
            await process.log(f"Network error: {str(e)}")
            logger.error(f"Request exception: {e}")
            return None
            
        except json.JSONDecodeError:
            await process.log("Failed to parse API response as JSON")
            logger.error("JSON decode error")
            return None
    
    async def _create_json_artifact(
        self,
        process: IChatBioAgentProcess,
        data: dict,
        description: str,
        uris: list[str],
        metadata: dict
    ) -> bool:
        """
        Centralized method for creating JSON artifacts.
        
        Args:
            process: The current process context
            data: Data to include in the artifact
            description: Human-readable description
            uris: List of URIs related to the artifact
            metadata: Additional metadata
            
        Returns:
            True if artifact was created successfully, False otherwise
        """
        try:
            await process.create_artifact(
                mimetype="application/json",
                description=description,
                content=json.dumps(data, indent=2).encode('utf-8'),
                uris=uris,
                metadata=metadata
            )
            logger.info(f"Created artifact: {description}")
            return True
        except Exception as e:
            await process.log(f"Failed to create artifact: {str(e)}")
            logger.error(f"Artifact creation failed: {e}")
            return False
    
    async def _search_for_taxon_id(
        self,
        process: IChatBioAgentProcess,
        scientific_name: str
    ) -> Optional[tuple[str, str]]:
        """
        Search for a taxon ID using a scientific name.
        This is used when a scientific name is provided but a taxon ID is needed.
        
        Args:
            process: The current process context for logging
            scientific_name: The scientific name to search for
            
        Returns:
            Tuple of (taxon_id, found_scientific_name) if found, None otherwise
        """
        url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/search"
        params = {
            "q": scientific_name,
            "limit": 1,
            "content": "SCIENTIFIC_NAME"
        }
        
        data = await self._make_api_request(process, url, params, expected_structure="dict")
        
        if not data:
            return None
        
        results = data.get("result", [])
        
        if len(results) == 0:
            await process.log(f"No results found for '{scientific_name}'")
            return None
        
        first_result = results[0]
        taxon_id = first_result.get("id")
        usage = first_result.get("usage", {})
        name_obj = usage.get("name", {})
        found_name = name_obj.get("scientificName", "")
        
        await process.log(f"Found taxon: {found_name} (ID: {taxon_id})")
        return (taxon_id, found_name)
    
    def _format_classification(self, taxonomy: dict) -> str:
        """
        Format a taxonomy dictionary into a readable classification string.
        
        Args:
            taxonomy: Dictionary mapping rank names to taxon names
            
        Returns:
            Formatted classification string with hierarchy
        """
        if not taxonomy:
            return ""
        
        ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
        lines = []
        
        for rank in ranks:
            if rank in taxonomy:
                lines.append(f"- {rank.capitalize()}: {taxonomy[rank]}")
        
        return "\n".join(lines)
    
    # Entrypoint Handlers
    
    async def _handle_search(
        self,
        context: ResponseContext,
        request: str,
        params: SearchParameters
    ):
        """
        Handle the search entrypoint for finding species by scientific name.
        
        Args:
            context: Response context for communicating with iChatBio
            request: Original natural language request from user
            params: Validated search parameters
        """
        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            search_term = params.query.strip()
            limit = params.limit or DEFAULT_SEARCH_LIMIT
            
            await process.log(f"Initiating search for: '{search_term}'")
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/search"
            api_params = {
                "q": search_term,
                "limit": min(limit, MAX_SEARCH_RESULTS),
                "content": "SCIENTIFIC_NAME"
            }
            
            data = await self._make_api_request(process, url, api_params)
            
            if not data:
                await context.reply(
                    f"Unable to complete search for '{search_term}'. "
                    "Please verify your internet connection and try again."
                )
                return
            
            results = data.get("result", [])
            total = data.get("total", 0)

            # Prioritize exact matches for binomial names (e.g., "Rattus rattus")
            if " " in search_term and len(results) > 0:
                search_term_lower = search_term.lower().strip()
                
                exact_matches = []
                other_matches = []
                
                for item in results:
                    usage = item.get("usage", {})
                    name_obj = usage.get("name", {})
                    scientific_name = name_obj.get("scientificName", "").lower()
                    
                    # Check for exact match (ignoring authorship)
                    if scientific_name == search_term_lower:
                        exact_matches.append(item)
                    else:
                        other_matches.append(item)
                
                # Reorder: exact matches first
                if exact_matches:
                    results = exact_matches + other_matches[:limit-len(exact_matches)]
                    await process.log(f"Prioritized {len(exact_matches)} exact match(es)")

            if len(results) == 0:
                await process.log("No results found")
                await context.reply(
                    f"No species found for '{search_term}' in the Catalogue of Life.\n\n"
                    "Suggestions:\n"
                    "- Ensure you are using a scientific name (common names are not supported)\n"
                    "- Check the spelling of the scientific name\n"
                    "- Try using just the genus name\n"
                    "- Try a broader taxonomic group\n\n"
                    "Examples of valid searches:\n"
                    "- Species: 'Homo sapiens', 'Panthera leo'\n"
                    "- Genus: 'Panthera', 'Quercus'\n"
                    "- Family: 'Felidae', 'Canidae'"
                )
                return
            
            formatted_results = []
            
            for item in results:
                try:
                    taxon_id = item.get("id", "")
                    usage = item.get("usage", {})
                    name_obj = usage.get("name", {})
                    
                    scientific_name = name_obj.get("scientificName", "Unknown")
                    rank = name_obj.get("rank", "Unknown")
                    status = usage.get("status", "Unknown")
                    
                    classification = item.get("classification", [])
                    taxonomy = {}
                    
                    desired_ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
                    
                    for taxon in classification:
                        taxon_rank = taxon.get("rank", "").lower()
                        taxon_name = taxon.get("name", "")
                        if taxon_rank in desired_ranks and taxon_name:
                            taxonomy[taxon_rank] = taxon_name
                    
                    formatted_results.append({
                        "id": taxon_id,
                        "scientificName": scientific_name,
                        "rank": rank,
                        "status": status,
                        "taxonomy": taxonomy
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            await process.log(
                f"Found {total} matches, showing top {len(formatted_results)}",
                data={
                    "results": [
                        {
                            "id": r["id"],
                            "scientific_name": r["scientificName"],
                            "rank": r["rank"],
                            "status": r["status"]
                        }
                        for r in formatted_results
                    ]
                }
            )
            
            if len(formatted_results) == 1:
                result = formatted_results[0]
                reply_text = (
                    f"Found {result['scientificName']} ({result['rank']}, {result['status']}).\n"
                    f"Taxon ID: {result['id']}\n\n"
                    "See artifact for complete taxonomic data."
                )
            else:
                result = formatted_results[0]
                reply_text = (
                    f"Found {total} matches for '{search_term}'.\n"
                    f"Top result: {result['scientificName']} ({result['rank']})\n\n"
                    "See artifact for complete results."
                )
            
            artifact_data = {
                "search_info": {
                    "query": search_term,
                    "original_request": request,
                    "total_found": total,
                    "showing": len(formatted_results)
                },
                "results": formatted_results,
                "raw_response": data
            }
            
            await self._create_json_artifact(
                process,
                artifact_data,
                f"COL search results for '{search_term}' - {len(formatted_results)} of {total} results",
                [f"{url}?{urlencode(api_params)}"],
                {
                    "data_source": "Catalogue of Life",
                    "query": search_term,
                    "total_found": total,
                    "results_count": len(formatted_results)
                }
            )
            
            await context.reply(reply_text)
            await process.log("Search completed successfully")
    
    async def _handle_taxon_details(
        self,
        context: ResponseContext,
        request: str,
        params: TaxonDetailsParameters
    ):
        """
        Handle the get_taxon_details entrypoint for retrieving complete taxonomic information.
        
        Args:
            context: Response context for communicating with iChatBio
            request: Original natural language request from user
            params: Validated taxon details parameters
        """
        async with context.begin_process(summary="Fetching taxon details") as process:
            query = params.taxon_id.strip()
            taxon_id = None
            scientific_name = None
            
            if self._is_taxon_id(query):
                taxon_id = query
                await process.log(f"Using taxon ID: '{taxon_id}'")
            else:
                await process.log(f"Searching for taxon ID using scientific name: '{query}'")
                result = await self._search_for_taxon_id(process, query)
                
                if not result:
                    await context.reply(
                        f"No exact match found for '{query}' in the Catalogue of Life.\n"
                        "Please check the spelling and try again."
                    )
                    return
                
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}"
            
            data = await self._make_api_request(process, url)
            
            if not data:
                await context.reply(
                    f"Taxon ID '{taxon_id}' not found in the Catalogue of Life.\n"
                    "Please verify the ID and try again."
                )
                return
            
            name_obj = data.get("name", {})
            scientific_name = name_obj.get("scientificName", "Unknown")
            authorship = name_obj.get("authorship", "")
            rank = name_obj.get("rank", "Unknown")
            status = data.get("status", "Unknown")
            
            classification = data.get("classification", [])
            taxonomy = {}
            
            desired_ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
            
            for taxon in classification:
                taxon_rank = taxon.get("rank", "").lower()
                taxon_name = taxon.get("name", "")
                if taxon_rank in desired_ranks and taxon_name:
                    taxonomy[taxon_rank] = taxon_name
            
            extinct = data.get("extinct", False)
            environments = data.get("environments", [])
            link = data.get("link", "")
            
            await process.log(
                f"Retrieved details for: {scientific_name}",
                data={
                    "scientific_name": scientific_name,
                    "rank": rank,
                    "status": status,
                    "extinct": extinct
                }
            )
            
            full_name = f"{scientific_name} {authorship}".strip()
            reply_text = f"**{full_name}**\n\n"
            reply_text += f"**Taxon ID:** {taxon_id}\n"
            reply_text += f"**Rank:** {rank}\n"
            reply_text += f"**Status:** {status}\n"
            
            if extinct:
                reply_text += "**Extinct:** Yes\n"
            
            if environments:
                reply_text += f"**Environments:** {', '.join(environments)}\n"
            
            if taxonomy:
                reply_text += "\n**Complete Taxonomic Classification:**\n"
                reply_text += self._format_classification(taxonomy)
            
            col_page = f"https://www.checklistbank.org/dataset/{self.dataset_key}/taxon/{taxon_id}"
            reply_text += f"\n\n**Catalogue of Life Page:** {col_page}\n"
            
            if link:
                reply_text += f"**Original Data Source:** {link}\n"
            
            reply_text += "\nSee artifact for complete data including references and additional details."
            
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
            
            await self._create_json_artifact(
                process,
                artifact_data,
                f"Complete taxon details for {scientific_name}",
                [link] if link else [url],
                {
                    "data_source": "Catalogue of Life",
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name,
                    "rank": rank
                }
            )
            
            await context.reply(reply_text)
            await process.log("Taxon details retrieved successfully")
    
    async def _handle_get_synonyms(
        self,
        context: ResponseContext,
        request: str,
        params: GetSynonymsParameters
    ):
        """
        Handle the get_synonyms entrypoint for retrieving alternative scientific names.
        
        Args:
            context: Response context for communicating with iChatBio
            request: Original natural language request from user
            params: Validated synonyms parameters
        """
        async with context.begin_process(summary="Fetching synonyms") as process:
            query = params.query.strip()
            taxon_id = None
            scientific_name = None
            
            if self._is_taxon_id(query):
                taxon_id = query
                await process.log(f"Using taxon ID: '{taxon_id}'")
            else:
                await process.log(f"Searching for taxon ID using scientific name: '{query}'")
                result = await self._search_for_taxon_id(process, query)
                
                if not result:
                    await context.reply(
                        f"No species found for '{query}' in the Catalogue of Life.\n"
                        "Please check the spelling or use a valid taxon ID."
                    )
                    return
                
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/info"
            
            data = await self._make_api_request(process, url)
            
            if not data:
                await context.reply(
                    f"Unable to retrieve information for taxon ID '{taxon_id}'.\n"
                    "The taxon may not have synonym data available."
                )
                return
            
            synonyms_data = data.get("synonyms", {})
            heterotypic = synonyms_data.get("heterotypic", [])
            homotypic = synonyms_data.get("homotypic", [])
            
            all_synonyms = heterotypic + homotypic
            
            if len(all_synonyms) == 0:
                await process.log("No synonyms found")
                display_name = scientific_name if scientific_name else f"taxon ID '{taxon_id}'"
                await context.reply(
                    f"No synonyms found for {display_name}.\n"
                    "This may be the currently accepted name with no alternative names."
                )
                return
            
            synonyms_list = []
            
            for item in all_synonyms:
                try:
                    name_obj = item.get("name", {})
                    syn_scientific_name = name_obj.get("scientificName", "Unknown")
                    authorship = name_obj.get("authorship", "")
                    rank = name_obj.get("rank", "Unknown")
                    status = item.get("status", "Unknown")
                    
                    synonyms_list.append({
                        "scientificName": syn_scientific_name,
                        "authorship": authorship,
                        "rank": rank,
                        "status": status
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing synonym: {e}")
                    continue
            
            await process.log(
                f"Found {len(synonyms_list)} synonyms",
                data={
                    "synonyms": [
                        {"scientific_name": s["scientificName"], "status": s["status"]}
                        for s in synonyms_list[:10]
                    ]
                }
            )
            
            display_name = scientific_name if scientific_name else f"taxon ID {taxon_id}"
            reply_text = f"**Found {len(synonyms_list)} synonym(s) for {display_name}:**\n\n"
            
            for i, synonym in enumerate(synonyms_list, 1):
                full_name = f"{synonym['scientificName']} {synonym['authorship']}".strip()
                reply_text += f"{i}. *{full_name}*\n"
                reply_text += f"   - Rank: {synonym['rank']}\n"
                reply_text += f"   - Status: {synonym['status']}\n\n"
            
            reply_text += "See artifact for complete synonym data."
            
            artifact_data = {
                "query": query,
                "taxon_id": taxon_id,
                "scientific_name": scientific_name,
                "synonym_count": len(synonyms_list),
                "synonyms": synonyms_list,
                "raw_response": data
            }
            
            await self._create_json_artifact(
                process,
                artifact_data,
                f"Synonyms for {scientific_name or taxon_id} - {len(synonyms_list)} total",
                [url],
                {
                    "data_source": "Catalogue of Life",
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name or "Unknown",
                    "synonym_count": len(synonyms_list)
                }
            )
            
            await context.reply(reply_text)
            await process.log("Synonyms retrieved successfully")
    
    async def _handle_vernacular_names(
        self,
        context: ResponseContext,
        request: str,
        params: GetVernacularNamesParameters
    ):
        """
        Handle the get_vernacular_names entrypoint for retrieving common names in multiple languages.
        
        Args:
            context: Response context for communicating with iChatBio
            request: Original natural language request from user
            params: Validated vernacular names parameters
        """
        async with context.begin_process(summary="Fetching vernacular names") as process:
            query = params.taxon_id.strip()
            taxon_id = None
            scientific_name = None
            
            if self._is_taxon_id(query):
                taxon_id = query
                await process.log(f"Using taxon ID: '{taxon_id}'")
            else:
                await process.log(f"Searching for taxon ID using scientific name: '{query}'")
                result = await self._search_for_taxon_id(process, query)
                
                if not result:
                    await context.reply(
                        f"No exact match found for '{query}' in the Catalogue of Life.\n"
                        "Please check the spelling and try again."
                    )
                    return
                
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/vernacular"
            
            data = await self._make_api_request(process, url, expected_structure="list")
            
            if not data:
                await context.reply(
                    f"Unable to retrieve vernacular names for taxon ID '{taxon_id}'.\n"
                    "Please verify the ID and try again."
                )
                return
            
            if len(data) == 0:
                await process.log("No vernacular names found")
                display_name = scientific_name if scientific_name else f"taxon ID '{taxon_id}'"
                await context.reply(
                    f"No common names found for {display_name}.\n"
                    "This species may not have vernacular names in the database."
                )
                return
            
            names_by_language = {}
            total_names = 0
            
            for item in data:
                try:
                    name = item.get("name", "")
                    language = item.get("language", "Unknown")
                    
                    if name:
                        if language not in names_by_language:
                            names_by_language[language] = []
                        
                        names_by_language[language].append(name)
                        total_names += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing vernacular name: {e}")
                    continue
            
            await process.log(
                f"Found {total_names} vernacular names in {len(names_by_language)} languages",
                data={"languages": list(names_by_language.keys())}
            )
            
            display_name = scientific_name if scientific_name else f"taxon ID {taxon_id}"
            reply_text = f"**Common names found for {display_name}:**\n\n"
            
            for language in sorted(names_by_language.keys()):
                names = names_by_language[language]
                reply_text += f"**{language}:**\n"
                for name in sorted(names):
                    reply_text += f"  - {name}\n"
                reply_text += "\n"
            
            reply_text += f"Total: {total_names} names in {len(names_by_language)} languages\n\n"
            reply_text += "See artifact for complete data."
            
            artifact_data = {
                "query": query,
                "taxon_id": taxon_id,
                "scientific_name": scientific_name or "Not provided",
                "total_names": total_names,
                "language_count": len(names_by_language),
                "names_by_language": names_by_language,
                "raw_response": data
            }
            
            await self._create_json_artifact(
                process,
                artifact_data,
                f"Vernacular names for {scientific_name or taxon_id} - {total_names} names in {len(names_by_language)} languages",
                [url],
                {
                    "data_source": "Catalogue of Life",
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name or "Unknown",
                    "total_names": total_names,
                    "language_count": len(names_by_language)
                }
            )
            
            await context.reply(reply_text)
            await process.log("Vernacular names retrieved successfully")
    
    async def _handle_classification(
        self,
        context: ResponseContext,
        request: str,
        params: GetClassificationParameters
    ):
        """
        Handle the get_classification entrypoint for retrieving taxonomic hierarchy.
        
        Args:
            context: Response context for communicating with iChatBio
            request: Original natural language request from user
            params: Validated classification parameters
        """
        async with context.begin_process(summary="Fetching classification hierarchy") as process:
            query = params.taxon_id.strip()
            taxon_id = None
            scientific_name = None
            
            if self._is_taxon_id(query):
                taxon_id = query
                await process.log(f"Using taxon ID: '{taxon_id}'")
            else:
                await process.log(f"Searching for taxon ID using scientific name: '{query}'")
                result = await self._search_for_taxon_id(process, query)
                
                if not result:
                    await context.reply(
                        f"No exact match found for '{query}' in the Catalogue of Life.\n"
                        "Please check the spelling and try again."
                    )
                    return
                
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/classification"
            
            data = await self._make_api_request(process, url, expected_structure="list")
            
            if not data:
                await context.reply(
                    f"Unable to retrieve classification for taxon ID '{taxon_id}'.\n"
                    "The taxon may not have classification data available."
                )
                return
            
            if len(data) == 0:
                await process.log("No classification data found")
                display_name = scientific_name if scientific_name else f"taxon ID '{taxon_id}'"
                await context.reply(
                    f"No classification hierarchy found for {display_name}.\n"
                    "This taxon may be at the top of the hierarchy."
                )
                return
            
            classification_list = []
            
            for item in data:
                try:
                    class_id = item.get("id", "")
                    name = item.get("name", "Unknown")
                    authorship = item.get("authorship", "")
                    rank = item.get("rank", "Unknown")
                    
                    classification_list.append({
                        "id": class_id,
                        "name": name,
                        "authorship": authorship,
                        "rank": rank
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing classification item: {e}")
                    continue
            
            await process.log(
                f"Retrieved {len(classification_list)} classification levels",
                data={
                    "levels": [
                        {"rank": c["rank"], "name": c["name"]}
                        for c in classification_list[:5]
                    ]
                }
            )
            
            display_name = scientific_name if scientific_name else f"taxon ID {taxon_id}"
            reply_text = f"**Complete Classification Hierarchy for {display_name}:**\n\n"
            
            for item in classification_list:
                full_name = f"{item['name']} {item['authorship']}".strip()
                reply_text += f"**{item['rank'].capitalize()}:** {full_name}\n"
            
            reply_text += f"\n**Total Levels:** {len(classification_list)}\n\n"
            reply_text += "See artifact for complete classification data."
            
            artifact_data = {
                "query": query,
                "taxon_id": taxon_id,
                "scientific_name": scientific_name or "Not provided",
                "classification_levels": len(classification_list),
                "classification": classification_list,
                "raw_response": data
            }
            
            await self._create_json_artifact(
                process,
                artifact_data,
                f"Classification hierarchy for {scientific_name or taxon_id} - {len(classification_list)} levels",
                [url],
                {
                    "data_source": "Catalogue of Life",
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name or "Unknown",
                    "levels": len(classification_list)
                }
            )
            
            await context.reply(reply_text)
            await process.log("Classification hierarchy retrieved successfully")
    
    async def _handle_taxon_children(
        self,
        context: ResponseContext,
        request: str,
        params: GetTaxonChildrenParameters
    ):
        """
        Handle the get_taxon_children entrypoint for retrieving child taxa.
        
        Args:
            context: Response context for communicating with iChatBio
            request: Original natural language request from user
            params: Validated taxon children parameters
        """
        async with context.begin_process(summary="Fetching child taxa") as process:
            query = params.taxon_id.strip()
            limit = params.limit or DEFAULT_CHILDREN_LIMIT
            taxon_id = None
            scientific_name = None
            
            if self._is_taxon_id(query):
                taxon_id = query
                await process.log(f"Using taxon ID: '{taxon_id}'")
            else:
                await process.log(f"Searching for taxon ID using scientific name: '{query}'")
                result = await self._search_for_taxon_id(process, query)
                
                if not result:
                    await context.reply(
                        f"No exact match found for '{query}' in the Catalogue of Life.\n"
                        "Please check the spelling and try again."
                    )
                    return
                
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/tree/{taxon_id}/children"
            api_params = {
                "limit": limit
            }
            
            data = await self._make_api_request(process, url, api_params, expected_structure="dict")
            
            if not data:
                await context.reply(
                    f"Unable to retrieve children for taxon ID '{taxon_id}'.\n"
                    "Please verify the ID and try again."
                )
                return
            
            results = data.get("result", [])
            total = data.get("total", 0)
            
            if len(results) == 0:
                await process.log("No children found")
                display_name = scientific_name if scientific_name else f"taxon ID '{taxon_id}'"
                await context.reply(
                    f"No child taxa found for {display_name}.\n"
                    "This taxon may be a terminal node (e.g., species with no subspecies)."
                )
                return
            
            children_list = []
            
            for item in results:
                try:
                    child_id = item.get("id", "")
                    child_name = item.get("name", "Unknown")
                    child_authorship = item.get("authorship", "")
                    child_rank = item.get("rank", "Unknown")
                    child_status = item.get("status", "Unknown")
                    
                    children_list.append({
                        "id": child_id,
                        "name": child_name,
                        "authorship": child_authorship,
                        "rank": child_rank,
                        "status": child_status
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing child taxon: {e}")
                    continue
            
            await process.log(
                f"Found {total} children, showing {len(children_list)}",
                data={
                    "children": [
                        {"name": c["name"], "rank": c["rank"], "status": c["status"]}
                        for c in children_list[:10]
                    ]
                }
            )
            
            display_name = scientific_name if scientific_name else f"taxon ID {taxon_id}"
            reply_text = f"**Found {total} child taxa for {display_name}:**\n\n"
            
            if total > limit:
                reply_text += f"*(Showing first {limit} of {total} total)*\n\n"
            
            for i, child in enumerate(children_list, 1):
                full_name = f"{child['name']} {child['authorship']}".strip()
                reply_text += f"{i}. **{full_name}**\n"
                reply_text += f"   - Rank: {child['rank']}\n"
                reply_text += f"   - Status: {child['status']}\n\n"
            
            reply_text += "See artifact for complete children data."
            
            artifact_data = {
                "query": query,
                "taxon_id": taxon_id,
                "scientific_name": scientific_name or "Not provided",
                "total_children": total,
                "showing": len(children_list),
                "children": children_list,
                "raw_response": data
            }
            
            await self._create_json_artifact(
                process,
                artifact_data,
                f"Children of {scientific_name or taxon_id} - {len(children_list)} of {total} total",
                [f"{url}?{urlencode(api_params)}"],
                {
                    "data_source": "Catalogue of Life",
                    "taxon_id": taxon_id,
                    "scientific_name": scientific_name or "Unknown",
                    "total_children": total,
                    "showing": len(children_list)
                }
            )
            
            await context.reply(reply_text)
            await process.log("Child taxa retrieved successfully")
    
    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: Union[SearchParameters, TaxonDetailsParameters, GetSynonymsParameters, GetVernacularNamesParameters, GetClassificationParameters, GetTaxonChildrenParameters]
    ):
        """
        Main agent entry point. Routes requests to the appropriate handler based on entrypoint.
        
        Args:
            context: Response context for communicating with iChatBio
            request: Natural language description of what the user wants
            entrypoint: The entrypoint ID selected by iChatBio
            params: Validated parameters for the selected entrypoint
        """
        logger.info(f"Agent invoked with entrypoint: {entrypoint}")
        logger.debug(f"Request: {request}")
        logger.debug(f"Parameters: {params}")
        
        try:
            if entrypoint == "search":
                await self._handle_search(context, request, params)
            elif entrypoint == "get_taxon_details":
                await self._handle_taxon_details(context, request, params)
            elif entrypoint == "get_synonyms":
                await self._handle_get_synonyms(context, request, params)
            elif entrypoint == "get_vernacular_names":
                await self._handle_vernacular_names(context, request, params)
            elif entrypoint == "get_classification":
                await self._handle_classification(context, request, params)
            elif entrypoint == "get_taxon_children":
                await self._handle_taxon_children(context, request, params)
            else:
                error_msg = f"Unknown entrypoint: {entrypoint}"
                logger.error(error_msg)
                await context.reply(
                    f"Error: Unrecognized entrypoint '{entrypoint}'.\n"
                    "Valid entrypoints: search, get_taxon_details, get_synonyms, get_vernacular_names, get_classification, get_taxon_children"
                )
        
        except Exception as e:
            logger.exception(f"Unexpected error in entrypoint handler: {e}")
            await context.reply(
                "An unexpected error occurred while processing your request.\n"
                "Please try again or contact support if the problem persists."
            )


def validate_environment():
    """
    Validate that required environment variables are set.
    Returns True if environment is properly configured, False otherwise.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set. LLM features will be unavailable.")
        return False
    
    return True


def run_agent_server():
    """
    Initialize and run the COL agent server.
    This function should be called from the main entry point.
    """
    logger.info("Initializing Catalogue of Life Agent")
    
    validate_environment()
    
    try:
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        logger.info(f"Agent card created: {card.name}")
        logger.info(f"Available entrypoints: {[ep.id for ep in card.entrypoints]}")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise
    
    try:
        from ichatbio.server import run_agent_server as start_server
        
        logger.info("Starting agent server on http://localhost:9999")
        logger.info("Agent card available at: http://localhost:9999/.well-known/agent.json")
        
        start_server(agent, host="0.0.0.0", port=9999)
        
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


if __name__ == "__main__":
    run_agent_server()
