"""
Catalogue of Life Agent
A professional implementation for searching and retrieving taxonomic information
from the Catalogue of Life database via the Checklistbank API.
"""

import os
import json
import traceback
import logging
from typing import Optional, List, Union, Dict, Any
from typing_extensions import override
from urllib.parse import urlencode
from dataclasses import dataclass

import dotenv
import instructor
import requests
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, validator

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Constants
COL_API_BASE = "https://api.checklistbank.org/dataset/3LR"
COL_WEB_BASE = "https://www.catalogueoflife.org/data/taxon"
DEFAULT_TIMEOUT = 10
MAX_RESULTS_LIMIT = 20
DEFAULT_RESULTS_LIMIT = 5

# Taxonomic ranks in hierarchical order
TAXONOMIC_RANKS = [
    "domain", "kingdom", "phylum", "class", 
    "order", "family", "genus", "species"
]


class SearchParameters(BaseModel):
    """Parameters for searching the Catalogue of Life"""
    query: str = Field(
        description="Search term for species or taxonomic group",
        examples=["tiger", "Panthera leo", "oak tree"],
        min_length=1,
        max_length=200
    )
    
    @validator('query')
    def clean_query(cls, v):
        return v.strip()


class TaxonDetailsParameters(BaseModel):
    """Parameters for retrieving detailed taxon information"""
    taxon_id: str = Field(
        description="Catalogue of Life taxon identifier",
        examples=["4CGXP", "5K5L5", "9BBLS"],
        min_length=1,
        max_length=20
    )
    
    @validator('taxon_id')
    def clean_taxon_id(cls, v):
        return v.strip()


class GetSynonymsParameters(BaseModel):
    """Parameters for retrieving taxonomic synonyms"""
    query: str = Field(
        description="Scientific name or taxon ID",
        examples=["Panthera leo", "4CGXP", "Homo sapiens"],
        min_length=1,
        max_length=200
    )
    
    @validator('query')
    def clean_query(cls, v):
        return v.strip()


class CoLQueryParams(BaseModel):
    """Validated parameters for COL API queries"""
    search_term: str = Field(
        description="Scientific or common name to search",
        min_length=2,
        max_length=100
    )
    limit: int = Field(
        default=DEFAULT_RESULTS_LIMIT,
        ge=1,
        le=MAX_RESULTS_LIMIT
    )


@dataclass
class TaxonInfo:
    """Structured taxon information"""
    id: str
    scientific_name: str
    rank: str
    status: str
    authorship: Optional[str] = None
    extinct: bool = False
    col_url: Optional[str] = None
    taxonomy: Optional[Dict[str, str]] = None


class APIError(Exception):
    """Custom exception for API-related errors"""
    pass


class CatalogueOfLifeAgent(IChatBioAgent):
    """Agent for interacting with the Catalogue of Life database"""
    
    def __init__(self):
        """Initialize the agent with necessary clients and configurations"""
        logger.info("Initializing Catalogue of Life Agent")
        super().__init__()
        
        # Initialize OpenAI client for query processing
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            self.openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            self.instructor_client = instructor.patch(self.openai_client)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    @override
    def get_agent_card(self) -> AgentCard:
        """Define the agent's capabilities and endpoints"""
        return AgentCard(
            name="Catalogue of Life Agent",
            description="Professional taxonomic information service using the Catalogue of Life database",
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="search",
                    description="Search for species or taxonomic groups",
                    parameters=SearchParameters
                ),
                AgentEntrypoint(
                    id="get_taxon_details",
                    description="Retrieve comprehensive taxonomic details for a specific taxon",
                    parameters=TaxonDetailsParameters
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description="Find alternative scientific names (synonyms) for a taxon",
                    parameters=GetSynonymsParameters
                )
            ]
        )
    
    async def _convert_to_scientific_name(self, query: str) -> CoLQueryParams:
        """Convert common names to scientific names using GPT"""
        conversion_prompt = """
        Convert the given query to the most appropriate scientific name for searching the Catalogue of Life database.
        
        Rules:
        1. Common to scientific name conversions:
           - lion -> Panthera leo
           - tiger -> Panthera tigris
           - elephant -> Loxodonta (African) or Elephas (Asian)
           - human -> Homo sapiens
           - dog -> Canis lupus familiaris
           - cat -> Felis catus
           - wolf -> Canis lupus
           - bear -> Ursus (genus)
           - eagle -> Aquila (genus) or specific species
           
        2. Plants (use botanical names):
           - oak/oak tree -> Quercus
           - pine/pine tree -> Pinus
           - maple -> Acer
           - rose -> Rosa
           - sunflower -> Helianthus
           
        3. Extinct species:
           - tyrannosaurus/t-rex -> Tyrannosaurus rex
           - mammoth -> Mammuthus
           - dodo -> Raphus cucullatus
           - dinosaur -> Dinosauria
           
        4. If already a scientific name, preserve it exactly
        5. When uncertain, use genus level (single word)
        
        Return only the scientific name, nothing else.
        """
        
        try:
            result = await self.instructor_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": conversion_prompt},
                    {"role": "user", "content": f"Convert: {query}"}
                ],
                response_model=CoLQueryParams,
                temperature=0,
                max_retries=2
            )
            logger.info(f"Query converted: '{query}' -> '{result.search_term}'")
            return result
            
        except Exception as e:
            logger.warning(f"GPT conversion failed: {e}. Using original query.")
            return CoLQueryParams(search_term=query, limit=DEFAULT_RESULTS_LIMIT)
    
    async def _make_api_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request to the COL API with error handling"""
        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise APIError("Request timeout - the service may be busy")
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot connect to Catalogue of Life service")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise APIError("Resource not found")
            raise APIError(f"HTTP {e.response.status_code}: {e.response.reason}")
        except json.JSONDecodeError:
            raise APIError("Invalid response format from service")
    
    def _extract_taxon_info(self, data: Dict[str, Any], taxon_id: Optional[str] = None) -> TaxonInfo:
        """Extract and structure taxon information from API response"""
        # Handle different response structures
        if "usage" in data:
            # Search result structure
            taxon_id = taxon_id or data.get("id", "")
            usage = data.get("usage", {})
            name_obj = usage.get("name", {})
            status = usage.get("status", "Unknown")
            extinct = usage.get("extinct", False)
        else:
            # Direct taxon structure
            taxon_id = taxon_id or data.get("id", "")
            name_obj = data.get("name", {})
            status = data.get("status", "Unknown")
            extinct = data.get("extinct", False)
        
        # Extract classification
        classification = data.get("classification", [])
        taxonomy = {}
        for item in classification:
            rank = item.get("rank", "").lower()
            if rank in TAXONOMIC_RANKS:
                taxonomy[rank] = item.get("name", "")
        
        return TaxonInfo(
            id=taxon_id,
            scientific_name=name_obj.get("scientificName", "Unknown"),
            rank=name_obj.get("rank", "Unknown"),
            status=status,
            authorship=name_obj.get("authorship", ""),
            extinct=extinct,
            col_url=f"{COL_WEB_BASE}/{taxon_id}" if taxon_id else None,
            taxonomy=taxonomy
        )
    
    def _format_search_results(self, results: List[TaxonInfo], total: int, query: str) -> str:
        """Format search results for user display"""
        if not results:
            return self._format_no_results_message(query)
        
        if len(results) == 1:
            result = results[0]
            extinct_note = " [EXTINCT]" if result.extinct else ""
            return (
                f"Found: **{result.scientific_name}**{extinct_note}\n"
                f"Rank: {result.rank}\n"
                f"Status: {result.status}\n"
                f"COL Page: {result.col_url}\n\n"
                f"See artifact for complete taxonomic data."
            )
        
        # Multiple results
        message = f"Found {total} matches for '{query}':\n\n"
        
        for i, result in enumerate(results[:3], 1):
            extinct_note = " [EXTINCT]" if result.extinct else ""
            message += (
                f"{i}. **{result.scientific_name}**{extinct_note}\n"
                f"   Rank: {result.rank} | Status: {result.status}\n"
                f"   COL: {result.col_url}\n\n"
            )
        
        if len(results) > 3:
            message += f"... and {len(results) - 3} more results. "
        
        message += "See artifact for complete data."
        return message
    
    def _format_no_results_message(self, query: str) -> str:
        """Generate helpful message when no results are found"""
        suggestions = [
            "Verify the spelling of scientific names",
            "Try searching at genus level (first word only)",
            "Use scientific names instead of common names"
        ]
        
        # Add context-specific suggestions
        query_lower = query.lower()
        if any(term in query_lower for term in ["tree", "plant", "flower"]):
            suggestions.insert(0, "For plants, use botanical names (e.g., 'Quercus' for oak)")
        elif any(term in query_lower for term in ["extinct", "fossil", "dinosaur"]):
            suggestions.insert(0, "For extinct species, use full scientific names")
        
        return (
            f"No results found for '{query}' in Catalogue of Life.\n\n"
            f"Suggestions:\n" + 
            "\n".join(f"- {suggestion}" for suggestion in suggestions)
        )
    
    async def _handle_search(self, context: ResponseContext, request: str, params: SearchParameters):
        """Handle species search requests"""
        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            try:
                # Convert query to scientific name if needed
                query_params = await self._convert_to_scientific_name(params.query)
                
                await process.log(f"Searching for: '{query_params.search_term}'")
                
                # Build API request
                api_url = f"{COL_API_BASE}/nameusage/search"
                api_params = {
                    "q": query_params.search_term,
                    "limit": query_params.limit,
                    "content": "SCIENTIFIC_NAME",
                    "type": "TAXON"
                }
                
                # Make API request
                data = await self._make_api_request(api_url, api_params)
                
                # Process results
                results = data.get("result", [])
                total = data.get("total", 0)
                
                await process.log(f"Found {total} results, processing top {len(results)}")
                
                # Extract taxon information
                taxon_list = []
                for item in results[:query_params.limit]:
                    try:
                        taxon_info = self._extract_taxon_info(item, item.get("id"))
                        taxon_list.append(taxon_info)
                    except Exception as e:
                        logger.warning(f"Failed to process result: {e}")
                        continue
                
                # Format response
                response_text = self._format_search_results(taxon_list, total, query_params.search_term)
                
                # Create artifact with detailed data
                artifact_data = {
                    "search_metadata": {
                        "original_query": params.query,
                        "scientific_query": query_params.search_term,
                        "total_results": total,
                        "returned_results": len(taxon_list),
                        "api_endpoint": api_url
                    },
                    "results": [
                        {
                            "id": t.id,
                            "scientific_name": t.scientific_name,
                            "authorship": t.authorship,
                            "rank": t.rank,
                            "status": t.status,
                            "extinct": t.extinct,
                            "col_url": t.col_url,
                            "taxonomy": t.taxonomy
                        }
                        for t in taxon_list
                    ]
                }
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Search results for '{query_params.search_term}'",
                    content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                    metadata={
                        "source": "Catalogue of Life",
                        "query": query_params.search_term,
                        "result_count": len(taxon_list)
                    }
                )
                
                await context.reply(response_text)
                
            except APIError as e:
                await process.log(f"API error: {str(e)}")
                await context.reply(f"Search failed: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in search: {e}")
                await context.reply("An unexpected error occurred during search. Please try again.")
    
    async def _handle_taxon_details(self, context: ResponseContext, request: str, params: TaxonDetailsParameters):
        """Handle requests for detailed taxon information"""
        async with context.begin_process(summary="Fetching taxon details") as process:
            try:
                taxon_id = params.taxon_id
                await process.log(f"Retrieving details for taxon ID: {taxon_id}")
                
                # Make API request
                api_url = f"{COL_API_BASE}/taxon/{taxon_id}"
                data = await self._make_api_request(api_url)
                
                # Extract taxon information
                taxon_info = self._extract_taxon_info(data, taxon_id)
                
                # Build detailed response
                response_lines = [
                    f"**{taxon_info.scientific_name}**",
                    f"Author: {taxon_info.authorship}" if taxon_info.authorship else None,
                    f"Rank: {taxon_info.rank}",
                    f"Status: {taxon_info.status}",
                    f"Extinct: Yes" if taxon_info.extinct else None,
                    "",
                    "**Taxonomic Classification:**"
                ]
                
                # Add taxonomy
                if taxon_info.taxonomy:
                    for rank in TAXONOMIC_RANKS:
                        if rank in taxon_info.taxonomy:
                            response_lines.append(f"- {rank.title()}: {taxon_info.taxonomy[rank]}")
                
                response_lines.extend([
                    "",
                    f"**Catalogue of Life Page:** {taxon_info.col_url}",
                    "",
                    "See artifact for complete data."
                ])
                
                # Filter out None values and join
                response_text = "\n".join(line for line in response_lines if line is not None)
                
                # Create artifact
                artifact_data = {
                    "taxon_details": {
                        "id": taxon_info.id,
                        "scientific_name": taxon_info.scientific_name,
                        "authorship": taxon_info.authorship,
                        "rank": taxon_info.rank,
                        "status": taxon_info.status,
                        "extinct": taxon_info.extinct,
                        "col_url": taxon_info.col_url
                    },
                    "taxonomy": taxon_info.taxonomy,
                    "raw_data": data
                }
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Taxon details for {taxon_info.scientific_name}",
                    content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                    metadata={
                        "source": "Catalogue of Life",
                        "taxon_id": taxon_id,
                        "scientific_name": taxon_info.scientific_name
                    }
                )
                
                await context.reply(response_text)
                
            except APIError as e:
                if "not found" in str(e).lower():
                    await context.reply(
                        f"Taxon ID '{params.taxon_id}' not found.\n\n"
                        f"Possible reasons:\n"
                        f"- The ID may be incorrect or outdated\n"
                        f"- This might be a synonym ID rather than an accepted taxon\n"
                        f"- The taxon may have been reclassified\n\n"
                        f"Try using the search function to find the correct ID."
                    )
                else:
                    await context.reply(f"Failed to retrieve taxon details: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in taxon details: {e}")
                await context.reply("An unexpected error occurred. Please try again.")
    
    async def _handle_get_synonyms(self, context: ResponseContext, request: str, params: GetSynonymsParameters):
        """Handle requests for taxonomic synonyms"""
        async with context.begin_process(summary="Fetching synonyms") as process:
            try:
                query = params.query
                
                # Determine if query is a taxon ID or scientific name
                is_taxon_id = len(query) < 10 and " " not in query and not query[0].isupper()
                
                if is_taxon_id:
                    taxon_id = query
                    await process.log(f"Using taxon ID: {taxon_id}")
                else:
                    # Search for the taxon ID
                    await process.log(f"Searching for taxon: {query}")
                    
                    api_url = f"{COL_API_BASE}/nameusage/search"
                    api_params = {
                        "q": query,
                        "limit": 1,
                        "content": "SCIENTIFIC_NAME",
                        "type": "TAXON"
                    }
                    
                    data = await self._make_api_request(api_url, api_params)
                    results = data.get("result", [])
                    
                    if not results:
                        await context.reply(
                            f"No taxon found for '{query}'.\n\n"
                            f"Please verify the scientific name and try again."
                        )
                        return
                    
                    taxon_id = results[0].get("id")
                
                # Try to get synonyms
                await process.log(f"Fetching synonyms for taxon ID: {taxon_id}")
                
                # First attempt: direct synonyms endpoint
                try:
                    api_url = f"{COL_API_BASE}/taxon/{taxon_id}/synonyms"
                    data = await self._make_api_request(api_url)
                    
                    # Process synonyms
                    synonyms = self._process_synonyms_response(data)
                    
                except APIError:
                    # Fallback: search for all names
                    await process.log("Direct synonyms endpoint failed, trying alternative method")
                    synonyms = await self._find_synonyms_by_search(query, process)
                
                if not synonyms:
                    await context.reply(
                        f"No synonyms found for '{query}'.\n\n"
                        f"This may be the currently accepted name with no recorded alternatives."
                    )
                    return
                
                # Format response
                response_text = f"Found {len(synonyms)} synonym(s):\n\n"
                
                for i, syn in enumerate(synonyms[:10], 1):
                    full_name = f"{syn['scientific_name']} {syn.get('authorship', '')}".strip()
                    response_text += (
                        f"{i}. **{full_name}**\n"
                        f"   Status: {syn['status']}\n\n"
                    )
                
                if len(synonyms) > 10:
                    response_text += f"... and {len(synonyms) - 10} more.\n\n"
                
                response_text += f"COL Page: {COL_WEB_BASE}/{taxon_id}\n"
                response_text += "See artifact for complete list."
                
                # Create artifact
                artifact_data = {
                    "query": query,
                    "taxon_id": taxon_id,
                    "synonym_count": len(synonyms),
                    "synonyms": synonyms,
                    "col_url": f"{COL_WEB_BASE}/{taxon_id}"
                }
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Synonyms list - {len(synonyms)} entries",
                    content=json.dumps(artifact_data, indent=2).encode('utf-8'),
                    metadata={
                        "source": "Catalogue of Life",
                        "synonym_count": len(synonyms)
                    }
                )
                
                await context.reply(response_text)
                
            except APIError as e:
                await process.log(f"API error: {str(e)}")
                await context.reply(f"Failed to retrieve synonyms: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in synonyms: {e}")
                await context.reply("An unexpected error occurred. Please try again.")
    
    def _process_synonyms_response(self, data: Any) -> List[Dict[str, Any]]:
        """Process various synonym response formats"""
        synonyms = []
        
        # Handle different response structures
        if isinstance(data, dict):
            items = data.get("result", data.get("heterotypic", []))
        elif isinstance(data, list):
            items = data
        else:
            return synonyms
        
        for item in items:
            if isinstance(item, dict):
                name_obj = item.get("name", item)
                synonyms.append({
                    "scientific_name": name_obj.get("scientificName", "Unknown"),
                    "authorship": name_obj.get("authorship", ""),
                    "status": item.get("status", "synonym")
                })
        
        return synonyms
    
    async def _find_synonyms_by_search(self, query: str, process: IChatBioAgentProcess) -> List[Dict[str, Any]]:
        """Alternative method to find synonyms through search"""
        api_url = f"{COL_API_BASE}/nameusage/search"
        api_params = {
            "q": query,
            "limit": 50,
            "content": "SCIENTIFIC_NAME"
        }
        
        try:
            data = await self._make_api_request(api_url, api_params)
            results = data.get("result", [])
            
            synonyms = []
            for item in results:
                usage = item.get("usage", {})
                status = usage.get("status", "").lower()
                
                if status in ["synonym", "ambiguous synonym", "provisional"]:
                    name_obj = usage.get("name", {})
                    synonyms.append({
                        "scientific_name": name_obj.get("scientificName", "Unknown"),
                        "authorship": name_obj.get("authorship", ""),
                        "status": usage.get("status", "Unknown")
                    })
            
            return synonyms
            
        except APIError:
            return []
    
    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Union[SearchParameters, TaxonDetailsParameters, GetSynonymsParameters]):
        """Main entry point routing requests to appropriate handlers"""
        logger.info(f"Processing request - Entrypoint: {entrypoint}, Request: {request}")
        
        handlers = {
            "search": self._handle_search,
            "get_taxon_details": self._handle_taxon_details,
            "get_synonyms": self._handle_get_synonyms
        }
        
        handler = handlers.get(entrypoint)
        if not handler:
            await context.reply(f"Unknown entrypoint: {entrypoint}")
            return
        
        await handler(context, request, params)


def validate_environment():
    """Validate required environment variables"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    return True


def test_agent():
    """Test agent initialization"""
    logger.info("Testing agent initialization...")
    
    try:
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        
        logger.info(f"Agent initialized successfully")
        logger.info(f"Agent name: {card.name}")
        logger.info(f"Endpoints: {', '.join(ep.id for ep in card.entrypoints)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Please check your .env file.")
        exit(1)
    
    # Test agent
    if not test_agent():
        logger.error("Agent test failed. Please check the error messages above.")
        exit(1)
    
    # Start server
    logger.info("Starting Catalogue of Life Agent server...")
    
    try:
        from ichatbio.server import run_agent_server
        
        agent = CatalogueOfLifeAgent()
        
        logger.info("Server configuration:")
        logger.info("- Host: 0.0.0.0")
        logger.info("- Port: 9999")
        logger.info("- Agent card: http://localhost:9999/.well-known/agent.json")
        logger.info("Press Ctrl+C to stop")
        
        run_agent_server(agent, host="0.0.0.0", port=9999)
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())