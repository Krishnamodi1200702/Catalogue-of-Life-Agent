"""
Catalogue of Life Agent for iChatBio Platform
==============================================

API Documentation: https://api.checklistbank.org
Dataset: Catalogue of Life Latest Release (3LR)

Author: Krishna Modi
Version: 2.3.0
License: MIT

Changelog:
    2.3.0 - Three-stage search: SCIENTIFIC_NAME first, then broad, then VERNACULAR
          - Scoring algorithm for result ranking (exact match + rank + status)
          - Synonym resolution layer to prevent 404s on /taxon/ endpoint
          - Kept all artifacts, logging, and entrypoint descriptions from 2.2.0
    2.2.0 - First attempt at tautonym fix (incomplete)
    2.1.0 - Added get_classification and get_taxon_children entrypoints
    2.0.0 - Initial release
"""

import os
import json
import logging
from typing import Optional, Union, Tuple, List, Literal
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

# Fetch enough results to find exact matches even when API ranking is poor.
# For tautonyms like "Rattus rattus", the species can rank below the genus.
INTERNAL_FETCH_LIMIT = 50

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Entrypoint Parameter Models ---

class SearchParameters(BaseModel):
    """Parameters for searching species by scientific name or common name."""
    query: str = Field(
        description=(
            "Scientific name or common name to search. "
            "Examples: 'Panthera leo', 'Homo sapiens', 'Felidae', 'lion', 'black rat'."
        ),
        examples=["Panthera leo", "lion", "Homo sapiens", "black rat", "Quercus"]
    )
    limit: Optional[int] = Field(default=DEFAULT_SEARCH_LIMIT, description="Max results to return", ge=1, le=MAX_SEARCH_RESULTS)


class TaxonDetailsParameters(BaseModel):
    """Parameters for retrieving detailed taxonomic information."""
    taxon_id: str = Field(
        description="COL taxon ID (e.g., '4CGXP'), scientific name (e.g., 'Panthera leo'), or common name (e.g., 'lion').",
        examples=["4CGXP", "Panthera leo", "lion", "Homo sapiens"]
    )


class GetSynonymsParameters(BaseModel):
    """Parameters for retrieving taxonomic synonyms."""
    query: str = Field(
        description="Scientific name, common name, or taxon ID. Example: 'Panthera leo', 'lion', '4CGXP'.",
        examples=["Panthera leo", "lion", "4CGXP"]
    )


class GetVernacularNamesParameters(BaseModel):
    """Parameters for retrieving common names in multiple languages."""
    taxon_id: str = Field(
        description="Taxon ID, scientific name, or common name. Example: '4CGXP', 'Panthera leo', 'lion'.",
        examples=["4CGXP", "Panthera leo", "lion"]
    )


class GetClassificationParameters(BaseModel):
    """Parameters for retrieving taxonomic classification hierarchy."""
    taxon_id: str = Field(
        description="Taxon ID, scientific name, or common name. Example: '4CGXP', 'Panthera leo', 'lion'.",
        examples=["4CGXP", "Panthera leo", "lion", "Homo sapiens"]
    )


class GetTaxonChildrenParameters(BaseModel):
    """Parameters for retrieving child taxa."""
    taxon_id: str = Field(
        description="Taxon ID, scientific name, or common name. Example: '6DBT', 'Panthera', 'Felidae'.",
        examples=["6DBT", "Panthera", "Felidae", "Carnivora"]
    )
    limit: Optional[int] = Field(default=DEFAULT_CHILDREN_LIMIT, description="Max children to return", ge=1, le=100)


class CatalogueOfLifeAgent(IChatBioAgent):
    """
    Catalogue of Life agent with three-stage search, scoring, and synonym resolution.
    """
    
    def __init__(self, dataset_key: str = COL_DATASET_KEY, timeout: int = COL_TIMEOUT):
        super().__init__()
        self.dataset_key = dataset_key
        self.timeout = timeout
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
        return AgentCard(
            name="Catalogue of Life Agent",
            description=(
                "Access the Catalogue of Life database to search for species information, "
                "taxonomic classifications, synonyms, and common names. "
                "Supports both scientific names (e.g., 'Panthera leo') and "
                "common names (e.g., 'lion')."
            ),
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="search",
                    description=(
                        "Search for species using scientific names or common names when you need to "
                        "find taxa or get an overview. Supports vernacular names like 'lion', 'black rat', 'oak'. "
                        "Returns taxonomic information including rank and status for multiple matching results. "
                        "Use for: discovering species, exploring taxonomy, finding taxon IDs, getting quick overviews."
                    ),
                    parameters=SearchParameters
                ),
                AgentEntrypoint(
                    id="get_taxon_details",
                    description=(
                        "Retrieve comprehensive details for a specific taxon including extinction status, "
                        "habitat information, authorship, and direct link to Catalogue of Life page. "
                        "Returns all available information in one call. "
                        "Use when user needs: extinction data, habitat, environments, complete overview."
                    ),
                    parameters=TaxonDetailsParameters
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description=(
                        "Get all alternative scientific names (synonyms) for a taxon. "
                        "Returns historical names and nomenclatural variants. "
                        "Use for: taxonomic history, alternative names, nomenclature research."
                    ),
                    parameters=GetSynonymsParameters
                ),
                AgentEntrypoint(
                    id="get_vernacular_names",
                    description=(
                        "Get common names in various languages for a taxon. "
                        "Returns vernacular names used in different regions and languages. "
                        "Use for: common names, translations, regional names."
                    ),
                    parameters=GetVernacularNamesParameters
                ),
                AgentEntrypoint(
                    id="get_classification",
                    description=(
                        "Retrieve ONLY the taxonomic classification hierarchy (parent lineage) as a clean list. "
                        "Returns: genus -> family -> order -> class -> phylum -> kingdom -> domain. "
                        "Use when user specifically asks about: 'what family', 'what order', 'what class', "
                        "'classification', 'hierarchy', 'taxonomy', 'parent lineage', or 'belongs to which family/order'."
                    ),
                    parameters=GetClassificationParameters
                ),
                AgentEntrypoint(
                    id="get_taxon_children",
                    description=(
                        "Get all immediate child taxa of a given taxon (e.g., all species in a genus, "
                        "all genera in a family). "
                        "Returns names, ranks, and status for up to 100 children. "
                        "Use for: exploring taxonomic groups, listing species within a genus, discovering related taxa."
                    ),
                    parameters=GetTaxonChildrenParameters
                ),
            ]
        )

    # ==================== UTILITY METHODS ====================
    
    def _is_taxon_id(self, query: str) -> bool:
        """Check if query looks like a COL taxon ID (e.g., '4CGXP', '5K5L5')."""
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

    async def _make_api_request(self, process, url, params=None, expected_structure="dict"):
        """Make an API request with full logging and error handling."""
        full_url = f"{url}?{urlencode(params)}" if params else url
        await process.log("Executing API request", data={"endpoint": url, "parameters": params or {}, "full_url": full_url})
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            logger.info(f"API {url} -> {response.status_code}")
            if response.status_code == 404:
                await process.log("Resource not found (404)")
                return None
            if response.status_code != 200:
                await process.log(f"API error: HTTP {response.status_code}")
                return None
            data = response.json()
            if expected_structure == "dict" and not isinstance(data, dict): return None
            if expected_structure == "list" and not isinstance(data, list): return None
            return data
        except requests.Timeout:
            await process.log(f"Request timeout after {self.timeout}s")
            return None
        except requests.RequestException as e:
            await process.log(f"Network error: {str(e)}")
            return None
        except json.JSONDecodeError:
            await process.log("Failed to parse JSON response")
            return None

    async def _create_json_artifact(self, process, data, description, uris, metadata):
        """Create a JSON artifact for rich structured output."""
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
            await process.log(f"Artifact creation failed: {str(e)}")
            return False
    
    def _format_classification(self, taxonomy):
        """Format taxonomy dict into readable string."""
        if not taxonomy: return ""
        ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
        return "\n".join(f"- {r.capitalize()}: {taxonomy[r]}" for r in ranks if r in taxonomy)

    # ==================== SEARCH ENGINE (Core Fix) ====================
    
    def _score_result(self, item: dict, query: str) -> int:
        """
        Score a search result for relevance. Higher = better match.
        
        Scoring:
          +100  exact scientific name match (case-insensitive)
          +50   rank is 'species' when query is binomial (2+ words)
          +30   rank is 'genus/family/order' when query is single word
          +20   status is 'accepted' (not synonym)
        
        This fixes the tautonym bug: "Rattus rattus" species scores 170,
        while genus "Rattus" scores only 20.
        """
        usage = item.get("usage", {})
        name_obj = usage.get("name", {})
        sci_name = name_obj.get("scientificName", "").lower()
        rank = name_obj.get("rank", "").lower()
        status = usage.get("status", "").lower()
        
        query_lower = query.lower().strip()
        is_binomial = " " in query_lower
        
        score = 0
        if sci_name == query_lower:
            score += 100
        if is_binomial and rank == "species":
            score += 50
        if not is_binomial and rank in ("genus", "family", "order", "class", "phylum", "kingdom"):
            score += 30
        if status == "accepted":
            score += 20
        return score
    
    async def _search_for_taxon_id(self, process, query: str) -> Optional[Tuple[str, str]]:
        """
        Three-stage search to find the best taxon ID for any query.
        
        Stage 1: content=SCIENTIFIC_NAME (narrowest, most relevant for species)
                 This is where tautonyms like "Rattus rattus" get fixed — the species
                 will be in the scientific name results and score highest.
        Stage 2: No content filter (broad, catches partial matches)
        Stage 3: content=VERNACULAR (for common names like "lion", "black rat")
        
        Each stage: fetch up to 50 results, score them, pick the highest.
        For binomial queries: require an exact name match (score >= 100).
        For single-word queries: accept any reasonable match.
        """
        url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/search"
        is_binomial = " " in query.strip()
        
        strategies = [
            {"q": query, "content": "SCIENTIFIC_NAME", "limit": INTERNAL_FETCH_LIMIT},
            {"q": query, "limit": INTERNAL_FETCH_LIMIT},
            {"q": query, "content": "VERNACULAR", "limit": INTERNAL_FETCH_LIMIT},
        ]
        strategy_names = ["SCIENTIFIC_NAME", "BROAD", "VERNACULAR"]
        
        best_overall = None
        best_overall_score = -1
        
        for params, stage_name in zip(strategies, strategy_names):
            data = await self._make_api_request(process, url, params, expected_structure="dict")
            results = data.get("result", []) if data else []
            
            if not results:
                await process.log(f"Stage {stage_name}: 0 results")
                continue
            
            await process.log(
                f"Stage {stage_name}: {len(results)} results (total: {data.get('total', '?')})",
                data={"first_3": [r.get("usage", {}).get("name", {}).get("scientificName", "?") for r in results[:3]]}
            )
            
            # Score all results and pick the best
            scored = [(r, self._score_result(r, query)) for r in results]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_result, top_score = scored[0]
            
            top_name = top_result.get("usage", {}).get("name", {}).get("scientificName", "?")
            top_rank = top_result.get("usage", {}).get("name", {}).get("rank", "?")
            await process.log(f"Stage {stage_name} best: {top_name} [{top_rank}] (score={top_score})")
            
            # For binomial queries, require exact match (score >= 100) to avoid
            # returning genus "Rattus" when user asked for species "Rattus rattus"
            if is_binomial and top_score >= 100:
                taxon_id = top_result.get("id")
                found_name = top_result.get("usage", {}).get("name", {}).get("scientificName", "")
                await process.log(f"Exact binomial match found: {found_name} (ID: {taxon_id}, score={top_score})")
                return (taxon_id, found_name)
            
            # For single-word queries, accept any match from stage 1 or 2
            if not is_binomial and stage_name in ("SCIENTIFIC_NAME", "BROAD") and top_score > 0:
                taxon_id = top_result.get("id")
                found_name = top_result.get("usage", {}).get("name", {}).get("scientificName", "")
                await process.log(f"Match found: {found_name} (ID: {taxon_id}, score={top_score})")
                return (taxon_id, found_name)
            
            # For vernacular stage, accept if we have results (common name resolved)
            if stage_name == "VERNACULAR" and results:
                # Pick accepted over synonym
                for r in results:
                    if r.get("usage", {}).get("status", "").lower() == "accepted":
                        taxon_id = r.get("id")
                        found_name = r.get("usage", {}).get("name", {}).get("scientificName", "")
                        await process.log(f"Vernacular match: '{query}' -> {found_name} (ID: {taxon_id})")
                        return (taxon_id, found_name)
                # Fallback to first vernacular result
                taxon_id = results[0].get("id")
                found_name = results[0].get("usage", {}).get("name", {}).get("scientificName", "")
                await process.log(f"Vernacular fallback: '{query}' -> {found_name} (ID: {taxon_id})")
                return (taxon_id, found_name)
            
            # Track best overall as fallback
            if top_score > best_overall_score:
                best_overall = top_result
                best_overall_score = top_score
        
        # Last resort: return best we found across all stages
        if best_overall:
            taxon_id = best_overall.get("id")
            found_name = best_overall.get("usage", {}).get("name", {}).get("scientificName", "")
            await process.log(f"Fallback match: {found_name} (ID: {taxon_id}, score={best_overall_score})")
            return (taxon_id, found_name)
        
        await process.log(f"No match found for '{query}' across all search stages")
        return None
    
    async def _resolve_to_accepted_taxon(self, process, taxon_id: str) -> Optional[str]:
        """
        If /taxon/{id} returns 404, the ID might be a synonym.
        Check /nameusage/{id} to find the accepted taxon ID.
        
        Returns the accepted taxon_id, or None if resolution fails.
        """
        url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/{taxon_id}"
        data = await self._make_api_request(process, url)
        
        if not data:
            return None
        
        # Check if this is a synonym with a pointer to accepted usage
        status = data.get("status", "").lower()
        if status == "synonym":
            # The accepted usage might be in different fields depending on API version
            accepted = data.get("accepted", {})
            accepted_id = accepted.get("id") if isinstance(accepted, dict) else data.get("acceptedUsageId")
            
            if accepted_id:
                await process.log(f"Resolved synonym {taxon_id} -> accepted ID {accepted_id}")
                return accepted_id
        
        return None

    # ==================== ENTRYPOINT HANDLERS ====================
    
    async def _handle_search(self, context, request, params):
        """Search entrypoint — returns multiple results with scoring."""
        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            search_term = params.query.strip()
            limit = params.limit or DEFAULT_SEARCH_LIMIT
            await process.log(f"Search for: '{search_term}', limit={limit}")
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/search"
            
            # Three-stage search for the search entrypoint too
            results = []
            total = 0
            resolved_from_common = False
            resolved_scientific_name = None
            
            strategies = [
                ("SCIENTIFIC_NAME", {"q": search_term, "content": "SCIENTIFIC_NAME", "limit": INTERNAL_FETCH_LIMIT}),
                ("BROAD", {"q": search_term, "limit": INTERNAL_FETCH_LIMIT}),
                ("VERNACULAR", {"q": search_term, "content": "VERNACULAR", "limit": INTERNAL_FETCH_LIMIT}),
            ]
            
            for stage_name, api_params in strategies:
                data = await self._make_api_request(process, url, api_params)
                stage_results = data.get("result", []) if data else []
                total = data.get("total", 0) if data else 0
                
                await process.log(f"Search stage {stage_name}: {len(stage_results)} results (total: {total})")
                
                if stage_results:
                    results = stage_results
                    
                    if stage_name == "VERNACULAR":
                        resolved_from_common = True
                        first_sci = results[0].get("usage", {}).get("name", {}).get("scientificName", "")
                        resolved_scientific_name = first_sci
                        await process.log(f"Vernacular resolved: '{search_term}' -> {first_sci}")
                    break  # Use first stage that returns results
            
            if not results:
                await context.reply(
                    f"No species found for '{search_term}' in the Catalogue of Life.\n\n"
                    "Suggestions:\n"
                    "- Check the spelling of the name\n"
                    "- Try using just the genus name\n"
                    "- Try the scientific name if you used a common name"
                )
                return
            
            # Score and sort results
            scored = [(r, self._score_result(r, search_term)) for r in results]
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [r for r, s in scored]
            
            # Trim to user's requested limit
            results = results[:limit]
            
            # Format results
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
                    for taxon in classification:
                        tr = taxon.get("rank", "").lower()
                        tn = taxon.get("name", "")
                        if tr in ("domain","kingdom","phylum","class","order","family","genus","species") and tn:
                            taxonomy[tr] = tn
                    
                    formatted_results.append({"id": taxon_id, "scientificName": scientific_name, "rank": rank, "status": status, "taxonomy": taxonomy})
                except Exception as e:
                    logger.warning(f"Error formatting result: {e}")
            
            if not formatted_results:
                await context.reply(f"Found results for '{search_term}' but could not parse them.")
                return
            
            await process.log(
                f"Returning {len(formatted_results)} results (scored and sorted)",
                data={"top_3": [{"name": r["scientificName"], "rank": r["rank"], "score": self._score_result(results[i], search_term)} for i, r in enumerate(formatted_results[:3])]}
            )
            
            # Build reply
            name_note = f"(Resolved from common name '{search_term}' -> *{resolved_scientific_name}*)\n\n" if resolved_from_common else ""
            top = formatted_results[0]
            
            if len(formatted_results) == 1:
                reply = f"{name_note}Found {top['scientificName']} ({top['rank']}, {top['status']}).\nTaxon ID: {top['id']}\n\nSee artifact for complete data."
            else:
                reply = f"{name_note}Found {total} matches for '{search_term}'.\nTop result: {top['scientificName']} ({top['rank']})\n\nSee artifact for complete results."
            
            artifact_data = {
                "search_info": {"query": search_term, "total_found": total, "showing": len(formatted_results), "resolved_from_common_name": resolved_from_common},
                "results": formatted_results,
                "raw_response": data
            }
            await self._create_json_artifact(process, artifact_data,
                f"COL search: '{search_term}' - {len(formatted_results)} of {total} results",
                [f"{url}?{urlencode(strategies[0][1])}"],
                {"data_source": "Catalogue of Life", "query": search_term, "total_found": total})
            
            await context.reply(reply)

    async def _handle_taxon_details(self, context, request, params):
        """Get full details for a taxon, with synonym resolution for 404s."""
        async with context.begin_process(summary="Fetching taxon details") as process:
            query = params.taxon_id.strip()
            taxon_id = None
            scientific_name = None
            
            if self._is_taxon_id(query):
                taxon_id = query
                await process.log(f"Using taxon ID directly: '{taxon_id}'")
            else:
                await process.log(f"Searching for: '{query}'")
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(
                        f"No match found for '{query}' in the Catalogue of Life.\n"
                        "Please check the spelling. You can use scientific names (e.g., 'Panthera leo') or common names (e.g., 'lion')."
                    )
                    return
                taxon_id, scientific_name = result
            
            # Fetch taxon details
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}"
            data = await self._make_api_request(process, url)
            
            # If 404, try synonym resolution
            if not data:
                await process.log(f"Taxon {taxon_id} not found, attempting synonym resolution")
                resolved_id = await self._resolve_to_accepted_taxon(process, taxon_id)
                if resolved_id:
                    taxon_id = resolved_id
                    url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}"
                    data = await self._make_api_request(process, url)
            
            if not data:
                await context.reply(f"Could not retrieve details for '{query}' (ID: {taxon_id}).\nThe taxon may not exist in the current Catalogue of Life release.")
                return
            
            name_obj = data.get("name", {})
            scientific_name = name_obj.get("scientificName", "Unknown")
            authorship = name_obj.get("authorship", "")
            rank = name_obj.get("rank", "Unknown")
            status = data.get("status", "Unknown")
            extinct = data.get("extinct", False)
            environments = data.get("environments", [])
            link = data.get("link", "")
            
            classification = data.get("classification", [])
            taxonomy = {}
            for taxon in classification:
                tr = taxon.get("rank", "").lower()
                tn = taxon.get("name", "")
                if tr in ("domain","kingdom","phylum","class","order","family","genus","species") and tn:
                    taxonomy[tr] = tn
            
            await process.log(f"Details retrieved: {scientific_name} [{rank}] extinct={extinct}")
            
            full_name = f"{scientific_name} {authorship}".strip()
            reply = f"**{full_name}**\n\n"
            reply += f"**Taxon ID:** {taxon_id}\n**Rank:** {rank}\n**Status:** {status}\n"
            if extinct:
                reply += "**Extinct:** Yes\n"
            if environments:
                reply += f"**Environments:** {', '.join(environments)}\n"
            if taxonomy:
                reply += "\n**Classification:**\n" + self._format_classification(taxonomy)
            col_page = f"https://www.checklistbank.org/dataset/{self.dataset_key}/taxon/{taxon_id}"
            reply += f"\n\n**Catalogue of Life Page:** {col_page}\n"
            if link:
                reply += f"**Original Data Source:** {link}\n"
            reply += "\nSee artifact for complete data."
            
            artifact_data = {
                "taxon_info": {"id": taxon_id, "scientific_name": scientific_name, "authorship": authorship, "rank": rank, "status": status, "extinct": extinct, "environments": environments, "link": link},
                "taxonomy": taxonomy, "raw_response": data
            }
            await self._create_json_artifact(process, artifact_data, f"Taxon details: {scientific_name}", [link] if link else [url],
                {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "scientific_name": scientific_name, "rank": rank})
            await context.reply(reply)
    
    async def _handle_get_synonyms(self, context, request, params):
        """Get alternative scientific names for a taxon."""
        async with context.begin_process(summary="Fetching synonyms") as process:
            query = params.query.strip()
            if self._is_taxon_id(query):
                taxon_id, scientific_name = query, None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No species found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/info"
            data = await self._make_api_request(process, url)
            
            # Synonym resolution if 404
            if not data:
                resolved_id = await self._resolve_to_accepted_taxon(process, taxon_id)
                if resolved_id:
                    taxon_id = resolved_id
                    url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/info"
                    data = await self._make_api_request(process, url)
            
            if not data:
                await context.reply(f"Unable to retrieve synonyms for '{query}'.")
                return
            
            syn_data = data.get("synonyms", {})
            all_syns = syn_data.get("heterotypic", []) + syn_data.get("homotypic", [])
            
            if not all_syns:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No synonyms found for {display}. This may be the only accepted name.")
                return
            
            synonyms_list = []
            for item in all_syns:
                try:
                    n = item.get("name", {})
                    synonyms_list.append({"scientificName": n.get("scientificName", "Unknown"), "authorship": n.get("authorship", ""), "rank": n.get("rank", "Unknown"), "status": item.get("status", "Unknown")})
                except: continue
            
            display = scientific_name or f"taxon ID {taxon_id}"
            reply = f"**Found {len(synonyms_list)} synonym(s) for {display}:**\n\n"
            for i, s in enumerate(synonyms_list, 1):
                full = f"{s['scientificName']} {s['authorship']}".strip()
                reply += f"{i}. *{full}*\n   - Rank: {s['rank']}, Status: {s['status']}\n\n"
            reply += "See artifact for complete data."
            
            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name, "synonym_count": len(synonyms_list), "synonyms": synonyms_list, "raw_response": data}
            await self._create_json_artifact(process, artifact_data, f"Synonyms for {scientific_name or taxon_id}: {len(synonyms_list)} total", [url],
                {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "synonym_count": len(synonyms_list)})
            await context.reply(reply)

    async def _handle_vernacular_names(self, context, request, params):
        """Get common names in various languages."""
        async with context.begin_process(summary="Fetching vernacular names") as process:
            query = params.taxon_id.strip()
            if self._is_taxon_id(query):
                taxon_id, scientific_name = query, None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/vernacular"
            data = await self._make_api_request(process, url, expected_structure="list")
            
            # Synonym resolution if 404
            if data is None:
                resolved_id = await self._resolve_to_accepted_taxon(process, taxon_id)
                if resolved_id:
                    taxon_id = resolved_id
                    url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/vernacular"
                    data = await self._make_api_request(process, url, expected_structure="list")
            
            if not data:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No common names found for {display}.")
                return
            
            names_by_lang = {}
            total = 0
            for item in data:
                name = item.get("name", "")
                lang = item.get("language", "Unknown")
                if name:
                    names_by_lang.setdefault(lang, []).append(name)
                    total += 1
            
            display = scientific_name or f"taxon ID {taxon_id}"
            reply = f"**Common names for {display}:**\n\n"
            for lang in sorted(names_by_lang):
                reply += f"**{lang}:**\n"
                for name in sorted(names_by_lang[lang]):
                    reply += f"  - {name}\n"
                reply += "\n"
            reply += f"Total: {total} names in {len(names_by_lang)} languages\n\nSee artifact for complete data."
            
            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name, "total_names": total, "names_by_language": names_by_lang, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Vernacular names for {scientific_name or taxon_id}: {total} names",
                [url], {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "total_names": total})
            await context.reply(reply)
    
    async def _handle_classification(self, context, request, params):
        """Get full parent lineage hierarchy."""
        async with context.begin_process(summary="Fetching classification") as process:
            query = params.taxon_id.strip()
            if self._is_taxon_id(query):
                taxon_id, scientific_name = query, None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/classification"
            data = await self._make_api_request(process, url, expected_structure="list")
            
            # Synonym resolution if 404
            if data is None:
                resolved_id = await self._resolve_to_accepted_taxon(process, taxon_id)
                if resolved_id:
                    taxon_id = resolved_id
                    url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/classification"
                    data = await self._make_api_request(process, url, expected_structure="list")
            
            if not data:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No classification data found for {display}.")
                return
            
            classification_list = []
            for item in data:
                try:
                    classification_list.append({"id": item.get("id", ""), "name": item.get("name", "Unknown"), "authorship": item.get("authorship", ""), "rank": item.get("rank", "Unknown")})
                except: continue
            
            display = scientific_name or f"taxon ID {taxon_id}"
            reply = f"**Classification for {display}:**\n\n"
            for item in classification_list:
                full = f"{item['name']} {item['authorship']}".strip()
                reply += f"**{item['rank'].capitalize()}:** {full}\n"
            reply += f"\n**Levels:** {len(classification_list)}\n\nSee artifact for complete data."
            
            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name, "levels": len(classification_list), "classification": classification_list, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Classification for {scientific_name or taxon_id}: {len(classification_list)} levels",
                [url], {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "levels": len(classification_list)})
            await context.reply(reply)
    
    async def _handle_taxon_children(self, context, request, params):
        """List immediate child taxa."""
        async with context.begin_process(summary="Fetching child taxa") as process:
            query = params.taxon_id.strip()
            limit = params.limit or DEFAULT_CHILDREN_LIMIT
            if self._is_taxon_id(query):
                taxon_id, scientific_name = query, None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result
            
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/tree/{taxon_id}/children"
            api_params = {"limit": limit}
            data = await self._make_api_request(process, url, api_params, expected_structure="dict")
            
            # Synonym resolution if 404
            if data is None:
                resolved_id = await self._resolve_to_accepted_taxon(process, taxon_id)
                if resolved_id:
                    taxon_id = resolved_id
                    url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/tree/{taxon_id}/children"
                    data = await self._make_api_request(process, url, api_params, expected_structure="dict")
            
            results = data.get("result", []) if data else []
            total = data.get("total", 0) if data else 0
            
            if not results:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No child taxa found for {display}. This may be a terminal node (species with no subspecies).")
                return
            
            children = []
            for item in results:
                try:
                    children.append({"id": item.get("id", ""), "name": item.get("name", "Unknown"), "authorship": item.get("authorship", ""), "rank": item.get("rank", "Unknown"), "status": item.get("status", "Unknown")})
                except: continue
            
            display = scientific_name or f"taxon ID {taxon_id}"
            reply = f"**Found {total} child taxa for {display}:**\n\n"
            if total > limit:
                reply += f"*(Showing first {limit} of {total})*\n\n"
            for i, c in enumerate(children, 1):
                full = f"{c['name']} {c['authorship']}".strip()
                reply += f"{i}. **{full}**\n   - Rank: {c['rank']}, Status: {c['status']}\n\n"
            reply += "See artifact for complete data."
            
            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name, "total_children": total, "showing": len(children), "children": children, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Children of {scientific_name or taxon_id}: {len(children)} of {total}",
                [f"{url}?{urlencode(api_params)}"],
                {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "total_children": total})
            await context.reply(reply)

    # ==================== MAIN ROUTER ====================
    
    @override
    async def run(self, context, request, entrypoint, params):
        logger.info(f"Agent invoked: entrypoint={entrypoint}")
        logger.debug(f"Request: {request}, Params: {params}")
        
        handlers = {
            "search": self._handle_search,
            "get_taxon_details": self._handle_taxon_details,
            "get_synonyms": self._handle_get_synonyms,
            "get_vernacular_names": self._handle_vernacular_names,
            "get_classification": self._handle_classification,
            "get_taxon_children": self._handle_taxon_children,
        }
        
        handler = handlers.get(entrypoint)
        if handler:
            try:
                await handler(context, request, params)
            except Exception as e:
                logger.exception(f"Error in {entrypoint}: {e}")
                await context.reply("An unexpected error occurred. Please try again.")
        else:
            await context.reply(
                f"Unknown entrypoint '{entrypoint}'.\n"
                "Valid: search, get_taxon_details, get_synonyms, get_vernacular_names, get_classification, get_taxon_children"
            )


def validate_environment():
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. LLM features unavailable.")
        return False
    return True


def run_agent_server():
    logger.info("Initializing Catalogue of Life Agent v2.3.0")
    validate_environment()
    try:
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        logger.info(f"Agent: {card.name}, Entrypoints: {[ep.id for ep in card.entrypoints]}")
    except Exception as e:
        logger.error(f"Init failed: {e}")
        raise
    try:
        from ichatbio.server import run_agent_server as start_server
        logger.info("Starting on http://0.0.0.0:9999")
        start_server(agent, host="0.0.0.0", port=9999)
    except Exception as e:
        logger.error(f"Server failed: {e}")
        raise


if __name__ == "__main__":
    run_agent_server()
