"""
Catalogue of Life Agent for iChatBio Platform
==============================================

This agent provides access to the Catalogue of Life database through the ChecklistBank API.
It enables biodiversity researchers and educators to query species information, taxonomic
classifications, synonyms, and vernacular names.

API Documentation: https://api.checklistbank.org
Dataset: Catalogue of Life Latest Release (3LR)

Author: Krishna Modi
Version: 2.0.1 (stable rollback)
License: MIT

Changelog:
    2.0.1 - Stable rollback: all 6 entrypoints, original search behavior
          - No experimental tautonym or common name fixes
          - Known limitation: tautonyms (Rattus rattus) may return genus instead of species
          - Known limitation: common names (lion, dog) not supported yet
    2.0.0 - Initial release with search, get_taxon_details, get_synonyms, get_vernacular_names
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Entrypoint Parameter Models ---

class SearchParameters(BaseModel):
    query: str = Field(
        description="Scientific name to search. Examples: 'Panthera leo', 'Homo sapiens', 'Quercus', 'Felidae'.",
        examples=["Panthera leo", "Homo sapiens", "Quercus", "Felidae"]
    )
    limit: Optional[int] = Field(default=DEFAULT_SEARCH_LIMIT, description="Maximum results to return", ge=1, le=MAX_SEARCH_RESULTS)


class TaxonDetailsParameters(BaseModel):
    taxon_id: str = Field(
        description="COL taxon ID (e.g., '4CGXP') or scientific name (e.g., 'Panthera leo').",
        examples=["4CGXP", "Panthera leo", "Homo sapiens"]
    )


class GetSynonymsParameters(BaseModel):
    query: str = Field(
        description="Scientific name or taxon ID. Example: 'Panthera leo' or '4CGXP'.",
        examples=["Panthera leo", "4CGXP", "Homo sapiens"]
    )


class GetVernacularNamesParameters(BaseModel):
    taxon_id: str = Field(
        description="Taxon ID or scientific name. Example: '4CGXP' or 'Panthera leo'.",
        examples=["4CGXP", "Panthera leo"]
    )


class GetClassificationParameters(BaseModel):
    taxon_id: str = Field(
        description="Taxon ID or scientific name. Example: '4CGXP' or 'Panthera leo'.",
        examples=["4CGXP", "Panthera leo", "Homo sapiens"]
    )


class GetTaxonChildrenParameters(BaseModel):
    taxon_id: str = Field(
        description="Taxon ID or scientific name. Example: '6DBT' or 'Panthera' or 'Felidae'.",
        examples=["6DBT", "Panthera", "Felidae", "Carnivora"]
    )
    limit: Optional[int] = Field(default=DEFAULT_CHILDREN_LIMIT, description="Maximum children to return", ge=1, le=100)
    
class GetDistributionParameters(BaseModel):
    taxon_id: str = Field(
        description="Taxon ID or scientific name. Example: '4RM6W' or 'Rattus rattus'.",
        examples=["4RM6W", "Rattus rattus", "Panthera leo"]
    )
    
class GetReferencesParameters(BaseModel):
    taxon_id: str = Field(
        description="Taxon ID or scientific name. Example: '4RM6W' or 'Rattus rattus'.",
        examples=["4RM6W", "Rattus rattus", "Panthera leo"]
    )

# --- Agent ---

class CatalogueOfLifeAgent(IChatBioAgent):

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
                "taxonomic classifications, synonyms, and common names."
            ),
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="search",
                    description=(
                        "Search for species using scientific names when you need to find taxa or get an overview. "
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
                    id="get_distribution",
                    description=(
                        "Get geographic distribution information for a taxon. "
                        "Returns regions and areas where the species is found. "
                        "Use for: geographic range, habitat locations, distribution data, "
                        "biogeography questions like 'where is X found' or 'what regions does X live in'."
                        ),
                        parameters=GetDistributionParameters
                ),
                AgentEntrypoint(
                    id="get_references",
                    description=(
                        "Get bibliographic references and citations for a taxon. "
                        "Returns scientific publications, books, and sources that document this species. "
                        "Use for: citations, sources, bibliography, literature, "
                        "or when user asks 'what are the references for X' or 'sources for X'."
                    ),
                    parameters=GetReferencesParameters
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

    # --- Utility Methods ---

    def _is_taxon_id(self, query: str) -> bool:
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
    
    def _normalize_name(self, name: str) -> str:
        """Normalize scientific name for comparison."""
        return " ".join(name.strip().lower().split())

    def _extract_result_fields(self, item: dict) -> dict:
        """Extract relevant fields from API result item."""
        usage = item.get("usage", {})
        name_obj = usage.get("name", {})

        scientific_name = name_obj.get("scientificName", "")
        rank = name_obj.get("rank", "")
        status = usage.get("status", "")

        return {
            "id": item.get("id"),
            "scientific_name": scientific_name,
            "rank": rank.lower().strip(),
            "status": status.lower().strip(),
            "item": item,
        }
    
    def _is_exact_binomial_match(self, result: dict, query: str) -> bool:
    
    # Parse query first
       query_parts = query.strip().lower().split()
       name_parts = result["scientific_name"].strip().lower().split()
    
    # Only applies to binomial queries (two words)
       if len(query_parts) != 2:
          return False
    
    # Result must also be binomial and at species rank
       if len(name_parts) < 2 or result["rank"] != "species":
           return False
    
    # Check genus and species epithet match
       query_genus = query_parts[0]
       query_species = query_parts[1]
       result_genus = name_parts[0]
       result_species = name_parts[1]
    
       return query_genus == result_genus and query_species == result_species
   
    def _choose_best_match(self, results: list, query: str):

        query_norm = self._normalize_name(query)
        parsed = [self._extract_result_fields(r) for r in results]
        
        if not parsed:
            return None
        
        # Detect if query is binomial (two words)
        query_parts = query.strip().split()
        is_binomial = len(query_parts) == 2
        
        if is_binomial:
            query_genus = query_parts[0].lower()
            query_species = query_parts[1].lower()
            
            # PRIORITY 1: Exact genus + species + ACCEPTED status
            for r in parsed:
                sci_name = r["scientific_name"]
                name_parts = sci_name.split()
                
                if len(name_parts) >= 2:
                    result_genus = name_parts[0].lower()
                    result_species = name_parts[1].lower()
                    
                    if (result_genus == query_genus and 
                        result_species == query_species and 
                        r["rank"] == "species" and 
                        r["status"] == "accepted"):  # ← MUST be accepted!
                        return r
            
            # PRIORITY 2: Exact genus + species (synonym is okay if no accepted found)
            # But only if it has a valid accepted parent
            for r in parsed:
                sci_name = r["scientific_name"]
                name_parts = sci_name.split()
                
                if len(name_parts) >= 2:
                    result_genus = name_parts[0].lower()
                    result_species = name_parts[1].lower()
                    
                    if (result_genus == query_genus and 
                        result_species == query_species and 
                        r["rank"] == "species" and
                        r["status"] == "synonym"):
                        # For synonyms, try to get the accepted name from the result
                        item = r.get("item", {})
                        usage = item.get("usage", {})
                        accepted = usage.get("accepted")
                        
                        if accepted:
                            # Create a new result dict for the accepted name
                            accepted_name_obj = accepted.get("name", {})
                            return {
                                "id": accepted.get("id"),
                                "scientific_name": accepted_name_obj.get("scientificName", sci_name),
                                "rank": accepted_name_obj.get("rank", "species"),
                                "status": "accepted",
                                "item": item
                            }
                        # If no accepted info, use the synonym anyway
                        return r
        
        # Fallback to original matching rules for non-binomial or if no exact match
        
        # Rule 1: exact scientific name + species + accepted
        for r in parsed:
            if (
                self._normalize_name(r["scientific_name"]) == query_norm
                and r["rank"] == "species"
                and r["status"] == "accepted"
            ):
                return r

        # Rule 2: exact scientific name + species
        for r in parsed:
            if (
                self._normalize_name(r["scientific_name"]) == query_norm
                and r["rank"] == "species"
            ):
                return r

        # Rule 3: exact scientific name + accepted
        for r in parsed:
            if (
                self._normalize_name(r["scientific_name"]) == query_norm
                and r["status"] == "accepted"
            ):
                return r

        # Rule 4: exact scientific name
        for r in parsed:
            if self._normalize_name(r["scientific_name"]) == query_norm:
                return r

        # Rule 5: accepted species (any name)
        for r in parsed:
            if r["rank"] == "species" and r["status"] == "accepted":
                return r

        # Rule 6: fallback to first result
        return parsed[0]

    async def _make_api_request(self, process, url, params=None, expected_structure="dict"):
        full_url = f"{url}?{urlencode(params)}" if params else url
        await process.log("Executing API request", data={"endpoint": url, "parameters": params or {}, "full_url": full_url})
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            logger.info(f"API request to {url} returned status {response.status_code}")
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
        try:
            await process.create_artifact(
                mimetype="application/json",
                description=description,
                uris=uris,
                metadata=metadata
            )
            logger.info(f"Created artifact: {description}")
            return True
        except Exception as e:
            await process.log(f"Artifact creation failed: {str(e)}")
            return False

    async def _search_for_taxon_id(self, process, scientific_name):
        """
        Search with PREFIX type and follow synonyms to accepted names.
        """
        url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/search"
        
        query_parts = scientific_name.strip().split()
        is_binomial = len(query_parts) == 2
        
        params = {
            "q": scientific_name,
            "content": "SCIENTIFIC_NAME",
            "limit": 50
        }
        
        # Use PREFIX for binomial queries
        if is_binomial:
            params["type"] = "PREFIX"
            await process.log(f"Using PREFIX matching for binomial: '{scientific_name}'")
        
        data = await self._make_api_request(process, url, params, expected_structure="dict")
        
        if not data or not data.get("result"):
            return None
        
        results = data.get("result", [])
        total = data.get("total", 0)
        
        await process.log(f"Found {total} matches with PREFIX type")
        
        # Find best match
        best_result = self._choose_best_match(results, scientific_name)
        
        if not best_result:
            return None
        
        taxon_id = best_result["id"]
        found_name = best_result["scientific_name"]
        status = best_result["status"]
        
        # If it's a synonym, get the accepted name
        if status == "synonym":
            item = best_result.get("item", {})
            usage = item.get("usage", {})
            accepted = usage.get("accepted")
            
            if accepted:
                accepted_id = accepted.get("id")
                accepted_name_obj = accepted.get("name", {})
                accepted_name = accepted_name_obj.get("scientificName", found_name)
                
                await process.log(
                    f"Found synonym '{found_name}' (ID: {taxon_id}), "
                    f"using accepted name '{accepted_name}' (ID: {accepted_id})"
                )
                
                return (accepted_id, accepted_name)
        
        await process.log(f"Found accepted name: {found_name} (ID: {taxon_id})")
        return (taxon_id, found_name)

    def _format_classification(self, taxonomy):
        if not taxonomy:
            return ""
        ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
        return "\n".join(f"- {r.capitalize()}: {taxonomy[r]}" for r in ranks if r in taxonomy)

    # --- Entrypoint Handlers ---

    async def _handle_search(self, context, request, params):
        async with context.begin_process(summary="Searching Catalogue of Life") as process:
            search_term = params.query.strip()
            limit = params.limit or DEFAULT_SEARCH_LIMIT
            await process.log(f"Search for: '{search_term}', limit={limit}")

            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/search"
            api_params = {
                "q": search_term,
                "content": "SCIENTIFIC_NAME",
                "limit": limit,
            }

            data = await self._make_api_request(process, url, api_params)

            if not data:
                await context.reply(f"Unable to complete search for '{search_term}'. Please try again.")
                return

            results = data.get("result", [])
            total = data.get("total", 0)

            await process.log(f"Got {len(results)} results (total: {total})")

            if len(results) == 0:
                await context.reply(
                    f"No species found for '{search_term}' in the Catalogue of Life.\n\n"
                    "Suggestions:\n"
                    "- Check the spelling\n"
                    "- Try using just the genus name\n"
                    "- Try a broader taxonomic group"
                )
                return

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

                    formatted_results.append({
                        "id": taxon_id, "scientificName": scientific_name,
                        "rank": rank, "status": status, "taxonomy": taxonomy
                    })
                except Exception as e:
                    logger.warning(f"Error formatting result: {e}")

            if not formatted_results:
                await context.reply(f"Found results for '{search_term}' but could not parse them.")
                return

            top = formatted_results[0]
            if len(formatted_results) == 1:
                reply = f"Found {top['scientificName']} ({top['rank']}, {top['status']}).\nTaxon ID: {top['id']}\n\nSee artifact for complete data."
            else:
                reply = f"Found {total} matches for '{search_term}'.\nTop result: {top['scientificName']} ({top['rank']})\n\nSee artifact for complete results."

            artifact_data = {
                "search_info": {"query": search_term, "total_found": total, "showing": len(formatted_results)},
                "results": formatted_results,
                "raw_response": data
            }
            await self._create_json_artifact(process, artifact_data,
                f"COL search results for '{search_term}' - {len(formatted_results)} of {total} results",
                [f"{url}?{urlencode(api_params)}"],
                {"data_source": "Catalogue of Life", "query": search_term, "total_found": total})

            await context.reply(reply)

    async def _handle_taxon_details(self, context, request, params):
        async with context.begin_process(summary="Fetching taxon details") as process:
            query = params.taxon_id.strip()

            if self._is_taxon_id(query):
                taxon_id = query
                scientific_name = None
                await process.log(f"Using taxon ID directly: '{taxon_id}'")
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result

            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}"
            data = await self._make_api_request(process, url)

            if not data:
                await context.reply(f"Taxon ID '{taxon_id}' not found in the Catalogue of Life.\nPlease verify the ID and try again.")
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

            full_name = f"{scientific_name} {authorship}".strip()
            reply = f"**{full_name}**\n\n"
            reply += f"**Taxon ID:** {taxon_id}\n**Rank:** {rank}\n**Status:** {status}\n"
            if extinct: reply += "**Extinct:** Yes\n"
            if environments: reply += f"**Environments:** {', '.join(environments)}\n"
            if taxonomy: reply += "\n**Classification:**\n" + self._format_classification(taxonomy)
            col_page = f"https://www.checklistbank.org/dataset/{self.dataset_key}/taxon/{taxon_id}"
            reply += f"\n\n**Catalogue of Life Page:** {col_page}\n"
            if link: reply += f"**Original Data Source:** {link}\n"
            reply += "\nSee artifact for complete data including references and additional details."

            artifact_data = {
                "taxon_info": {"id": taxon_id, "scientific_name": scientific_name, "authorship": authorship,
                    "rank": rank, "status": status, "extinct": extinct, "environments": environments, "link": link},
                "taxonomy": taxonomy, "raw_response": data
            }
            await self._create_json_artifact(process, artifact_data,
                f"Complete taxon details for {scientific_name}",
                [link] if link else [url],
                {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "scientific_name": scientific_name, "rank": rank})

            await context.reply(reply)

    async def _handle_get_synonyms(self, context, request, params):
        async with context.begin_process(summary="Fetching synonyms") as process:
            query = params.query.strip()

            if self._is_taxon_id(query):
                taxon_id = query
                scientific_name = None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No species found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result

            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/info"
            data = await self._make_api_request(process, url)

            if not data:
                await context.reply(f"Unable to retrieve information for taxon ID '{taxon_id}'.")
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
                    synonyms_list.append({
                        "scientificName": n.get("scientificName", "Unknown"),
                        "authorship": n.get("authorship", ""),
                        "rank": n.get("rank", "Unknown"),
                        "status": item.get("status", "Unknown")
                    })
                except: continue

            display = scientific_name or f"taxon ID {taxon_id}"
            reply = f"Found {len(synonyms_list)} synonym(s) for {display}. See artifact for complete synonym data."

            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name,
                "synonym_count": len(synonyms_list), "synonyms": synonyms_list, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Synonyms for {scientific_name or taxon_id} - {len(synonyms_list)} total",
                [url], {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "synonym_count": len(synonyms_list)})

            await context.reply(reply)

    async def _handle_vernacular_names(self, context, request, params):
        async with context.begin_process(summary="Fetching vernacular names") as process:
            query = params.taxon_id.strip()

            if self._is_taxon_id(query):
                taxon_id = query
                scientific_name = None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result

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
            reply = f"Found {total} common name(s) for {display} in {len(names_by_lang)} language(s). See artifact for complete data."

            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name,
                "total_names": total, "names_by_language": names_by_lang, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Vernacular names for {scientific_name or taxon_id} - {total} names",
                [url], {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "total_names": total})

            await context.reply(reply)

    async def _handle_classification(self, context, request, params):
        async with context.begin_process(summary="Fetching classification hierarchy") as process:
            query = params.taxon_id.strip()

            if self._is_taxon_id(query):
                taxon_id = query
                scientific_name = None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result

            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/classification"
            data = await self._make_api_request(process, url, expected_structure="list")

            if not data:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No classification data found for {display}.")
                return

            classification_list = []
            for item in data:
                try:
                    classification_list.append({
                        "id": item.get("id", ""), "name": item.get("name", "Unknown"),
                        "authorship": item.get("authorship", ""), "rank": item.get("rank", "Unknown")
                    })
                except: continue

            display = scientific_name or f"taxon ID {taxon_id}"
            reply = f"Retrieved classification hierarchy for {display} ({len(classification_list)} levels). See artifact for complete data."

            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name,
                "levels": len(classification_list), "classification": classification_list, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Classification for {scientific_name or taxon_id} - {len(classification_list)} levels",
                [url], {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "levels": len(classification_list)})

            await context.reply(reply)

    async def _handle_taxon_children(self, context, request, params):
        async with context.begin_process(summary="Fetching child taxa") as process:
            query = params.taxon_id.strip()
            limit = params.limit or DEFAULT_CHILDREN_LIMIT

            if self._is_taxon_id(query):
                taxon_id = query
                scientific_name = None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result

            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/tree/{taxon_id}/children"
            api_params = {"limit": limit}
            data = await self._make_api_request(process, url, api_params, expected_structure="dict")

            if not data:
                await context.reply(f"Unable to retrieve children for '{query}'.")
                return

            results = data.get("result", [])
            total = data.get("total", 0)

            if not results:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No child taxa found for {display}. This may be a terminal node (species with no subspecies).")
                return

            children_list = []
            for item in results:
                try:
                    children_list.append({
                        "id": item.get("id", ""), "name": item.get("name", "Unknown"),
                        "authorship": item.get("authorship", ""), "rank": item.get("rank", "Unknown"),
                        "status": item.get("status", "Unknown")
                    })
                except: continue

            display = scientific_name or f"taxon ID {taxon_id}"
            reply = f"Found {total} child taxa for {display}."
            if total > limit:
                reply += f" Showing first {limit} in the API request."
            reply += " See artifact for complete data."

            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name,
                "total_children": total, "showing": len(children_list), "children": children_list, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Children of {scientific_name or taxon_id} - {len(children_list)} of {total}",
                [f"{url}?{urlencode(api_params)}"],
                {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "total_children": total})

            await context.reply(reply)
            
    async def _handle_distribution(self, context, request, params):
        async with context.begin_process(summary="Fetching distribution data") as process:
            query = params.taxon_id.strip()

            # Get taxon ID
            if self._is_taxon_id(query):
                taxon_id = query
                scientific_name = None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result

            # Fetch distribution data
            url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/distribution"
            data = await self._make_api_request(process, url, expected_structure="list")

            if not data:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No distribution data found for {display}.")
                return

            # Extract area names
            areas = []
            for item in data:
                try:
                    area_info = item.get("area", {})
                    area_name = area_info.get("name", "")
                    if area_name:
                        areas.append(area_name)
                except:
                    continue

            if not areas:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No distribution areas found for {display}.")
                return

            # Create short reply (manager-compliant)
            display = scientific_name or f"taxon ID {taxon_id}"
            reply = (
                f"Found distribution data for {display} across {len(areas)} region(s). "
                "See artifact for complete geographic distribution."
            )

            # Create artifact with URL only (no content)
            await self._create_json_artifact(
                process,
                None,  # ← No data passed
                f"Distribution data for {scientific_name or taxon_id} - {len(areas)} regions",
                [url],  # ← Just the API URL
                {
                    "data_source": "Catalogue of Life",
                    "taxon_id": taxon_id,
                    "region_count": len(areas)
                }
            )

            await context.reply(reply)
            
    async def _handle_references(self, context, request, params):
        async with context.begin_process(summary="Fetching references") as process:
            query = params.taxon_id.strip()

            # Get taxon ID
            if self._is_taxon_id(query):
                taxon_id = query
                scientific_name = None
            else:
                result = await self._search_for_taxon_id(process, query)
                if not result:
                    await context.reply(f"No match found for '{query}'. Please check the spelling.")
                    return
                taxon_id, scientific_name = result

            # Fetch taxon info to get reference IDs
            info_url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/taxon/{taxon_id}/info"
            info_data = await self._make_api_request(process, info_url, expected_structure="dict")

            if not info_data:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"Unable to retrieve information for {display}.")
                return

            # Extract reference IDs
            usage = info_data.get("usage", {})
            reference_ids = usage.get("referenceIds", [])

            if not reference_ids:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(f"No references found for {display}.")
                return

            await process.log(f"Found {len(reference_ids)} reference ID(s)")

            # Fetch each reference (just to count successes)
            references = []
            reference_urls = []

            for ref_id in reference_ids:
                ref_url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/reference/{ref_id}"
                reference_urls.append(ref_url)

                ref_data = await self._make_api_request(process, ref_url, expected_structure="dict")
                if ref_data:
                    references.append(ref_data)

            if not references:
                display = scientific_name or f"taxon ID '{taxon_id}'"
                await context.reply(
                    f"Found {len(reference_ids)} reference ID(s) for {display}, but could not retrieve them."
                )
                return

            # Create short reply (manager-compliant - no citations shown)
            display = scientific_name or f"taxon ID {taxon_id}"
            reply = (
                f"Found {len(references)} bibliographic reference(s) for {display}. "
                "See artifact for complete reference data."
            )

            # Create artifact with all reference URLs (no content)
            await self._create_json_artifact(
                process,
                None,  # ← No data passed
                f"References for {scientific_name or taxon_id} - {len(references)} citations",
                [info_url] + reference_urls,  # ← Info URL + all reference URLs
                {
                    "data_source": "Catalogue of Life",
                    "taxon_id": taxon_id,
                    "reference_count": len(references)
                }
            )

            await context.reply(reply)

    # --- Main Router ---

    @override
    async def run(self, context, request, entrypoint, params):
        logger.info(f"Agent invoked with entrypoint: {entrypoint}")

        handlers = {
            "search": self._handle_search,
            "get_taxon_details": self._handle_taxon_details,
            "get_synonyms": self._handle_get_synonyms,
            "get_vernacular_names": self._handle_vernacular_names,
            "get_classification": self._handle_classification,
            "get_taxon_children": self._handle_taxon_children,
            "get_distribution": self._handle_distribution,
            "get_references": self._handle_references,
        }

        handler = handlers.get(entrypoint)
        if handler:
            try:
                await handler(context, request, params)
            except Exception as e:
                logger.exception(f"Error in {entrypoint}: {e}")
                await context.reply("An unexpected error occurred. Please try again.")
        else:
            await context.reply(f"Unknown entrypoint '{entrypoint}'. Valid: search, get_taxon_details, get_synonyms, get_vernacular_names, get_classification, get_taxon_children, get_distribution, get_references")

# --- Server ---

def validate_environment():
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set.")
        return False
    return True


def run_agent_server():
    logger.info("Initializing Catalogue of Life Agent v2.0.1 (stable)")
    validate_environment()
    try:
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        logger.info(f"Agent: {card.name}")
        logger.info(f"Entrypoints: {[ep.id for ep in card.entrypoints]}")
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
