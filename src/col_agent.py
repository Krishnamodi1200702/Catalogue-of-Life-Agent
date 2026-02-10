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
                content=json.dumps(data, indent=2).encode('utf-8'),
                uris=uris,
                metadata=metadata
            )
            logger.info(f"Created artifact: {description}")
            return True
        except Exception as e:
            await process.log(f"Artifact creation failed: {str(e)}")
            return False

    async def _search_for_taxon_id(self, process, scientific_name):
        """Simple search: content=SCIENTIFIC_NAME, limit=5, return first result."""
        url = f"{COL_BASE_URL}/dataset/{self.dataset_key}/nameusage/search"
        params = {
            "q": scientific_name,
            "content": "SCIENTIFIC_NAME",
            "limit": 5,
        }

        data = await self._make_api_request(process, url, params, expected_structure="dict")

        if not data:
            return None

        results = data.get("result", [])

        if len(results) == 0:
            await process.log(f"No results found for '{scientific_name}'")
            return None

        await process.log(
            f"Got {len(results)} results for '{scientific_name}'",
            data={"first_3": [r.get("usage", {}).get("name", {}).get("scientificName", "?") for r in results[:3]]}
        )

        first_result = results[0]
        taxon_id = first_result.get("id")
        usage = first_result.get("usage", {})
        name_obj = usage.get("name", {})
        found_name = name_obj.get("scientificName", "")

        await process.log(f"Using first result: {found_name} (ID: {taxon_id})")
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
            reply = f"**Found {len(synonyms_list)} synonym(s) for {display}:**\n\n"
            for i, s in enumerate(synonyms_list, 1):
                full = f"{s['scientificName']} {s['authorship']}".strip()
                reply += f"{i}. *{full}*\n   - Rank: {s['rank']}, Status: {s['status']}\n\n"
            reply += "See artifact for complete synonym data."

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
            reply = f"**Common names for {display}:**\n\n"
            for lang in sorted(names_by_lang):
                reply += f"**{lang}:**\n"
                for name in sorted(names_by_lang[lang]):
                    reply += f"  - {name}\n"
                reply += "\n"
            reply += f"Total: {total} names in {len(names_by_lang)} languages\n\nSee artifact for complete data."

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
            reply = f"**Classification for {display}:**\n\n"
            for item in classification_list:
                full = f"{item['name']} {item['authorship']}".strip()
                reply += f"**{item['rank'].capitalize()}:** {full}\n"
            reply += f"\n**Total Levels:** {len(classification_list)}\n\nSee artifact for complete data."

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
            reply = f"**Found {total} child taxa for {display}:**\n\n"
            if total > limit:
                reply += f"*(Showing first {limit} of {total})*\n\n"
            for i, c in enumerate(children_list, 1):
                full = f"{c['name']} {c['authorship']}".strip()
                reply += f"{i}. **{full}**\n   - Rank: {c['rank']}, Status: {c['status']}\n\n"
            reply += "See artifact for complete data."

            artifact_data = {"taxon_id": taxon_id, "scientific_name": scientific_name,
                "total_children": total, "showing": len(children_list), "children": children_list, "raw_response": data}
            await self._create_json_artifact(process, artifact_data,
                f"Children of {scientific_name or taxon_id} - {len(children_list)} of {total}",
                [f"{url}?{urlencode(api_params)}"],
                {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "total_children": total})

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
        }

        handler = handlers.get(entrypoint)
        if handler:
            try:
                await handler(context, request, params)
            except Exception as e:
                logger.exception(f"Error in {entrypoint}: {e}")
                await context.reply("An unexpected error occurred. Please try again.")
        else:
            await context.reply(f"Unknown entrypoint '{entrypoint}'. Valid: search, get_taxon_details, get_synonyms, get_vernacular_names, get_classification, get_taxon_children")


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
