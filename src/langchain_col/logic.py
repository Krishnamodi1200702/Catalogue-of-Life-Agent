"""
Catalogue of Life Logic Functions
==================================

Reusable logic functions extracted from handler methods.
Each function performs the core logic for one entrypoint.
"""

import logging
from urllib.parse import urlencode

from ichatbio.agent_response import ResponseContext
from .col_api import ColAPI, COL_BASE_URL

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHILDREN_LIMIT = 20


async def _do_search(context: ResponseContext, api: ColAPI, query: str, limit: int):
    """Search logic extracted from _handle_search."""
    async with context.begin_process(summary="Searching Catalogue of Life") as process:
        limit = limit or DEFAULT_SEARCH_LIMIT
        search_term = query.strip()
        await process.log(f"Search for: '{search_term}', limit={limit}")

        url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/nameusage/search"
        api_params = {
            "q": search_term,
            "content": "SCIENTIFIC_NAME",
            "limit": limit,
        }

        data = await api._make_api_request(process, url, api_params)

        if not data:
            await context.reply(f"Unable to complete search for '{search_term}'. Please try again.")
            return

        results = data.get("result", [])
        total = data.get("total", 0)

        await process.log(f"Got {len(results)} results (total: {total})")

        if len(results) == 0:
            await process.log("No results found")
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

        await api._create_json_artifact(
            process,
            None,
            f"COL search results for '{search_term}' - {len(formatted_results)} of {total} results",
            [f"{url}?{urlencode(api_params)}"],
            {"data_source": "Catalogue of Life", "query": search_term, "total_found": total}
        )

        await process.log(f"Search completed: found {total} results")


async def _do_taxon_details(context: ResponseContext, api: ColAPI, taxon_id: str):
    """Taxon details logic extracted from _handle_taxon_details."""
    async with context.begin_process(summary="Fetching taxon details") as process:
        query = taxon_id.strip()

        if api._is_taxon_id(query):
            taxon_id = query
            scientific_name = None
            await process.log(f"Using taxon ID directly: '{taxon_id}'")
        else:
            result = await api._search_for_taxon_id(process, query)
            if not result:
                await context.reply(f"No match found for '{query}'. Please check the spelling.")
                return
            taxon_id, scientific_name = result

        url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/taxon/{taxon_id}"
        data = await api._make_api_request(process, url)

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
        if taxonomy: reply += "\n**Classification:**\n" + api._format_classification(taxonomy)
        col_page = f"https://www.checklistbank.org/dataset/{api.dataset_key}/taxon/{taxon_id}"
        reply += f"\n\n**Catalogue of Life Page:** {col_page}\n"
        if link: reply += f"**Original Data Source:** {link}\n"
        reply += "\nSee artifact for complete data including references and additional details."

        await api._create_json_artifact(
            process,
            None,
            f"Complete taxon details for {scientific_name}",
            [link] if link else [url],
            {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "scientific_name": scientific_name, "rank": rank}
        )

        await process.log(f"Retrieved details for {scientific_name}")


async def _do_get_synonyms(context: ResponseContext, api: ColAPI, query: str):
    """Synonyms logic extracted from _handle_get_synonyms."""
    async with context.begin_process(summary="Fetching synonyms") as process:
        query = query.strip()

        if api._is_taxon_id(query):
            taxon_id = query
            scientific_name = None
        else:
            result = await api._search_for_taxon_id(process, query)
            if not result:
                await context.reply(f"No species found for '{query}'. Please check the spelling.")
                return
            taxon_id, scientific_name = result

        url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/taxon/{taxon_id}/info"
        data = await api._make_api_request(process, url)

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

        await api._create_json_artifact(
            process,
            None,
            f"Synonyms for {scientific_name or taxon_id} - {len(synonyms_list)} total",
            [url],
            {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "synonym_count": len(synonyms_list)}
        )

        await context.reply(reply)


async def _do_get_vernacular_names(context: ResponseContext, api: ColAPI, taxon_id: str):
    """Vernacular names logic extracted from _handle_vernacular_names."""
    async with context.begin_process(summary="Fetching vernacular names") as process:
        query = taxon_id.strip()

        if api._is_taxon_id(query):
            taxon_id = query
            scientific_name = None
        else:
            result = await api._search_for_taxon_id(process, query)
            if not result:
                await context.reply(f"No match found for '{query}'. Please check the spelling.")
                return
            taxon_id, scientific_name = result

        url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/taxon/{taxon_id}/vernacular"
        data = await api._make_api_request(process, url, expected_structure="list")

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

        await api._create_json_artifact(
            process,
            None,
            f"Vernacular names for {scientific_name or taxon_id} - {total} names",
            [url],
            {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "total_names": total}
        )

        await context.reply(reply)


async def _do_classification(context: ResponseContext, api: ColAPI, taxon_id: str):
    """Classification logic extracted from _handle_classification."""
    async with context.begin_process(summary="Fetching classification hierarchy") as process:
        query = taxon_id.strip()

        if api._is_taxon_id(query):
            taxon_id = query
            scientific_name = None
        else:
            result = await api._search_for_taxon_id(process, query)
            if not result:
                await context.reply(f"No match found for '{query}'. Please check the spelling.")
                return
            taxon_id, scientific_name = result

        url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/taxon/{taxon_id}/classification"
        data = await api._make_api_request(process, url, expected_structure="list")

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

        await api._create_json_artifact(
            process,
            None,
            f"Classification for {scientific_name or taxon_id} - {len(classification_list)} levels",
            [url],
            {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "levels": len(classification_list)}
        )

        await context.reply(reply)


async def _do_taxon_children(context: ResponseContext, api: ColAPI, taxon_id: str, limit: int):
    """Taxon children logic extracted from _handle_taxon_children."""
    async with context.begin_process(summary="Fetching child taxa") as process:
        limit = limit or DEFAULT_CHILDREN_LIMIT
        query = taxon_id.strip()

        if api._is_taxon_id(query):
            taxon_id = query
            scientific_name = None
        else:
            result = await api._search_for_taxon_id(process, query)
            if not result:
                await context.reply(f"No match found for '{query}'. Please check the spelling.")
                return
            taxon_id, scientific_name = result

        url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/tree/{taxon_id}/children"
        api_params = {"limit": limit}
        data = await api._make_api_request(process, url, api_params, expected_structure="dict")

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

        await api._create_json_artifact(
            process,
            None,
            f"Children of {scientific_name or taxon_id} - {len(children_list)} of {total}",
            [f"{url}?{urlencode(api_params)}"],
            {"data_source": "Catalogue of Life", "taxon_id": taxon_id, "total_children": total}
        )

        await context.reply(reply)


async def _do_distribution(context: ResponseContext, api: ColAPI, taxon_id: str):
    """Distribution logic extracted from _handle_distribution."""
    async with context.begin_process(summary="Fetching distribution data") as process:
        query = taxon_id.strip()

        # Get taxon ID
        if api._is_taxon_id(query):
            taxon_id = query
            scientific_name = None
        else:
            result = await api._search_for_taxon_id(process, query)
            if not result:
                await context.reply(f"No match found for '{query}'. Please check the spelling.")
                return
            taxon_id, scientific_name = result

        # Fetch distribution data
        url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/taxon/{taxon_id}/distribution"
        data = await api._make_api_request(process, url, expected_structure="list")

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

        # Create short reply
        display = scientific_name or f"taxon ID {taxon_id}"
        reply = (
            f"Found distribution data for {display} across {len(areas)} region(s). "
            "See artifact for complete geographic distribution."
        )

        # Create artifact with URL only (no content)
        await api._create_json_artifact(
            process,
            None,
            f"Distribution data for {scientific_name or taxon_id} - {len(areas)} regions",
            [url],
            {
                "data_source": "Catalogue of Life",
                "taxon_id": taxon_id,
                "region_count": len(areas)
            }
        )

        await context.reply(reply)


async def _do_references(context: ResponseContext, api: ColAPI, taxon_id: str):
    """References logic extracted from _handle_references."""
    async with context.begin_process(summary="Fetching references") as process:
        query = taxon_id.strip()

        # Get taxon ID
        if api._is_taxon_id(query):
            taxon_id = query
            scientific_name = None
        else:
            result = await api._search_for_taxon_id(process, query)
            if not result:
                await context.reply(f"No match found for '{query}'. Please check the spelling.")
                return
            taxon_id, scientific_name = result

        # Fetch taxon info to get reference IDs
        info_url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/taxon/{taxon_id}/info"
        info_data = await api._make_api_request(process, info_url, expected_structure="dict")

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
            ref_url = f"{COL_BASE_URL}/dataset/{api.dataset_key}/reference/{ref_id}"
            reference_urls.append(ref_url)

            ref_data = await api._make_api_request(process, ref_url, expected_structure="dict")
            if ref_data:
                references.append(ref_data)

        if not references:
            display = scientific_name or f"taxon ID '{taxon_id}'"
            await context.reply(
                f"Found {len(reference_ids)} reference ID(s) for {display}, but could not retrieve them."
            )
            return

        # Create short reply
        display = scientific_name or f"taxon ID {taxon_id}"
        reply = (
            f"Found {len(references)} bibliographic reference(s) for {display}. "
            "See artifact for complete reference data."
        )

        # Create artifact with all reference URLs (no content)
        await api._create_json_artifact(
            process,
            None,
            f"References for {scientific_name or taxon_id} - {len(references)} citations",
            [info_url] + reference_urls,
            {
                "data_source": "Catalogue of Life",
                "taxon_id": taxon_id,
                "reference_count": len(references)
            }
        )

        await context.reply(reply)