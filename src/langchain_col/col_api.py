"""
Catalogue of Life API Client
=============================

Handles all interactions with the ChecklistBank API, including:
- Name resolution (scientific name → taxon ID)
- Search and matching logic
- HTTP request handling
- Artifact creation

This module contains all helper methods extracted from the original agent.
"""

import json
import logging
import requests
from typing import Optional, Tuple, Dict, List, Any
from urllib.parse import urlencode

from ichatbio.agent_response import IChatBioAgentProcess

logger = logging.getLogger(__name__)

COL_BASE_URL = "https://api.checklistbank.org"


class ColAPI:
    """
    Client for interacting with the ChecklistBank API.
    
    Encapsulates all the helper methods and API logic from the original agent.
    """
    
    def __init__(self, dataset_key: str = "3LR", timeout: int = 10):
        self.dataset_key = dataset_key
        self.timeout = timeout
    
    # ========================================================================
    # HELPER METHODS (Preserved from original agent)
    # ========================================================================
    
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