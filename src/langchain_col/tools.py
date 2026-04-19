"""
LangChain Tools for Catalogue of Life Agent
===========================================

Tools with call deduplication and direct termination.
Each tool is cached to prevent duplicate execution within a single request.
"""

from functools import wraps
from langchain_core.tools import tool

from .context import current_context
from .col_api import ColAPI
from .logic import (
    _do_search,
    _do_taxon_details,
    _do_get_synonyms,
    _do_get_vernacular_names,
    _do_classification,
    _do_taxon_children,
    _do_distribution,
    _do_references,
)
from .util import context_tool


# ============================================================================
# TOOL CALL TRACKING
# ============================================================================

_tool_call_tracker = {}


def reset_tool_tracker():
    """Reset the tool call tracker at the start of each request."""
    global _tool_call_tracker
    _tool_call_tracker = {}


def _create_call_key(tool_name: str, **kwargs) -> str:
    """Create a stable unique key for tool + arguments."""
    parts = []
    for k in sorted(kwargs.keys()):
        parts.append(f"{k}={str(kwargs[k])}")
    return f"{tool_name}:" + "|".join(parts)


def cache_tool_result(func):
    """Decorator to cache tool calls and prevent duplicates."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        call_key = _create_call_key(func.__name__, **kwargs)
        
        if call_key in _tool_call_tracker:
            return ""
        
        await func(*args, **kwargs)
        _tool_call_tracker[call_key] = True
        return ""
    
    return wrapper


# ============================================================================
# DATA TOOLS (8 total)
# ============================================================================

@context_tool
@cache_tool_result
async def search_species(query: str, limit: int = 5):
    """
    Search for species by scientific name in the Catalogue of Life.
    Returns multiple matching taxa with their taxonomic information.
    Use this when the user wants to find species, explore taxonomy, or discover taxa.
    
    Args:
        query: Scientific name to search for (e.g., "Panthera leo", "Orchidaceae", "Rattus")
        limit: Maximum number of results to return (default: 5, max: 20)
    """
    context = current_context.get()
    api = ColAPI()
    await _do_search(context, api, query, limit)


@context_tool
@cache_tool_result
async def get_taxon_details(taxon_id: str):
    """
    Get comprehensive information about a specific taxon including extinction status,
    habitat, authorship, and classification. Use when user needs complete species overview.
    This is your PRIMARY tool for most species questions.
    
    Args:
        taxon_id: Either a taxon ID (e.g., "4RM6W") or scientific name (e.g., "Rattus rattus")
    """
    context = current_context.get()
    api = ColAPI()
    await _do_taxon_details(context, api, taxon_id)


@context_tool
@cache_tool_result
async def get_synonyms(query: str):
    """
    Get all alternative scientific names (synonyms) for a taxon.
    Returns historical names and nomenclatural variants.
    Use for taxonomic history or alternative naming questions.
    
    Args:
        query: Either a taxon ID or scientific name
    """
    context = current_context.get()
    api = ColAPI()
    await _do_get_synonyms(context, api, query)


@context_tool
@cache_tool_result
async def get_vernacular_names(taxon_id: str):
    """
    Get common names in various languages for a taxon.
    Returns vernacular names used in different regions.
    Use when user asks for common names or translations.
    
    Args:
        taxon_id: Either a taxon ID or scientific name
    """
    context = current_context.get()
    api = ColAPI()
    await _do_get_vernacular_names(context, api, taxon_id)


@context_tool
@cache_tool_result
async def get_classification(taxon_id: str):
    """
    Get the complete taxonomic hierarchy (kingdom → species) for a taxon.
    Returns the parent lineage showing how this taxon fits in the tree of life.
    Use when user asks about classification, hierarchy, or "what family/order is X".
    
    Args:
        taxon_id: Either a taxon ID or scientific name
    """
    context = current_context.get()
    api = ColAPI()
    await _do_classification(context, api, taxon_id)


@context_tool
@cache_tool_result
async def get_taxon_children(taxon_id: str, limit: int = 20):
    """
    Get all immediate child taxa (e.g., species in a genus, genera in a family).
    Use when user wants to explore a taxonomic group or list species within a higher taxon.
    
    Args:
        taxon_id: Either a taxon ID or scientific name (e.g., "Panthera", "Felidae")
        limit: Maximum children to return (default: 20, max: 100)
    """
    context = current_context.get()
    api = ColAPI()
    await _do_taxon_children(context, api, taxon_id, limit)


@context_tool
@cache_tool_result
async def get_distribution(taxon_id: str):
    """
    Get geographic distribution information for a taxon.
    Returns regions and areas where the species is found.
    Use when user asks "where is X found" or about geographic range.
    
    Args:
        taxon_id: Either a taxon ID or scientific name
    """
    context = current_context.get()
    api = ColAPI()
    await _do_distribution(context, api, taxon_id)


@context_tool
@cache_tool_result
async def get_references(taxon_id: str):
    """
    Get bibliographic references and citations for a taxon.
    Returns scientific publications and sources that document this species.
    Use when user asks for citations, sources, or bibliography.
    
    Args:
        taxon_id: Either a taxon ID or scientific name
    """
    context = current_context.get()
    api = ColAPI()
    await _do_references(context, api, taxon_id)


# ============================================================================
# CONTROL TOOLS: ABORT & FINISH
# ============================================================================

@tool(return_direct=True)
async def finish(answer: str):
    """
    Call this when you have successfully completed the user's request.
    Provide a helpful summary of what was found.
    
    MANDATORY: You MUST call this tool to provide your final response.
    Tools return empty strings - this means they succeeded. Now call finish.
    
    Args:
        answer: Final response summarizing the results
    """
    context = current_context.get()
    await context.reply(answer)


@tool(return_direct=True)
async def abort(reason: str):
    """
    Call this if you cannot fulfill the user's request.
    Explain why you cannot help and suggest alternatives if possible.
    
    Args:
        reason: Clear explanation of why the request cannot be fulfilled
    """
    context = current_context.get()
    await context.reply(reason)