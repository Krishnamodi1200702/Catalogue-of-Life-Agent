"""
Unit Tests for Catalogue of Life Agent v2.0.1
=============================================

Tests all 6 entrypoints against the current development code.
Uses the iChatBio SDK's InMemoryResponseChannel via conftest.py fixtures.

Fixtures (from conftest.py):
    - messages: list that collects all ResponseMessage objects
    - context: ResponseContext wired to InMemoryResponseChannel

Usage:
    pytest tests/test_agent.py -v
"""

import pytest
from unittest.mock import Mock, patch
import json

from col_agent import (
    CatalogueOfLifeAgent,
    SearchParameters,
    TaxonDetailsParameters,
    GetSynonymsParameters,
    GetVernacularNamesParameters,
    GetClassificationParameters,
    GetTaxonChildrenParameters,
)
from ichatbio.agent_response import (
    DirectResponse,
    ProcessBeginResponse,
    ProcessLogResponse,
    ArtifactResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_api_response(status_code: int, json_data):
    """Create a mock requests.Response with the given status and JSON body."""
    mock_resp = Mock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    return mock_resp


def _get_messages_by_type(messages, msg_type):
    """Filter collected messages by type."""
    return [m for m in messages if isinstance(m, msg_type)]


# ---------------------------------------------------------------------------
# Mock API response data
# ---------------------------------------------------------------------------

SEARCH_RESPONSE_SINGLE = {
    "result": [
        {
            "id": "6W3C4",
            "usage": {
                "name": {
                    "scientificName": "Panthera leo",
                    "rank": "species",
                },
                "status": "accepted",
            },
            "classification": [
                {"rank": "kingdom", "name": "Animalia"},
                {"rank": "phylum", "name": "Chordata"},
                {"rank": "class", "name": "Mammalia"},
                {"rank": "order", "name": "Carnivora"},
                {"rank": "family", "name": "Felidae"},
                {"rank": "genus", "name": "Panthera"},
                {"rank": "species", "name": "Panthera leo"},
            ],
        }
    ],
    "total": 1,
    "limit": 5,
    "offset": 0,
}

SEARCH_RESPONSE_MULTIPLE = {
    "result": [
        {
            "id": "6W3C4",
            "usage": {
                "name": {"scientificName": "Panthera leo", "rank": "species"},
                "status": "accepted",
            },
            "classification": [
                {"rank": "kingdom", "name": "Animalia"},
                {"rank": "family", "name": "Felidae"},
            ],
        },
        {
            "id": "7X2B3",
            "usage": {
                "name": {"scientificName": "Panthera leo persica", "rank": "subspecies"},
                "status": "accepted",
            },
            "classification": [
                {"rank": "kingdom", "name": "Animalia"},
                {"rank": "family", "name": "Felidae"},
            ],
        },
    ],
    "total": 2,
    "limit": 5,
    "offset": 0,
}

SEARCH_RESPONSE_EMPTY = {
    "result": [],
    "total": 0,
    "limit": 5,
    "offset": 0,
}

TAXON_DETAIL_RESPONSE = {
    "name": {
        "scientificName": "Panthera leo",
        "authorship": "(Linnaeus, 1758)",
        "rank": "species",
    },
    "status": "accepted",
    "extinct": False,
    "environments": ["terrestrial"],
    "link": "https://example.com/source",
    "classification": [
        {"rank": "kingdom", "name": "Animalia"},
        {"rank": "phylum", "name": "Chordata"},
        {"rank": "class", "name": "Mammalia"},
        {"rank": "order", "name": "Carnivora"},
        {"rank": "family", "name": "Felidae"},
        {"rank": "genus", "name": "Panthera"},
    ],
}

TAXON_INFO_WITH_SYNONYMS = {
    "synonyms": {
        "heterotypic": [
            {
                "name": {
                    "scientificName": "Felis leo",
                    "authorship": "Linnaeus, 1758",
                    "rank": "species",
                },
                "status": "synonym",
            }
        ],
        "homotypic": [
            {
                "name": {
                    "scientificName": "Leo leo",
                    "authorship": "(Linnaeus, 1758)",
                    "rank": "species",
                },
                "status": "synonym",
            }
        ],
    }
}

TAXON_INFO_NO_SYNONYMS = {
    "synonyms": {
        "heterotypic": [],
        "homotypic": [],
    }
}

VERNACULAR_NAMES_RESPONSE = [
    {"name": "Lion", "language": "eng"},
    {"name": "African Lion", "language": "eng"},
    {"name": "León", "language": "spa"},
    {"name": "Löwe", "language": "deu"},
]

CLASSIFICATION_RESPONSE = [
    {"id": "1", "name": "Animalia", "authorship": "", "rank": "kingdom"},
    {"id": "2", "name": "Chordata", "authorship": "", "rank": "phylum"},
    {"id": "3", "name": "Mammalia", "authorship": "", "rank": "class"},
    {"id": "4", "name": "Carnivora", "authorship": "", "rank": "order"},
    {"id": "5", "name": "Felidae", "authorship": "Fischer, 1817", "rank": "family"},
    {"id": "6", "name": "Panthera", "authorship": "Oken, 1816", "rank": "genus"},
]

CHILDREN_RESPONSE = {
    "result": [
        {"id": "6W3C4", "name": "Panthera leo", "authorship": "(Linnaeus, 1758)", "rank": "species", "status": "accepted"},
        {"id": "8Y4D5", "name": "Panthera tigris", "authorship": "(Linnaeus, 1758)", "rank": "species", "status": "accepted"},
        {"id": "9Z5E6", "name": "Panthera pardus", "authorship": "(Linnaeus, 1758)", "rank": "species", "status": "accepted"},
        {"id": "AW6F7", "name": "Panthera onca", "authorship": "(Linnaeus, 1758)", "rank": "species", "status": "accepted"},
    ],
    "total": 4,
    "limit": 20,
    "offset": 0,
}

CHILDREN_RESPONSE_EMPTY = {
    "result": [],
    "total": 0,
    "limit": 20,
    "offset": 0,
}


# ===========================================================================
# 1. AgentCard Tests
# ===========================================================================

class TestAgentCard:
    """Verify the agent card matches the expected structure."""

    def test_card_name_and_description(self):
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        assert card.name == "Catalogue of Life Agent"
        assert "species" in card.description.lower()

    def test_card_has_six_entrypoints(self):
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        assert len(card.entrypoints) == 6

    def test_entrypoint_ids(self):
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        ids = [ep.id for ep in card.entrypoints]
        assert ids == [
            "search",
            "get_taxon_details",
            "get_synonyms",
            "get_vernacular_names",
            "get_classification",
            "get_taxon_children",
        ]

    def test_entrypoint_parameter_models(self):
        agent = CatalogueOfLifeAgent()
        card = agent.get_agent_card()
        expected = {
            "search": SearchParameters,
            "get_taxon_details": TaxonDetailsParameters,
            "get_synonyms": GetSynonymsParameters,
            "get_vernacular_names": GetVernacularNamesParameters,
            "get_classification": GetClassificationParameters,
            "get_taxon_children": GetTaxonChildrenParameters,
        }
        for ep in card.entrypoints:
            assert ep.parameters == expected[ep.id], f"Wrong parameters for {ep.id}"


# ===========================================================================
# 2. Parameter Model Tests
# ===========================================================================

class TestParameterModels:
    """Verify parameter models validate correctly."""

    def test_search_params_required_query(self):
        with pytest.raises(Exception):
            SearchParameters()

    def test_search_params_defaults(self):
        p = SearchParameters(query="Panthera leo")
        assert p.query == "Panthera leo"
        assert p.limit == 5

    def test_search_params_custom_limit(self):
        p = SearchParameters(query="Felidae", limit=10)
        assert p.limit == 10

    def test_search_params_limit_bounds(self):
        with pytest.raises(Exception):
            SearchParameters(query="test", limit=0)
        with pytest.raises(Exception):
            SearchParameters(query="test", limit=21)

    def test_taxon_details_params_required(self):
        with pytest.raises(Exception):
            TaxonDetailsParameters()

    def test_taxon_details_params_valid(self):
        p = TaxonDetailsParameters(taxon_id="4CGXP")
        assert p.taxon_id == "4CGXP"

    def test_get_synonyms_params_required(self):
        with pytest.raises(Exception):
            GetSynonymsParameters()

    def test_vernacular_params_required(self):
        with pytest.raises(Exception):
            GetVernacularNamesParameters()

    def test_classification_params_required(self):
        with pytest.raises(Exception):
            GetClassificationParameters()

    def test_children_params_required(self):
        with pytest.raises(Exception):
            GetTaxonChildrenParameters()

    def test_children_params_defaults(self):
        p = GetTaxonChildrenParameters(taxon_id="6DBT")
        assert p.limit == 20

    def test_children_params_limit_bounds(self):
        with pytest.raises(Exception):
            GetTaxonChildrenParameters(taxon_id="6DBT", limit=0)
        with pytest.raises(Exception):
            GetTaxonChildrenParameters(taxon_id="6DBT", limit=101)


# ===========================================================================
# 3. Utility Method Tests
# ===========================================================================

class TestUtilityMethods:
    """Test internal helper methods."""

    def test_is_taxon_id_with_ids(self):
        agent = CatalogueOfLifeAgent()
        assert agent._is_taxon_id("4CGXP") is True
        assert agent._is_taxon_id("5K5L5") is True
        assert agent._is_taxon_id("RT") is True
        assert agent._is_taxon_id("N") is True  # single uppercase letter is a valid COL taxon ID

    def test_is_taxon_id_with_names(self):
        agent = CatalogueOfLifeAgent()
        assert agent._is_taxon_id("Panthera leo") is False
        assert agent._is_taxon_id("Homo sapiens") is False
        assert agent._is_taxon_id("Quercus") is False
        assert agent._is_taxon_id("Felidae") is False


# ===========================================================================
# 4. Search Entrypoint Tests
# ===========================================================================

class TestSearchEntrypoint:

    @pytest.mark.asyncio
    async def test_search_single_result(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, SEARCH_RESPONSE_SINGLE)):
            params = SearchParameters(query="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Find Panthera leo", "search", params)

        # Process begins
        assert isinstance(messages[0], ProcessBeginResponse)
        assert messages[0].summary == "Searching Catalogue of Life"

        # Artifact created
        artifacts = _get_messages_by_type(messages, ArtifactResponse)
        assert len(artifacts) == 1
        assert artifacts[0].mimetype == "application/json"

        # Reply
        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Panthera leo" in replies[0].text
        assert "6W3C4" in replies[0].text  # taxon ID in reply for single result

    @pytest.mark.asyncio
    async def test_search_multiple_results(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, SEARCH_RESPONSE_MULTIPLE)):
            params = SearchParameters(query="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Find Panthera", "search", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "2 matches" in replies[0].text
        assert "Panthera leo" in replies[0].text

    @pytest.mark.asyncio
    async def test_search_no_results(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, SEARCH_RESPONSE_EMPTY)):
            params = SearchParameters(query="Zzzznotreal")
            await CatalogueOfLifeAgent().run(context, "Find Zzzznotreal", "search", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "No species found" in replies[0].text

    @pytest.mark.asyncio
    async def test_search_api_failure(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(500, {})):
            params = SearchParameters(query="test")
            await CatalogueOfLifeAgent().run(context, "test", "search", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Unable to complete search" in replies[0].text

    @pytest.mark.asyncio
    async def test_search_network_timeout(self, context, messages):
        import requests as req
        with patch("requests.get", side_effect=req.Timeout("Connection timed out")):
            params = SearchParameters(query="test")
            await CatalogueOfLifeAgent().run(context, "test", "search", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Unable to complete search" in replies[0].text


# ===========================================================================
# 5. Taxon Details Entrypoint Tests
# ===========================================================================

class TestTaxonDetailsEntrypoint:

    @pytest.mark.asyncio
    async def test_details_by_taxon_id(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, TAXON_DETAIL_RESPONSE)):
            params = TaxonDetailsParameters(taxon_id="6W3C4")
            await CatalogueOfLifeAgent().run(context, "Details for 6W3C4", "get_taxon_details", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Panthera leo" in replies[0].text
        assert "(Linnaeus, 1758)" in replies[0].text

        artifacts = _get_messages_by_type(messages, ArtifactResponse)
        assert len(artifacts) == 1

    @pytest.mark.asyncio
    async def test_details_by_scientific_name(self, context, messages):
        """When a scientific name is given, the agent first searches for the taxon ID, then fetches details."""
        with patch("requests.get") as mock_get:
            # First call: search for taxon ID
            # Second call: fetch taxon details
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),
                _mock_api_response(200, TAXON_DETAIL_RESPONSE),
            ]
            params = TaxonDetailsParameters(taxon_id="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Details for Panthera leo", "get_taxon_details", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Panthera leo" in replies[0].text

    @pytest.mark.asyncio
    async def test_details_not_found(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(404, {})):
            params = TaxonDetailsParameters(taxon_id="XXXXX")
            await CatalogueOfLifeAgent().run(context, "Details for XXXXX", "get_taxon_details", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "not found" in replies[0].text.lower()


# ===========================================================================
# 6. Synonyms Entrypoint Tests
# ===========================================================================

class TestSynonymsEntrypoint:

    @pytest.mark.asyncio
    async def test_synonyms_found(self, context, messages):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),  # name lookup
                _mock_api_response(200, TAXON_INFO_WITH_SYNONYMS),  # taxon info
            ]
            params = GetSynonymsParameters(query="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Synonyms for Panthera leo", "get_synonyms", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "2 synonym" in replies[0].text
        assert "Felis leo" in replies[0].text
        assert "Leo leo" in replies[0].text

        artifacts = _get_messages_by_type(messages, ArtifactResponse)
        assert len(artifacts) == 1

    @pytest.mark.asyncio
    async def test_synonyms_none_found(self, context, messages):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),
                _mock_api_response(200, TAXON_INFO_NO_SYNONYMS),
            ]
            params = GetSynonymsParameters(query="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Synonyms for Panthera leo", "get_synonyms", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "No synonyms found" in replies[0].text

    @pytest.mark.asyncio
    async def test_synonyms_by_taxon_id(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, TAXON_INFO_WITH_SYNONYMS)):
            params = GetSynonymsParameters(query="6W3C4")
            await CatalogueOfLifeAgent().run(context, "Synonyms for 6W3C4", "get_synonyms", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "synonym" in replies[0].text.lower()


# ===========================================================================
# 7. Vernacular Names Entrypoint Tests
# ===========================================================================

class TestVernacularNamesEntrypoint:

    @pytest.mark.asyncio
    async def test_vernacular_names_found(self, context, messages):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),
                _mock_api_response(200, VERNACULAR_NAMES_RESPONSE),
            ]
            params = GetVernacularNamesParameters(taxon_id="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Common names for Panthera leo", "get_vernacular_names", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Lion" in replies[0].text
        assert "León" in replies[0].text
        assert "4 names" in replies[0].text
        assert "3 languages" in replies[0].text

        artifacts = _get_messages_by_type(messages, ArtifactResponse)
        assert len(artifacts) == 1

    @pytest.mark.asyncio
    async def test_vernacular_names_empty(self, context, messages):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),
                _mock_api_response(200, []),  # empty list
            ]
            params = GetVernacularNamesParameters(taxon_id="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Common names", "get_vernacular_names", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "No common names found" in replies[0].text


# ===========================================================================
# 8. Classification Entrypoint Tests
# ===========================================================================

class TestClassificationEntrypoint:

    @pytest.mark.asyncio
    async def test_classification_found(self, context, messages):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),
                _mock_api_response(200, CLASSIFICATION_RESPONSE),
            ]
            params = GetClassificationParameters(taxon_id="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Classification of Panthera leo", "get_classification", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Classification" in replies[0].text
        assert "Felidae" in replies[0].text
        assert "Carnivora" in replies[0].text

        artifacts = _get_messages_by_type(messages, ArtifactResponse)
        assert len(artifacts) == 1

    @pytest.mark.asyncio
    async def test_classification_by_taxon_id(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, CLASSIFICATION_RESPONSE)):
            params = GetClassificationParameters(taxon_id="6W3C4")
            await CatalogueOfLifeAgent().run(context, "Classification of 6W3C4", "get_classification", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Classification" in replies[0].text

    @pytest.mark.asyncio
    async def test_classification_not_found(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(404, {})):
            params = GetClassificationParameters(taxon_id="XXXXX")
            await CatalogueOfLifeAgent().run(context, "Classification of XXXXX", "get_classification", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "No classification data found" in replies[0].text


# ===========================================================================
# 9. Taxon Children Entrypoint Tests
# ===========================================================================

class TestTaxonChildrenEntrypoint:

    @pytest.mark.asyncio
    async def test_children_found(self, context, messages):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),
                _mock_api_response(200, CHILDREN_RESPONSE),
            ]
            params = GetTaxonChildrenParameters(taxon_id="Panthera")
            await CatalogueOfLifeAgent().run(context, "Species in Panthera", "get_taxon_children", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "4 child taxa" in replies[0].text
        assert "Panthera leo" in replies[0].text
        assert "Panthera tigris" in replies[0].text

        artifacts = _get_messages_by_type(messages, ArtifactResponse)
        assert len(artifacts) == 1

    @pytest.mark.asyncio
    async def test_children_by_taxon_id(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, CHILDREN_RESPONSE)):
            params = GetTaxonChildrenParameters(taxon_id="6DBT")
            await CatalogueOfLifeAgent().run(context, "Children of 6DBT", "get_taxon_children", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "child taxa" in replies[0].text.lower()

    @pytest.mark.asyncio
    async def test_children_empty(self, context, messages):
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                _mock_api_response(200, SEARCH_RESPONSE_SINGLE),
                _mock_api_response(200, CHILDREN_RESPONSE_EMPTY),
            ]
            params = GetTaxonChildrenParameters(taxon_id="Panthera leo")
            await CatalogueOfLifeAgent().run(context, "Children of Panthera leo", "get_taxon_children", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "No child taxa found" in replies[0].text

    @pytest.mark.asyncio
    async def test_children_custom_limit(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, CHILDREN_RESPONSE)):
            params = GetTaxonChildrenParameters(taxon_id="6DBT", limit=50)
            await CatalogueOfLifeAgent().run(context, "Children of 6DBT", "get_taxon_children", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1


# ===========================================================================
# 10. Cross-Cutting Tests
# ===========================================================================

class TestCrossCutting:

    @pytest.mark.asyncio
    async def test_unknown_entrypoint(self, context, messages):
        params = SearchParameters(query="test")
        await CatalogueOfLifeAgent().run(context, "test", "nonexistent_entrypoint", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "Unknown entrypoint" in replies[0].text

    @pytest.mark.asyncio
    async def test_search_begins_with_process(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, SEARCH_RESPONSE_SINGLE)):
            await CatalogueOfLifeAgent().run(context, "test", "search", SearchParameters(query="Panthera leo"))
        assert isinstance(messages[0], ProcessBeginResponse)

    @pytest.mark.asyncio
    async def test_taxon_details_begins_with_process(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, TAXON_DETAIL_RESPONSE)):
            await CatalogueOfLifeAgent().run(context, "test", "get_taxon_details", TaxonDetailsParameters(taxon_id="6W3C4"))
        assert isinstance(messages[0], ProcessBeginResponse)

    @pytest.mark.asyncio
    async def test_synonyms_begins_with_process(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, TAXON_INFO_WITH_SYNONYMS)):
            await CatalogueOfLifeAgent().run(context, "test", "get_synonyms", GetSynonymsParameters(query="6W3C4"))
        assert isinstance(messages[0], ProcessBeginResponse)

    @pytest.mark.asyncio
    async def test_vernacular_begins_with_process(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, VERNACULAR_NAMES_RESPONSE)):
            await CatalogueOfLifeAgent().run(context, "test", "get_vernacular_names", GetVernacularNamesParameters(taxon_id="6W3C4"))
        assert isinstance(messages[0], ProcessBeginResponse)

    @pytest.mark.asyncio
    async def test_classification_begins_with_process(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, CLASSIFICATION_RESPONSE)):
            await CatalogueOfLifeAgent().run(context, "test", "get_classification", GetClassificationParameters(taxon_id="6W3C4"))
        assert isinstance(messages[0], ProcessBeginResponse)

    @pytest.mark.asyncio
    async def test_children_begins_with_process(self, context, messages):
        with patch("requests.get", return_value=_mock_api_response(200, CHILDREN_RESPONSE)):
            await CatalogueOfLifeAgent().run(context, "test", "get_taxon_children", GetTaxonChildrenParameters(taxon_id="6DBT"))
        assert isinstance(messages[0], ProcessBeginResponse)

    @pytest.mark.asyncio
    async def test_search_name_not_found_then_reply(self, context, messages):
        """When a scientific name lookup fails, the agent should reply with an error."""
        with patch("requests.get", return_value=_mock_api_response(200, SEARCH_RESPONSE_EMPTY)):
            params = TaxonDetailsParameters(taxon_id="Nonexistentia impossibilis")
            await CatalogueOfLifeAgent().run(context, "Details for Nonexistentia", "get_taxon_details", params)

        replies = _get_messages_by_type(messages, DirectResponse)
        assert len(replies) == 1
        assert "No match found" in replies[0].text