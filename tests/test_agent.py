import pytest
from unittest.mock import Mock, patch
import json
from col_agent import CatalogueOfLifeAgent, SearchParameters
from ichatbio.agent_response import DirectResponse, ProcessBeginResponse, ProcessLogResponse, ArtifactResponse, ResponseMessage


@pytest.mark.asyncio
async def test_successful_search(context, messages):
    """Test a successful search with real API response structure"""
    # Mock the API response to avoid external dependencies
    mock_response_data = {
        "result": [
            {
                "scientificName": "Panthera leo",
                "rank": "species",
                "status": "accepted",
                "classification": [
                    {"rank": "kingdom", "name": "Animalia"},
                    {"rank": "phylum", "name": "Chordata"},
                    {"rank": "class", "name": "Mammalia"},
                    {"rank": "order", "name": "Carnivora"},
                    {"rank": "family", "name": "Felidae"},
                    {"rank": "genus", "name": "Panthera"},
                    {"rank": "species", "name": "Panthera leo"}
                ]
            }
        ],
        "total": 1,
        "limit": 5,
        "offset": 0
    }
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.url = "https://api.checklistbank.org/dataset/3LR/nameusage/search?q=lion&limit=5"
        mock_get.return_value = mock_response
        
        # Run the agent
        params = SearchParameters(query="lion")
        await CatalogueOfLifeAgent().run(context, "Find information about lions", "search", params)
    
    # Verify the message structure
    assert len(messages) > 0
    assert isinstance(messages[0], ProcessBeginResponse)
    assert messages[0].summary == "Searching Catalogue of Life"
    
    # Check that we have process logs
    log_messages = [msg for msg in messages if isinstance(msg, ProcessLogResponse)]
    assert len(log_messages) >= 4  # At least: starting, extracting, querying, processing
    
    # Check that an artifact was created
    artifact_messages = [msg for msg in messages if isinstance(msg, ArtifactResponse)]
    assert len(artifact_messages) == 1
    artifact = artifact_messages[0]
    assert artifact.mimetype == "application/json"
    assert "lion" in artifact.description.lower()
    
    # Check final response
    direct_responses = [msg for msg in messages if isinstance(msg, DirectResponse)]
    assert len(direct_responses) == 1
    final_response = direct_responses[0]
    assert "Found 1 matches" in final_response.text
    assert "Panthera leo" in final_response.text


@pytest.mark.asyncio
async def test_no_results_found(context, messages):
    """Test when no results are found"""
    mock_response_data = {
        "result": [],
        "total": 0,
        "limit": 5,
        "offset": 0
    }
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_get.return_value = mock_response
        
        params = SearchParameters(query="nonexistentspecies")
        await CatalogueOfLifeAgent().run(context, "Find nonexistentspecies", "search", params)
    
    # Should get a "no results" message
    direct_responses = [msg for msg in messages if isinstance(msg, DirectResponse)]
    assert len(direct_responses) == 1
    assert "No species found" in direct_responses[0].text


@pytest.mark.asyncio
async def test_api_error_handling(context, messages):
    """Test handling of API errors"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        params = SearchParameters(query="test")
        await CatalogueOfLifeAgent().run(context, "test query", "search", params)
    
    # Should get an error message
    direct_responses = [msg for msg in messages if isinstance(msg, DirectResponse)]
    assert len(direct_responses) == 1
    assert "API error" in direct_responses[0].text


@pytest.mark.asyncio 
async def test_invalid_entrypoint(context, messages):
    """Test invalid entrypoint handling"""
    params = SearchParameters(query="test")
    await CatalogueOfLifeAgent().run(context, "test", "invalid_endpoint", params)
    
    # Should get error about unknown entrypoint
    direct_responses = [msg for msg in messages if isinstance(msg, DirectResponse)]
    assert len(direct_responses) == 1
    assert "Unknown entrypoint" in direct_responses[0].text
    assert "Expected 'search'" in direct_responses[0].text


@pytest.mark.asyncio
async def test_network_error_handling(context, messages):
    """Test network connectivity issues"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("Connection timeout")
        
        params = SearchParameters(query="test")
        await CatalogueOfLifeAgent().run(context, "test query", "search", params)
    
    # Should handle the network error gracefully
    direct_responses = [msg for msg in messages if isinstance(msg, DirectResponse)]
    assert len(direct_responses) == 1
    assert "error occurred" in direct_responses[0].text.lower()


@pytest.mark.asyncio
async def test_gpt_extraction_fallback(context, messages):
    """Test fallback when GPT extraction fails"""
    mock_response_data = {
        "result": [
            {
                "scientificName": "Test species",
                "rank": "species", 
                "status": "accepted",
                "classification": []
            }
        ],
        "total": 1
    }
    
    with patch('requests.get') as mock_get, \
         patch.object(CatalogueOfLifeAgent, '__init__') as mock_init:
        
        # Mock the agent initialization to fail GPT setup
        mock_init.return_value = None
        agent = CatalogueOfLifeAgent.__new__(CatalogueOfLifeAgent)
        agent.instructor_client = None
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.url = "test_url"
        mock_get.return_value = mock_response
        
        params = SearchParameters(query="fallback test")
        await agent.run(context, "test", "search", params)
    
    # Should still work with fallback
    process_logs = [msg for msg in messages if isinstance(msg, ProcessLogResponse)]
    has_fallback_log = any("fallback" in msg.text.lower() for msg in process_logs)
    assert has_fallback_log


@pytest.mark.asyncio
async def test_multiple_results_formatting(context, messages):
    """Test formatting of multiple search results"""
    mock_response_data = {
        "result": [
            {
                "scientificName": "Species one",
                "rank": "species",
                "status": "accepted", 
                "classification": [
                    {"rank": "kingdom", "name": "Animalia"},
                    {"rank": "genus", "name": "Genus1"}
                ]
            },
            {
                "scientificName": "Species two", 
                "rank": "species",
                "status": "synonym",
                "classification": [
                    {"rank": "kingdom", "name": "Plantae"},
                    {"rank": "genus", "name": "Genus2"}
                ]
            }
        ],
        "total": 2
    }
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.url = "test_url"
        mock_get.return_value = mock_response
        
        params = SearchParameters(query="multiple species")
        await CatalogueOfLifeAgent().run(context, "find species", "search", params)
    
    # Check that both species appear in final response
    direct_responses = [msg for msg in messages if isinstance(msg, DirectResponse)]
    final_text = direct_responses[0].text
    assert "Species one" in final_text
    assert "Species two" in final_text
    assert "Found 2 matches" in final_text


def test_agent_card():
    """Test agent card creation"""
    agent = CatalogueOfLifeAgent()
    card = agent.get_agent_card()
    
    assert card.name == "Catalogue of Life Agent"
    assert "species" in card.description.lower()
    assert len(card.entrypoints) == 1
    assert card.entrypoints[0].id == "search"
    assert card.entrypoints[0].parameters == SearchParameters


def test_search_parameters():
    """Test SearchParameters model"""
    params = SearchParameters(query="test species")
    assert params.query == "test species"
    
    # Test that query is required
    with pytest.raises(Exception):
        SearchParameters()