"""
Test suite for Catalogue of Life agent using pytest and Allure.
DeepEval used only for select high-value tests.
"""

import pytest
import allure
from agent_client import call_agent


# ============================================================================
# NORMAL CASES
# ============================================================================

@allure.feature("Species Lookup")
@allure.story("Scientific Name Query")
def test_panthera_leo():
    """Test query for Panthera leo (Lion)"""
    query = "Panthera leo"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    assert not response.startswith("EXCEPTION"), f"Agent exception: {response}"
    assert len(response) > 10, "Response too short"
    assert "panthera" in response.lower() or "lion" in response.lower(), "Response missing species info"


@allure.feature("Species Lookup")
@allure.story("Scientific Name Query")
def test_rattus_rattus():
    """Test query for Rattus rattus (Black rat)"""
    query = "Rattus rattus"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    assert len(response) > 10, "Response too short"
    assert "rattus" in response.lower() or "rat" in response.lower(), "Response missing species info"


# ============================================================================
# FUNCTIONAL QUERIES
# ============================================================================

@allure.feature("Distribution")
@allure.story("Geographic Range Query")
def test_distribution_query():
    """Test distribution query for Panthera leo"""
    query = "Where is Panthera leo found?"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    assert len(response) > 10, "Response too short"


@allure.feature("Classification")
@allure.story("Taxonomic Family Query")
def test_classification_query():
    """Test classification query for Panthera leo"""
    query = "What family is Panthera leo?"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    assert len(response) > 10, "Response too short"
    # Should mention Felidae or family
    assert "felidae" in response.lower() or "family" in response.lower(), "Missing classification info"


@allure.feature("Children Taxa")
@allure.story("List Species in Genus")
def test_children_query():
    """Test listing species in genus Panthera"""
    query = "List species in genus Panthera"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    assert len(response) > 10, "Response too short"


@allure.feature("Synonyms")
@allure.story("Alternative Names Query")
def test_synonyms_query():
    """Test synonyms query for Panthera leo"""
    query = "Synonyms of Panthera leo"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    assert len(response) > 10, "Response too short"


# ============================================================================
# EDGE CASES
# ============================================================================

@allure.feature("Edge Cases")
@allure.story("Invalid Species Name")
def test_invalid_species():
    """Test handling of invalid species name"""
    query = "asdasdasd"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    # Should indicate not found or unable to help
    assert any(word in response.lower() for word in ["not found", "unable", "cannot", "no"]), \
        "Response doesn't indicate species not found"


@allure.feature("Edge Cases")
@allure.story("Very Long Query")
def test_long_query():
    """Test handling of very long descriptive query"""
    query = (
        "I am looking for comprehensive information about a large predatory cat species "
        "that is commonly known as the lion and lives primarily in Africa"
    )
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert not response.startswith("ERROR"), f"Agent error: {response}"
    assert len(response) > 10, "Response too short"


# ============================================================================
# STABILITY TESTS
# ============================================================================

@allure.feature("Stability")
@allure.story("Repeated Query Consistency")
def test_consistency():
    """Test that agent gives consistent non-empty responses"""
    query = "Panthera leo"
    
    response1 = call_agent(query)
    response2 = call_agent(query)
    
    assert response1 is not None and len(response1) > 10, "First call failed"
    assert response2 is not None and len(response2) > 10, "Second call failed"
    assert response1 != "TIMEOUT" and response2 != "TIMEOUT", "Timeout occurred"


@allure.feature("Stability")
@allure.story("Multiple Sequential Queries")
def test_multiple_queries():
    """Test that agent handles multiple sequential queries"""
    queries = [
        "Panthera leo",
        "Rattus rattus",
        "Where is Panthera leo found?"
    ]
    
    for i, query in enumerate(queries):
        response = call_agent(query)
        assert response is not None, f"Query {i+1} returned no response: {query}"
        assert response != "TIMEOUT", f"Query {i+1} timed out: {query}"
        assert not response.startswith("ERROR"), f"Query {i+1} error: {response}"
        assert len(response) > 10, f"Query {i+1} response too short: {query}"


# ============================================================================
# DEEPEVAL TESTS (LIMITED TO HIGH-VALUE CASES)
# ============================================================================

@allure.feature("LLM Evaluation")
@allure.story("Answer Relevancy")
@pytest.mark.slow
def test_deepeval_relevancy():
    """Test answer relevancy using DeepEval (slow - runs LLM eval)"""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    
    query = "Tell me about Panthera leo"
    response = call_agent(query)
    
    assert response is not None and response != "TIMEOUT", "Agent failed"
    
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=["Panthera leo is a species in the family Felidae"]
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [metric])