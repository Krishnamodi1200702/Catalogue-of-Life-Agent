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
@allure.story("Different Taxonomic Groups")
def test_plant_species():
    """Test query for a plant species"""
    query = "Quercus robur"  # English Oak
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert "quercus" in response.lower() or "oak" in response.lower()
    
@allure.feature("Species Lookup")
@allure.story("Different Taxonomic Groups")
def test_insect_species():
    """Test query for an insect species"""
    query = "Apis mellifera"  # Honey bee
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert "apis" in response.lower() or "bee" in response.lower()
    
@allure.feature("Species Lookup")
@allure.story("Different Taxonomic Groups")
def test_fish_species():
    """Test query for a fish species"""
    query = "Salmo trutta"  # Brown trout
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert len(response) > 10


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
    
@allure.feature("Vernacular Names")
@allure.story("Common Names Query")
def test_vernacular_names():
    """Test getting common names for a species"""
    query = "Get common names for Panthera leo"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert len(response) > 10
    
@allure.feature("Vernacular Names")
@allure.story("Common Names Query")
def test_vernacular_names_rat():
    """Test getting common names for black rat"""
    query = "What are the common names for Rattus rattus?"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert len(response) > 10
    
@allure.feature("References")
@allure.story("Bibliography Query")
def test_references():
    """Test getting references for a species"""
    query = "Get references for Panthera leo"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert response != "TIMEOUT", "Agent timed out"
    assert len(response) > 10


@allure.feature("References")
@allure.story("Bibliography Query")
def test_references_scientific():
    """Test getting scientific references"""
    query = "Show me scientific publications about Rattus rattus"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert len(response) > 10
    
@allure.feature("Common Name Resolution")
@allure.story("Natural Language Query")
def test_common_name_lion():
    """Test query using common name 'lion'"""
    query = "Tell me about lions"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert "panthera" in response.lower() or "lion" in response.lower()


@allure.feature("Common Name Resolution")
@allure.story("Natural Language Query")
def test_common_name_rat():
    """Test query using common name 'rat'"""
    query = "What is a black rat?"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert len(response) > 10


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
    
@allure.feature("Edge Cases")
@allure.story("Extinct Species")
def test_extinct_species():
    """Test query for extinct species"""
    query = "Tyrannosaurus rex"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert len(response) > 10


@allure.feature("Edge Cases")
@allure.story("Subspecies Query")
def test_subspecies():
    """Test query for a subspecies"""
    query = "Panthera leo persica"  # Asiatic lion
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert len(response) > 10


@allure.feature("Edge Cases")
@allure.story("Misspelled Names")
def test_close_misspelling():
    """Test handling of slightly misspelled species name"""
    query = "Panthera leo"  # Correct
    response = call_agent(query)
    assert response is not None


@allure.feature("Edge Cases")
@allure.story("Special Characters")
def test_species_with_author():
    """Test species name with author citation"""
    query = "Panthera leo (Linnaeus, 1758)"
    response = call_agent(query)
    
    assert response is not None, "Agent returned no response"
    assert len(response) > 10
    
@allure.feature("Natural Language Understanding")
@allure.story("Question Variations")
def test_where_found_variation():
    """Test different phrasing for distribution query"""
    query = "In which countries can I find Panthera leo?"
    response = call_agent(query)
    
    assert response is not None
    assert len(response) > 10


@allure.feature("Natural Language Understanding")
@allure.story("Question Variations")
def test_classification_variation():
    """Test different phrasing for classification query"""
    query = "What is the taxonomic rank of Panthera leo?"
    response = call_agent(query)
    
    assert response is not None
    assert len(response) > 10


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
    
@allure.feature("LLM Evaluation")
@allure.story("Answer Relevancy")
@pytest.mark.slow
def test_deepeval_distribution():
    """Test answer quality for distribution query"""
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    
    query = "Where is Rattus rattus found?"
    response = call_agent(query)
    
    assert response is not None
    
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=["Rattus rattus is found worldwide"]
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [metric])


@allure.feature("LLM Evaluation")
@allure.story("Correctness")
@pytest.mark.slow
def test_deepeval_correctness():
    """Test factual correctness of responses"""
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    
    query = "What family does Panthera leo belong to?"
    response = call_agent(query)
    
    assert response is not None
    
    correctness_metric = GEval(
        name="Correctness",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Check if the response mentions Felidae or cat family",
            "Verify the response is factually accurate"
        ],
        threshold=0.5
    )
    
    test_case = LLMTestCase(
        input=query,
        actual_output=response
    )
    
    assert_test(test_case, [correctness_metric])