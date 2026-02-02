"""
Comprehensive test suite for COL Agent - Adversarial Testing
Focus on edge cases, weird species, and scenarios likely to break the agent
No easy cases like Panthera leo or Homo sapiens
"""
import requests
import json
import time

BACKEND_URL = "http://localhost:8989/chat"

class TestResult:
    def __init__(self, name, query, expected_behavior, category):
        self.name = name
        self.query = query
        self.expected_behavior = expected_behavior
        self.category = category
        self.passed = None
        self.response = None
        self.artifact = None
        self.error = None
        self.warnings = []
        self.total_results = None
        self.execution_time = None

def run_test(test: TestResult) -> TestResult:
    """Execute a single test and analyze results"""
    start_time = time.time()
    
    try:
        response = requests.post(
            BACKEND_URL,
            json={"type": "user_text_message", "value": test.query},
            timeout=45
        )
        
        test.execution_time = time.time() - start_time
        
        if response.status_code != 200:
            test.error = f"HTTP {response.status_code}"
            test.passed = False
            return test
        
        data = response.json()
        
        # Extract all information
        for msg in data:
            if msg["type"] == "ai_text_message":
                test.response = msg["value"]
            elif msg["type"] == "ai_artifact_message":
                test.artifact = msg["value"]
                if "meta" in msg["value"]:
                    test.total_results = msg["value"]["meta"].get("total_found")
        
        # Analyze results
        test.passed = analyze_test_result(test)
            
    except Exception as e:
        test.error = str(e)
        test.passed = False
        test.execution_time = time.time() - start_time
    
    return test

def analyze_test_result(test: TestResult) -> bool:
    """Analyze test results and determine pass/fail"""
    
    # Extract query terms for validation
    query_lower = test.query.lower()
    
    # Check for common failure patterns
    if test.total_results and test.total_results > 1000000:
        test.warnings.append(f"CRITICAL: Suspiciously high result count: {test.total_results}")
        return False
    
    # Check if response mentions failure gracefully
    if test.response:
        failure_indicators = ["no results", "not found", "doesn't exist", "check spelling"]
        graceful_failure = any(indicator in test.response.lower() for indicator in failure_indicators)
        
        if "should fail gracefully" in test.expected_behavior.lower():
            return graceful_failure and test.artifact is None
        
        # Check for suspicious responses
        if test.artifact and test.total_results == 0:
            test.warnings.append("Artifact created but claims 0 results")
            return False
    
    # Success criteria
    if "should succeed" in test.expected_behavior.lower():
        if not test.artifact:
            test.warnings.append("Expected artifact but none was created")
            return False
        
        # Check if results seem relevant
        if test.total_results and test.total_results < 50:
            return True
        elif test.total_results and test.total_results > 500:
            test.warnings.append(f"Unexpectedly high results: {test.total_results}")
            return False
    
    return True

# Test Suite: Adversarial and Edge Cases

TESTS = [
    
    # Category 1: Bacterial Species (likely to fail)
    TestResult(
        "1.1 Common Bacteria",
        "Search for Escherichia coli in Catalogue of Life",
        "Should find the bacteria species, not millions of unrelated results",
        "bacteria"
    ),
    
    TestResult(
        "1.2 Pathogenic Bacteria",
        "Search for Mycobacterium tuberculosis in Catalogue of Life",
        "Should find TB bacteria specifically",
        "bacteria"
    ),
    
    TestResult(
        "1.3 Gut Bacteria",
        "Search for Lactobacillus acidophilus in Catalogue of Life",
        "Should find probiotic bacteria",
        "bacteria"
    ),
    
    TestResult(
        "1.4 Soil Bacteria",
        "Search for Streptomyces coelicolor in Catalogue of Life",
        "Should find antibiotic-producing bacteria",
        "bacteria"
    ),
    
    # Category 2: Fungi (often problematic)
    TestResult(
        "2.1 Common Yeast",
        "Search for Saccharomyces cerevisiae in Catalogue of Life",
        "Should find baker's yeast",
        "fungi"
    ),
    
    TestResult(
        "2.2 Pathogenic Fungus",
        "Search for Candida albicans in Catalogue of Life",
        "Should find pathogenic yeast",
        "fungi"
    ),
    
    TestResult(
        "2.3 Edible Mushroom",
        "Search for Agaricus bisporus in Catalogue of Life",
        "Should find common button mushroom",
        "fungi"
    ),
    
    TestResult(
        "2.4 Mold",
        "Search for Aspergillus niger in Catalogue of Life",
        "Should find black mold fungus",
        "fungi"
    ),
    
    # Category 3: Protists and Algae
    TestResult(
        "3.1 Malaria Parasite",
        "Search for Plasmodium falciparum in Catalogue of Life",
        "Should find malaria-causing protist",
        "protist"
    ),
    
    TestResult(
        "3.2 Common Algae",
        "Search for Chlamydomonas reinhardtii in Catalogue of Life",
        "Should find model green algae",
        "algae"
    ),
    
    TestResult(
        "3.3 Giant Kelp",
        "Search for Macrocystis pyrifera in Catalogue of Life",
        "Should find giant kelp species",
        "algae"
    ),
    
    # Category 4: Invertebrates
    TestResult(
        "4.1 Model Organism Nematode",
        "Search for Caenorhabditis elegans in Catalogue of Life",
        "Should find roundworm model organism",
        "invertebrate"
    ),
    
    TestResult(
        "4.2 Fruit Fly",
        "Search for Drosophila melanogaster in Catalogue of Life",
        "Should find common fruit fly",
        "invertebrate"
    ),
    
    TestResult(
        "4.3 Tardigrade",
        "Search for Hypsibius dujardini in Catalogue of Life",
        "Should find water bear species",
        "invertebrate"
    ),
    
    TestResult(
        "4.4 Jellyfish",
        "Search for Aurelia aurita in Catalogue of Life",
        "Should find moon jellyfish",
        "invertebrate"
    ),
    
    # Category 5: Plants (non-tree)
    TestResult(
        "5.1 Model Plant",
        "Search for Arabidopsis thaliana in Catalogue of Life",
        "Should find thale cress",
        "plant"
    ),
    
    TestResult(
        "5.2 Carnivorous Plant",
        "Search for Dionaea muscipula in Catalogue of Life",
        "Should find Venus flytrap",
        "plant"
    ),
    
    TestResult(
        "5.3 Crop Plant",
        "Search for Oryza sativa in Catalogue of Life",
        "Should find rice plant",
        "plant"
    ),
    
    TestResult(
        "5.4 Cactus",
        "Search for Opuntia ficus-indica in Catalogue of Life",
        "Should find prickly pear cactus",
        "plant"
    ),
    
    # Category 6: Extinct Species
    TestResult(
        "6.1 Dinosaur",
        "Search for Velociraptor mongoliensis in Catalogue of Life",
        "Should find extinct dinosaur and mark as extinct",
        "extinct"
    ),
    
    TestResult(
        "6.2 Mammoth",
        "Search for Mammuthus primigenius in Catalogue of Life",
        "Should find woolly mammoth",
        "extinct"
    ),
    
    TestResult(
        "6.3 Dodo",
        "Search for Raphus cucullatus in Catalogue of Life",
        "Should find extinct dodo bird",
        "extinct"
    ),
    
    # Category 7: Subspecies and Varieties
    TestResult(
        "7.1 Dog Subspecies",
        "Search for Canis lupus familiaris in Catalogue of Life",
        "Should find domestic dog as subspecies",
        "subspecies"
    ),
    
    TestResult(
        "7.2 Tiger Subspecies",
        "Search for Panthera tigris altaica in Catalogue of Life",
        "Should find Siberian tiger subspecies",
        "subspecies"
    ),
    
    TestResult(
        "7.3 Plant Variety",
        "Search for Solanum lycopersicum var. cerasiforme in Catalogue of Life",
        "Should handle variety notation",
        "variety"
    ),
    
    # Category 8: Names with Special Characters
    TestResult(
        "8.1 Hyphenated Name",
        "Search for Bufo bufo in Catalogue of Life",
        "Should handle simple binomial",
        "special_chars"
    ),
    
    TestResult(
        "8.2 Name with Parentheses",
        "Get taxon details for species described as (Linnaeus, 1758) from Catalogue of Life",
        "Should handle authorship in parentheses",
        "special_chars"
    ),
    
    # Category 9: Ambiguous or Partial Names
    TestResult(
        "9.1 Genus Only - Common",
        "Search for Quercus in Catalogue of Life",
        "Should return multiple oak species, not millions of unrelated results",
        "partial"
    ),
    
    TestResult(
        "9.2 Genus Only - Bacteria",
        "Search for Bacillus in Catalogue of Life",
        "Should return bacterial genus, manageable number of results",
        "partial"
    ),
    
    TestResult(
        "9.3 Single Word",
        "Search for Drosophila in Catalogue of Life",
        "Should return genus with multiple species",
        "partial"
    ),
    
    # Category 10: Invalid/Nonsense Queries
    TestResult(
        "10.1 Completely Fake",
        "Search for Dragonus maximus in Catalogue of Life",
        "Should fail gracefully with no results",
        "invalid"
    ),
    
    TestResult(
        "10.2 Misspelled Common",
        "Search for Homo sapienz in Catalogue of Life",
        "Should fail gracefully or suggest correction",
        "invalid"
    ),
    
    TestResult(
        "10.3 Common Name",
        "Search for elephant in Catalogue of Life",
        "Should fail gracefully - agent doesn't support common names",
        "invalid"
    ),
    
    TestResult(
        "10.4 Mixed Case Weird",
        "Search for pAnThErA LeO in Catalogue of Life",
        "Should handle case insensitivity",
        "invalid"
    ),
    
    TestResult(
        "10.5 Extra Spaces",
        "Search for Felis  catus in Catalogue of Life",
        "Should handle multiple spaces",
        "invalid"
    ),
    
    # Category 11: Taxon Details Edge Cases
    TestResult(
        "11.1 Fake Taxon ID",
        "Get taxon details for ZZZZZ from Catalogue of Life",
        "Should fail gracefully for invalid ID",
        "taxon_details"
    ),
    
    TestResult(
        "11.2 Very Short ID",
        "Get taxon details for A1 from Catalogue of Life",
        "Should handle or reject short IDs",
        "taxon_details"
    ),
    
    TestResult(
        "11.3 Numeric-only ID",
        "Get taxon details for 12345 from Catalogue of Life",
        "Should handle numeric IDs appropriately",
        "taxon_details"
    ),
    
    # Category 12: Synonyms Edge Cases
    TestResult(
        "12.1 Species with Many Synonyms",
        "Get synonyms for Bos taurus from Catalogue of Life",
        "Should find cattle synonyms if any",
        "synonyms"
    ),
    
    TestResult(
        "12.2 Recently Split Species",
        "Get synonyms for Giraffa camelopardalis from Catalogue of Life",
        "Should show if species has been reclassified",
        "synonyms"
    ),
    
    TestResult(
        "12.3 Bacteria Synonyms",
        "Get synonyms for Bacillus subtilis from Catalogue of Life",
        "Should find bacterial synonyms",
        "synonyms"
    ),
    
    # Category 13: Vernacular Names Edge Cases
    TestResult(
        "13.1 Common Bacteria",
        "Get vernacular names for Escherichia coli from Catalogue of Life",
        "Bacteria may not have common names - should handle gracefully",
        "vernacular"
    ),
    
    TestResult(
        "13.2 Model Organism",
        "Get vernacular names for Drosophila melanogaster from Catalogue of Life",
        "Should find 'fruit fly' in various languages",
        "vernacular"
    ),
    
    TestResult(
        "13.3 Obscure Species",
        "Get vernacular names for Hypsibius dujardini from Catalogue of Life",
        "May have no common names - should handle gracefully",
        "vernacular"
    ),
    
    # Category 14: High Volume Species
    TestResult(
        "14.1 Beetle Genus",
        "Search for Agrilus in Catalogue of Life",
        "Large genus - should return reasonable number not millions",
        "high_volume"
    ),
    
    TestResult(
        "14.2 Wasp Genus",
        "Search for Ichneumon in Catalogue of Life",
        "Should handle large insect genus",
        "high_volume"
    ),
]


def run_all_tests(interactive=False):
    """Run all tests and generate comprehensive report"""
    print("\n" + "="*80)
    print("COL AGENT ADVERSARIAL TEST SUITE")
    print("="*80)
    print(f"Total tests: {len(TESTS)}")
    print("Focus: Edge cases, problematic species, error handling")
    print("="*80 + "\n")
    
    results = []
    category_stats = {}
    
    for i, test in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] {test.name}")
        print(f"Category: {test.category}")
        print(f"Query: {test.query}")
        print("-" * 80)
        
        result = run_test(test)
        results.append(result)
        
        # Track category stats
        if result.category not in category_stats:
            category_stats[result.category] = {"passed": 0, "failed": 0, "total": 0}
        category_stats[result.category]["total"] += 1
        
        # Show immediate result
        status = "PASS" if result.passed else "FAIL"
        print(f"Status: {status} (took {result.execution_time:.2f}s)")
        
        if result.total_results:
            print(f"Results returned: {result.total_results}")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"WARNING: {warning}")
        
        if result.response:
            print(f"Response: {result.response[:120]}...")
        
        if result.error:
            print(f"ERROR: {result.error}")
        
        if result.passed:
            category_stats[result.category]["passed"] += 1
        else:
            category_stats[result.category]["failed"] += 1
        
        if interactive:
            input("\n[Press Enter to continue...]")
    
    # Generate comprehensive report
    print("\n\n" + "="*80)
    print("COMPREHENSIVE TEST REPORT")
    print("="*80 + "\n")
    
    # Overall stats
    total_passed = sum(1 for r in results if r.passed)
    total_failed = len(results) - total_passed
    pass_rate = (total_passed / len(results) * 100)
    
    print(f"Overall Results: {total_passed}/{len(results)} passed ({pass_rate:.1f}%)")
    print(f"Total Failures: {total_failed}")
    print(f"Average execution time: {sum(r.execution_time for r in results if r.execution_time) / len(results):.2f}s")
    
    # Category breakdown
    print("\n" + "-"*80)
    print("CATEGORY BREAKDOWN")
    print("-"*80)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        cat_pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"{category:20s}: {stats['passed']:2d}/{stats['total']:2d} passed ({cat_pass_rate:5.1f}%)")
    
    # Critical failures
    critical_failures = [r for r in results if not r.passed and r.warnings]
    if critical_failures:
        print("\n" + "-"*80)
        print("CRITICAL FAILURES (with warnings)")
        print("-"*80)
        for r in critical_failures:
            print(f"\n{r.name}")
            print(f"  Query: {r.query}")
            for warning in r.warnings:
                print(f"  - {warning}")
    
    # All failures
    all_failures = [r for r in results if not r.passed]
    if all_failures:
        print("\n" + "-"*80)
        print("ALL FAILED TESTS")
        print("-"*80)
        for r in all_failures:
            print(f"\n{r.name}")
            print(f"  Category: {r.category}")
            print(f"  Query: {r.query}")
            print(f"  Expected: {r.expected_behavior}")
            if r.error:
                print(f"  Error: {r.error}")
            if r.total_results:
                print(f"  Results: {r.total_results}")
    
    # Save detailed JSON report
    with open("adversarial_test_report.json", "w") as f:
        report = {
            "summary": {
                "total_tests": len(results),
                "passed": total_passed,
                "failed": total_failed,
                "pass_rate": pass_rate
            },
            "category_stats": category_stats,
            "tests": [{
                "name": r.name,
                "category": r.category,
                "query": r.query,
                "expected": r.expected_behavior,
                "passed": r.passed,
                "response": r.response,
                "total_results": r.total_results,
                "warnings": r.warnings,
                "error": r.error,
                "execution_time": r.execution_time
            } for r in results]
        }
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("Detailed JSON report saved to: adversarial_test_report.json")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    results = run_all_tests(interactive=False)