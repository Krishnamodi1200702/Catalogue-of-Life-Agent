"""
Interactive Agent Test - Catalogue of Life Agent
=================================================

Tests the agent against the LIVE ChecklistBank API with diverse species
across kingdoms, ranks, and edge cases. Each test includes expected values
so you can verify correctness — not just that it runs, but that it returns
the RIGHT data.

Usage:
    python tests/interactive_test.py --all
    python tests/interactive_test.py --test search
    python tests/interactive_test.py --entrypoint search --query "Escherichia coli"
    python tests/interactive_test.py --entrypoint get_taxon_children --query "Rosa" --limit 10
"""

import argparse
import asyncio
import json
import sys

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
    ResponseContext,
    ResponseMessage,
    DirectResponse,
    ProcessBeginResponse,
    ProcessLogResponse,
    ArtifactResponse,
)


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

class C:
    HEAD = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    R = "\033[0m"


# ---------------------------------------------------------------------------
# Minimal channel (matches conftest InMemoryResponseChannel pattern)
# ---------------------------------------------------------------------------

class TestChannel:
    """Collects messages in a list, matching InMemoryResponseChannel from conftest."""
    def __init__(self, buf: list):
        self.buf = buf

    async def submit(self, message, context_id: str = ""):
        self.buf.append(message)


# ---------------------------------------------------------------------------
# Message printer
# ---------------------------------------------------------------------------

def print_msg(msg, idx: int):
    p = f"  [{idx:2d}] "

    if isinstance(msg, ProcessBeginResponse):
        print(f"{p}{C.CYAN}{C.BOLD}[PROCESS]{C.R} {msg.summary}")

    elif isinstance(msg, ProcessLogResponse):
        print(f"{p}{C.BLUE}[LOG]{C.R} {msg.text}")
        if msg.data:
            s = json.dumps(msg.data, indent=2, default=str)
            if len(s) > 400:
                s = s[:400] + "\n       ... (truncated)"
            for line in s.split("\n"):
                print(f"         {C.DIM}{line}{C.R}")

    elif isinstance(msg, ArtifactResponse):
        print(f"{p}{C.GREEN}{C.BOLD}[ARTIFACT]{C.R} {msg.mimetype}")
        print(f"         desc: {msg.description}")
        if msg.uris:
            print(f"         uris: {msg.uris}")
        if msg.content:
            try:
                content = json.loads(msg.content.decode("utf-8"))
                preview = {k: v for k, v in content.items() if k != "raw_response"}
                s = json.dumps(preview, indent=2, default=str)
                if len(s) > 800:
                    s = s[:800] + "\n       ... (truncated)"
                print(f"         {C.DIM}content:{C.R}")
                for line in s.split("\n"):
                    print(f"           {line}")
            except Exception:
                print(f"         content: ({len(msg.content)} bytes)")

    elif isinstance(msg, DirectResponse):
        print(f"{p}{C.YELLOW}{C.BOLD}[REPLY]{C.R}")
        for line in msg.text.split("\n"):
            print(f"         {line}")

    else:
        print(f"{p}{C.RED}[?]{C.R} {type(msg).__name__}")


# ---------------------------------------------------------------------------
# Core runner with validation
# ---------------------------------------------------------------------------

async def run_test(agent, title, entrypoint, request_text, params, expect):
    """
    Run one test and validate the output.

    expect = {
        "reply_contains": ["str1", "str2"],      # strings that MUST appear in the reply
        "reply_excludes": ["str1"],               # strings that must NOT appear
        "artifact_count": 1,                      # expected number of artifacts
        "min_logs": 2,                            # minimum process log messages
        "artifact_contains": ["str1"],            # strings in artifact JSON content
    }
    """
    print(f"\n{'=' * 80}")
    print(f"{C.BOLD}{title}{C.R}")
    print(f"  entrypoint: {entrypoint}")
    print(f"  params:     {params}")
    print(f"{'=' * 80}")

    messages = []
    context = ResponseContext(TestChannel(messages), "test-context-interactive")

    try:
        await agent.run(context, request_text, entrypoint, params)
    except Exception as e:
        print(f"\n  {C.RED}EXCEPTION: {e}{C.R}")
        import traceback
        traceback.print_exc()
        return False

    # Print all messages
    print(f"\n  Messages ({len(messages)} total):\n")
    for i, msg in enumerate(messages):
        print_msg(msg, i)

    # Extract by type
    replies = [m for m in messages if isinstance(m, DirectResponse)]
    artifacts = [m for m in messages if isinstance(m, ArtifactResponse)]
    logs = [m for m in messages if isinstance(m, ProcessLogResponse)]
    begins = [m for m in messages if isinstance(m, ProcessBeginResponse)]

    # --- Validation ---
    print(f"\n  {C.DIM}--- Validation ---{C.R}")
    all_passed = True

    if begins:
        print(f"  {C.GREEN}✓{C.R} Process started: \"{begins[0].summary}\"")
    else:
        print(f"  {C.RED}✗ No ProcessBeginResponse{C.R}")
        all_passed = False

    if not replies:
        print(f"  {C.RED}✗ No reply received{C.R}")
        return False

    reply_text = replies[-1].text

    for needle in expect.get("reply_contains", []):
        if needle.lower() in reply_text.lower():
            print(f"  {C.GREEN}✓{C.R} Reply contains: \"{needle}\"")
        else:
            print(f"  {C.RED}✗ Reply missing: \"{needle}\"{C.R}")
            all_passed = False

    for needle in expect.get("reply_excludes", []):
        if needle.lower() not in reply_text.lower():
            print(f"  {C.GREEN}✓{C.R} Reply excludes: \"{needle}\"")
        else:
            print(f"  {C.RED}✗ Reply should not contain: \"{needle}\"{C.R}")
            all_passed = False

    expected_artifacts = expect.get("artifact_count", None)
    if expected_artifacts is not None:
        if len(artifacts) == expected_artifacts:
            print(f"  {C.GREEN}✓{C.R} Artifact count: {len(artifacts)}")
        else:
            print(f"  {C.RED}✗ Expected {expected_artifacts} artifacts, got {len(artifacts)}{C.R}")
            all_passed = False

    min_logs = expect.get("min_logs", 1)
    if len(logs) >= min_logs:
        print(f"  {C.GREEN}✓{C.R} Log messages: {len(logs)} (min: {min_logs})")
    else:
        print(f"  {C.RED}✗ Expected >= {min_logs} logs, got {len(logs)}{C.R}")
        all_passed = False

    for needle in expect.get("artifact_contains", []):
        found = False
        for a in artifacts:
            if a.content and needle.lower() in a.content.decode("utf-8", errors="ignore").lower():
                found = True
                break
        if found:
            print(f"  {C.GREEN}✓{C.R} Artifact contains: \"{needle}\"")
        else:
            print(f"  {C.RED}✗ Artifact missing: \"{needle}\"{C.R}")
            all_passed = False

    if all_passed:
        print(f"\n  {C.GREEN}{C.BOLD}PASSED ✓{C.R}")
    else:
        print(f"\n  {C.RED}{C.BOLD}FAILED ✗{C.R}")

    return all_passed


# ---------------------------------------------------------------------------
# 20 tests: Bacteria, Fungi, Plantae, Animalia (insects, molluscs, fish,
#           mammals, reptiles), extinct taxa, parasites, model organisms
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # ===== SEARCH (6 tests) =====
    {
        "title": "1. Search: Escherichia coli (Bacteria)",
        "entrypoint": "search",
        "request": "Search for E. coli",
        "params": SearchParameters(query="Escherichia coli"),
        "expect": {
            "reply_contains": ["Escherichia coli"],
            "artifact_count": 1,
            "artifact_contains": ["Bacteria"],
            "min_logs": 2,
        },
    },
    {
        "title": "2. Search: Amanita muscaria (Fungi — fly agaric)",
        "entrypoint": "search",
        "request": "Search for fly agaric mushroom",
        "params": SearchParameters(query="Amanita muscaria"),
        "expect": {
            "reply_contains": ["Amanita muscaria"],
            "artifact_count": 1,
            "artifact_contains": ["Fungi"],
            "min_logs": 2,
        },
    },
    {
        "title": "3. Search: Quercus robur (Plantae — English oak)",
        "entrypoint": "search",
        "request": "Search for English oak",
        "params": SearchParameters(query="Quercus robur"),
        "expect": {
            "reply_contains": ["Quercus robur"],
            "artifact_count": 1,
            "artifact_contains": ["Plantae"],
            "min_logs": 2,
        },
    },
    {
        "title": "4. Search: Octopus vulgaris (Mollusca)",
        "entrypoint": "search",
        "request": "Search for common octopus",
        "params": SearchParameters(query="Octopus vulgaris"),
        "expect": {
            "reply_contains": ["Octopus vulgaris"],
            "artifact_count": 1,
            "artifact_contains": ["Mollusca"],
            "min_logs": 2,
        },
    },
    {
        "title": "5. Search: Nonexistent species (graceful failure)",
        "entrypoint": "search",
        "request": "Search for Zzzzfakeus notarealspecies",
        "params": SearchParameters(query="Zzzzfakeus notarealspecies"),
        "expect": {
            "reply_contains": ["No species found"],
            "artifact_count": 0,
            "min_logs": 1,
        },
    },
    {
        "title": "6. Search: Orchidaceae (family-level — orchids)",
        "entrypoint": "search",
        "request": "Search for orchid family",
        "params": SearchParameters(query="Orchidaceae", limit=3),
        "expect": {
            "reply_contains": ["Orchidaceae"],
            "artifact_count": 1,
            "min_logs": 2,
        },
    },

    # ===== TAXON DETAILS (3 tests) =====
    {
        "title": "7. Details: Drosophila melanogaster (insect model organism)",
        "entrypoint": "get_taxon_details",
        "request": "Details for fruit fly",
        "params": TaxonDetailsParameters(taxon_id="Drosophila melanogaster"),
        "expect": {
            "reply_contains": ["Drosophila", "melanogaster", "species"],
            "artifact_count": 1,
            "min_logs": 2,
        },
    },
    {
        "title": "8. Details: Ginkgo biloba (living fossil plant)",
        "entrypoint": "get_taxon_details",
        "request": "Details for ginkgo tree",
        "params": TaxonDetailsParameters(taxon_id="Ginkgo biloba"),
        "expect": {
            "reply_contains": ["Ginkgo biloba", "species"],
            "artifact_count": 1,
            "min_logs": 2,
        },
    },
    {
        "title": "9. Details: Raphus cucullatus (dodo — extinct bird)",
        "entrypoint": "get_taxon_details",
        "request": "Tell me about the dodo",
        "params": TaxonDetailsParameters(taxon_id="Raphus cucullatus"),
        "expect": {
            "reply_contains": ["Raphus cucullatus"],
            "artifact_count": 1,
            "min_logs": 2,
        },
    },

    # ===== SYNONYMS (2 tests) =====
    {
        "title": "10. Synonyms: Bos taurus (domestic cattle)",
        "entrypoint": "get_synonyms",
        "request": "Synonyms of cattle",
        "params": GetSynonymsParameters(query="Bos taurus"),
        "expect": {
            "reply_contains": ["synonym"],
            "artifact_count": 1,
            "min_logs": 1,
        },
    },
    {
        "title": "11. Synonyms: Canis lupus (wolf)",
        "entrypoint": "get_synonyms",
        "request": "Synonyms of wolf",
        "params": GetSynonymsParameters(query="Canis lupus"),
        "expect": {
            "reply_contains": ["synonym"],
            "min_logs": 1,
        },
    },

    # ===== VERNACULAR NAMES (3 tests) =====
    {
        "title": "12. Vernacular: Ailuropoda melanoleuca (giant panda)",
        "entrypoint": "get_vernacular_names",
        "request": "Common names for giant panda",
        "params": GetVernacularNamesParameters(taxon_id="Ailuropoda melanoleuca"),
        "expect": {
            "reply_contains": ["names"],
            "artifact_count": 1,
            "min_logs": 2,
        },
    },
    {
        "title": "13. Vernacular: Coffea arabica (coffee — globally traded crop)",
        "entrypoint": "get_vernacular_names",
        "request": "Common names for coffee",
        "params": GetVernacularNamesParameters(taxon_id="Coffea arabica"),
        "expect": {
            "reply_contains": ["names"],
            "min_logs": 2,
        },
    },
    {
        "title": "14. Vernacular: Plasmodium falciparum (malaria parasite — edge case)",
        "entrypoint": "get_vernacular_names",
        "request": "Common names for malaria parasite",
        "params": GetVernacularNamesParameters(taxon_id="Plasmodium falciparum"),
        "expect": {
            "min_logs": 1,
        },
    },

    # ===== CLASSIFICATION (3 tests) =====
    {
        "title": "15. Classification: Apis mellifera (honeybee — Arthropoda)",
        "entrypoint": "get_classification",
        "request": "Classification of honeybee",
        "params": GetClassificationParameters(taxon_id="Apis mellifera"),
        "expect": {
            "reply_contains": ["Classification", "Arthropoda"],
            "artifact_count": 1,
            "min_logs": 1,
        },
    },
    {
        "title": "16. Classification: Saccharomyces cerevisiae (yeast — Fungi)",
        "entrypoint": "get_classification",
        "request": "Classification of baker's yeast",
        "params": GetClassificationParameters(taxon_id="Saccharomyces cerevisiae"),
        "expect": {
            "reply_contains": ["Classification", "Fungi"],
            "artifact_count": 1,
            "min_logs": 1,
        },
    },
    {
        "title": "17. Classification: Carcharodon carcharias (great white shark)",
        "entrypoint": "get_classification",
        "request": "What phylum is the great white shark?",
        "params": GetClassificationParameters(taxon_id="Carcharodon carcharias"),
        "expect": {
            "reply_contains": ["Classification", "Chordata"],
            "artifact_count": 1,
            "min_logs": 1,
        },
    },

    # ===== TAXON CHILDREN (3 tests) =====
    {
        "title": "18. Children: Rosa (genus — many rose species)",
        "entrypoint": "get_taxon_children",
        "request": "Species in genus Rosa",
        "params": GetTaxonChildrenParameters(taxon_id="Rosa", limit=10),
        "expect": {
            "reply_contains": ["child taxa"],
            "artifact_count": 1,
            "artifact_contains": ["Rosa"],
            "min_logs": 2,
        },
    },
    {
        "title": "19. Children: Canidae (family — dogs, wolves, foxes)",
        "entrypoint": "get_taxon_children",
        "request": "Genera in family Canidae",
        "params": GetTaxonChildrenParameters(taxon_id="Canidae", limit=15),
        "expect": {
            "reply_contains": ["child taxa"],
            "artifact_count": 1,
            "min_logs": 2,
        },
    },
    {
        "title": "20. Children: Pinales (order — conifer families)",
        "entrypoint": "get_taxon_children",
        "request": "Families in order Pinales",
        "params": GetTaxonChildrenParameters(taxon_id="Pinales"),
        "expect": {
            "reply_contains": ["child taxa"],
            "artifact_count": 1,
            "min_logs": 2,
        },
    },
]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

async def run_all(agent):
    passed = 0
    failed = 0
    results = []

    for t in ALL_TESTS:
        ok = await run_test(agent, t["title"], t["entrypoint"], t["request"], t["params"], t["expect"])
        results.append((t["title"], ok))
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"{C.BOLD}FINAL RESULTS: {passed} passed, {failed} failed out of {len(ALL_TESTS)}{C.R}")
    print(f"{'=' * 80}")

    if failed > 0:
        print(f"\n{C.RED}Failed tests:{C.R}")
        for title, ok in results:
            if not ok:
                print(f"  ✗ {title}")
    print()


async def run_by_name(agent, name):
    name_map = {
        "search": [0, 1, 2, 3, 4, 5],
        "details": [6, 7, 8],
        "synonyms": [9, 10],
        "vernacular": [11, 12, 13],
        "classification": [14, 15, 16],
        "children": [17, 18, 19],
    }
    if name not in name_map:
        print(f"Unknown group: '{name}'. Available: {', '.join(name_map.keys())}")
        return

    for idx in name_map[name]:
        t = ALL_TESTS[idx]
        await run_test(agent, t["title"], t["entrypoint"], t["request"], t["params"], t["expect"])


async def run_custom(agent, entrypoint, query, limit=None):
    param_map = {
        "search": lambda: SearchParameters(query=query, limit=limit or 5),
        "get_taxon_details": lambda: TaxonDetailsParameters(taxon_id=query),
        "get_synonyms": lambda: GetSynonymsParameters(query=query),
        "get_vernacular_names": lambda: GetVernacularNamesParameters(taxon_id=query),
        "get_classification": lambda: GetClassificationParameters(taxon_id=query),
        "get_taxon_children": lambda: GetTaxonChildrenParameters(taxon_id=query, limit=limit or 20),
    }
    if entrypoint not in param_map:
        print(f"Unknown entrypoint: '{entrypoint}'")
        print(f"Available: {', '.join(param_map.keys())}")
        return

    params = param_map[entrypoint]()
    await run_test(
        agent,
        f"Custom: {entrypoint} → {query}",
        entrypoint,
        f"User query: {query}",
        params,
        expect={"min_logs": 1},
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive COL Agent Tester (live API)")
    parser.add_argument("--test", type=str, help="Test group: search, details, synonyms, vernacular, classification, children")
    parser.add_argument("--entrypoint", type=str, help="Entrypoint for custom query")
    parser.add_argument("--query", type=str, help="Custom query string")
    parser.add_argument("--limit", type=int, help="Result limit")
    parser.add_argument("--all", action="store_true", help="Run all 20 tests")
    args = parser.parse_args()

    agent = CatalogueOfLifeAgent()

    if args.entrypoint and args.query:
        asyncio.run(run_custom(agent, args.entrypoint, args.query, args.limit))
    elif args.test:
        asyncio.run(run_by_name(agent, args.test))
    elif args.all:
        asyncio.run(run_all(agent))
    else:
        asyncio.run(run_all(agent))


if __name__ == "__main__":
    main()