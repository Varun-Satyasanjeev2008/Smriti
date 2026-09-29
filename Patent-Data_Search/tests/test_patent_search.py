import pytest
from patent_search import search_patents

def test_search_patents_basic_match():
    # US8927131B2 mentions "microencapsulated"
    results = search_patents("microencapsulated", top_k=5)
    assert len(results) > 0
    assert results[0]['patent_id'] == "US8927131B2"

def test_search_patents_multiple_keywords():
    # US9327577B2 mentions "heat pump system"
    results = search_patents("heat pump system", top_k=1)
    assert len(results) == 1
    assert results[0]['patent_id'] == "US9327577B2"

def test_search_patents_no_results():
    results = search_patents("quantum computing blockchain", top_k=5)
    assert len(results) == 0

def test_search_patents_top_k_limits():
    # Searching for "phase change" should match all 3 patents
    results = search_patents("phase change", top_k=2)
    assert len(results) == 2
