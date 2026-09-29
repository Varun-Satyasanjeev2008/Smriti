import json
import os
import re
from typing import List, Dict, Any

# Resolve the path to the data directory relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "sample_patents.json")

def load_patent_data() -> List[Dict[str, Any]]:
    """Loads the patent dataset from the local JSON file."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def search_patents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves patents matching the search query.
    For the MVP, this performs a basic case-insensitive keyword match 
    across the title, abstract, and claims.
    
    Args:
        query (str): The research query (e.g., "battery cooling").
        top_k (int): Maximum number of results to return.
        
    Returns:
        List[Dict[str, Any]]: A list of structured patent records.
    """
    if top_k <= 0 or not query or not query.strip():
        return []

    patents = load_patent_data()
    query_terms = list(dict.fromkeys(re.findall(r"[\w]+", query.casefold())))
    if not query_terms:
        return []
    
    # Score each patent based on term frequency in text fields
    scored_patents = []
    for patent in patents:
        score = 0
        title = patent.get('title') or ''
        abstract = patent.get('abstract') or ''
        claims = patent.get('claims') or ''
        searchable_text = f"{title} {abstract} {claims}".casefold()
        
        for term in query_terms:
            if term in searchable_text:
                # Basic scoring: count occurrences of the term
                score += searchable_text.count(term)
                
        if score > 0:
            scored_patents.append((score, patent))
            
    # Sort by score descending and return top_k
    # Python's sort is stable, so ties retain the source corpus order.
    scored_patents.sort(key=lambda x: x[0], reverse=True)
    
    results = [patent for score, patent in scored_patents[:top_k]]
    return results

if __name__ == "__main__":
    # Quick manual test
    res = search_patents("liquid cooling", top_k=2)
    print(json.dumps(res, indent=2))
