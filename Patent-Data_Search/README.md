# Patent Data & Search

## 1. Purpose
This module provides the patent-data retrieval layer for the AI Patent Research Agent. Its core responsibility is:
`public patent records → preprocessing → keyword retrieval → structured patent results.`

This module handles querying the underlying data source and returning structured representations of prior art, which the downstream AI Agent can consume for analysis and Hindsight memory integration.

## 2. Current MVP Scope
The current Minimum Viable Product (MVP) operates on a **small local development corpus of exactly 3 verified public patent records**. 

This dataset is intentionally small and serves purely as a development fixture and deterministic demo corpus for testing the core search and Hindsight integration. **It is NOT the complete USPTO patent corpus**, nor does it search live internet databases.

## 3. Data Provenance
The 3 records within the local dataset are genuine, real-world public patent records related to battery thermal management. The available fields within these records are strictly limited to information that could be independently verified. 

This local dataset is **not** a USPTO bulk download. If certain fields (like `claims` or `cpc`) could not be reliably verified from the search snippets, they are intentionally set to `null` or `[]` rather than being fabricated or synthetically generated.

## 4. USPTO/PatentsView Status
The official USPTO / PatentsView ecosystem was thoroughly investigated as the intended larger-scale data source for this project. However:
- The current USPTO Open Data Portal (ODP) strictly requires authenticated access (MFA and personalized API keys) for programmatic bulk data ingestion.
- Because automated authenticated access was prevented in this development environment, direct USPTO/PatentsView bulk-data ingestion is **not currently enabled**.
- No synthetic or fake USPTO datasets were substituted.
- The current, verified 3-patent local corpus remains the working MVP data source.

## 5. Architecture
The architecture is designed to keep the data retrieval strictly isolated from the reasoning and memory layers.

```text
       Patent Data
            ↓
      Normalization
            ↓
      Patent Search
            ↓
search_patents(query, top_k)
            ↓
   Structured Results
            ↓
   Integration Layer
            ↓
       AI Agent
            ↓
       Hindsight
```
*Note: The AI Agent and Hindsight memory layers are entirely external to this module and are handled elsewhere in the project.*

## 6. Search Method
The current retrieval method relies on a basic **term-frequency / keyword-based retrieval** algorithm.

Matching operates by counting the occurrences of search query terms across the available text fields (such as the `title` and `abstract`). 
**This system does NOT perform semantic search or vector search.**

## 7. Interface
The module exposes a single, clean function for integration:

```python
from patent_search import search_patents

results = search_patents(
    query="battery cooling using phase change materials",
    top_k=5
)
```
- **`query`** *(str)*: A natural-language research string.
- **`top_k`** *(int)*: The maximum number of results to return.
- **Returns**: A `List` of structured Python dictionaries containing the patent metadata.

## 8. Output Schema
The JSON structure returned by the search function maps exactly to the following schema:

```json
{
    "patent_id": "...",
    "title": "...",
    "abstract": "...",
    "claims": null,
    "inventors": [],
    "assignee": "...",
    "publication_date": "...",
    "cpc": [],
    "source": "..."
}
```
*Important Note:* If a field is `null` or `[]` (such as `claims` or `cpc`), it means that specific information is either unavailable in the local dataset or was intentionally excluded because it could not be strictly verified.

## 9. Installation
This module relies primarily on standard Python libraries. Testing requires `pytest`.

To install the testing dependencies:
```bash
pip install -r requirements.txt
```

## 10. Testing
To run the automated test suite and verify the integrity of the search logic against the local 3-patent fixture:

```bash
python -m pytest tests/
```
*Current Test Status:* 4/4 tests passing successfully.

## 11. Smriti integration
The Smriti backend mounts this project read-only and calls `search_patents()` through its existing Patent Search MCP tool. In Docker Compose, keep `Patent Data_Search` and `Smriti` as sibling folders and start Compose from `Smriti`; Compose mounts this directory at `/opt/patent_data_search`. For local backend development, set `PATENT_DATA_SEARCH_PATH` to this directory, or keep the two project folders as siblings under `Documents`.

Local results are normalized into the backend patent schema and evaluated by the same agent/LLM flow as Google Patents results. Hindsight retains the evaluations through the existing backend flow. Patent IDs found in both sources are deduplicated. The local records only include publication dates, not grant dates, and do not have CPC codes, so searches filtered by grant date or CPC omit these records.

## 12. Known Limitations
Please be aware of the following strict limitations of the current implementation:
- **Only 3 local patent records** exist in the database.
- **Keyword-based retrieval** is used; there is **no semantic/vector search**.
- This is **not a live USPTO search**; it does not connect to the internet during runtime.
- **Limited metadata** is available; specifically, `claims` and `cpc` classifications may be unavailable or `null`.

## 13. Future Improvements
While out of scope for the current MVP, future iterations of this module could include:
- Ingestion of a larger, authenticated USPTO/PatentsView bulk corpus using an API key.
- Implementation of semantic / vector retrieval (e.g., embeddings).
- A hybrid keyword + semantic search pipeline.
- Richer ingestion of full `claims`, `cpc` codes, and citation networks.
