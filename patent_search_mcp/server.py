"""
Patent Search MCP Server.

Exposes two MCP tools backed by Google Patents and the optional local corpus:
  - search_patents(query, cpc_class, date_from, date_to) -> list[Patent]
  - get_patent_details(patent_id) -> Patent

Run with:
    python -m patent_search_mcp.server          # stdio transport (default)
    python -m patent_search_mcp.server --sse    # SSE transport on :8001

The server is intended to be mounted as a subprocess by the agent or
consumed directly via MCP client from any compatible host.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .client import PatentsViewClient
from .models import Patent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server initialisation
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="PatentSearchMCP",
    instructions=(
        "Search public patent records using Google Patents and the configured local corpus. "
        "Tools: search_patents, get_patent_details."
    ),
)

# Module-level client — shared across all tool calls within a single process.
_client: Optional[PatentsViewClient] = None


def _get_client() -> PatentsViewClient:
    global _client
    if _client is None:
        _client = PatentsViewClient()
    return _client


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_patents(
    query: str,
    cpc_class: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search public patent records and supplement the results with the local corpus.

    Args:
        query: Free-text search string applied to patent abstracts and titles.
        cpc_class: Optional CPC subgroup code to restrict results (e.g. 'G06N3/04').
        date_from: Optional lower bound for the patent grant date (YYYY-MM-DD).
        date_to: Optional upper bound for the patent grant date (YYYY-MM-DD).
        max_results: Maximum number of results to return (1-100, default 10).

    Returns:
        A list of patent records, each containing: patent_id, title, abstract,
        claims, assignee, cpc_codes, filing_date, grant_date.
    """
    client = _get_client()
    patents: List[Patent] = await client.search(
        query=query,
        cpc_class=cpc_class,
        date_from=date_from,
        date_to=date_to,
        size=max_results,
    )
    return [p.model_dump() for p in patents]


@mcp.tool()
async def get_patent_details(patent_id: str) -> Dict[str, Any]:
    """
    Retrieve the full details of a specific patent by its USPTO ID.

    Args:
        patent_id: The USPTO patent identifier (e.g. '10,123,456' or '10123456').

    Returns:
        A single patent record with all available fields: patent_id, title,
        abstract, claims, assignee, cpc_codes, filing_date, grant_date.
    """
    client = _get_client()
    patent: Patent = await client.get_patent(patent_id)
    return patent.model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patent Search MCP Server")
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Use SSE transport instead of stdio (binds to :8001)",
    )
    parser.add_argument("--port", type=int, default=8001, help="SSE port (default 8001)")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.sse:
        mcp.run(transport="sse", host="0.0.0.0", port=args.port)
    else:
        mcp.run(transport="stdio")
