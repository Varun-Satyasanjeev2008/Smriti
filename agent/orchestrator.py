"""
Core agent orchestration loop.

Implements the five-step research pipeline:
  1. Receive research query
  2. Call Hindsight recall — surface prior research
  3. Call PatentsView MCP tool — fetch candidate patents
  4. For each candidate, call Groq to get accept/reject verdict + reasoning
  5. Call Hindsight retain — persist each evaluation for future recall

The orchestrator communicates with the Patent Search MCP server via the
FastMCP in-process client. This avoids the overhead of a subprocess for
the common case where everything runs in one Docker container; if you want
true process isolation, swap the InMemoryClient for an SSEClient pointing
at http://patent-mcp:8001.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional

from fastmcp import Client as MCPClient

from memory.hindsight_client import HindsightClient
from patent_search_mcp.server import mcp as patent_mcp_server
from agent.llm import PatentRelevanceEvaluator
from agent.schemas import (
    PatentFinding,
    RecalledContext,
    ResearchRequest,
    ResearchResponse,
)

logger = logging.getLogger(__name__)


class PriorArtResearchAgent:
    """
    The core orchestration object.

    Instantiate once at application startup (in FastAPI lifespan) and reuse
    across requests. All I/O is async.
    """

    def __init__(self) -> None:
        self._memory = HindsightClient()
        self._evaluator = PatentRelevanceEvaluator()
        # In-process MCP client — talks to the FastMCP server object directly.
        # Zero network overhead; same process as the FastAPI app.
        self._mcp_client = MCPClient(patent_mcp_server)

    async def aclose(self) -> None:
        await self._memory.aclose()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def research(self, request: ResearchRequest) -> ResearchResponse:
        """
        Execute the full prior-art research pipeline for a given query.

        This is the single entry-point called by the FastAPI route handler.
        """
        start_ts = time.monotonic()
        query = request.query

        logger.info("Agent research started: query=%r", query)

        # ----------------------------------------------------------------
        # Step 2 — Recall prior research from Hindsight
        # ----------------------------------------------------------------
        recalled_memories = await self._memory.recall(query, top_k=5)
        recalled_context: List[RecalledContext] = []
        for i, mem in enumerate(recalled_memories):
            content  = mem.get("content", "")
            score    = mem.get("score")     # None if Hindsight doesn't expose it
            metadata = mem.get("metadata") or {}
            logger.info(
                "Recalled memory [%d]: content_len=%d  score=%s  metadata_keys=%s",
                i,
                len(content),
                score,
                list(metadata.keys()),
            )
            recalled_context.append(
                RecalledContext(
                    content=content,
                    score=score,
                    metadata=metadata if metadata else None,
                )
            )

        if recalled_context:
            logger.info(
                "Recalled %d prior memories for query %r", len(recalled_context), query
            )
        else:
            logger.info("No prior memories found for query %r", query)

        # ----------------------------------------------------------------
        # Step 3 — Search patents via MCP tool
        # ----------------------------------------------------------------
        candidate_patents = await self._search_patents_via_mcp(
            query=query,
            cpc_class=request.cpc_class,
            date_from=request.date_from,
            date_to=request.date_to,
            max_results=request.max_patents,
        )
        logger.info("MCP search returned %d candidate patents", len(candidate_patents))

        # ----------------------------------------------------------------
        # Steps 4 & 5 — Evaluate each patent and persist findings
        # ----------------------------------------------------------------
        findings: List[PatentFinding] = []
        retain_items: List[dict] = []

        # Evaluate patents concurrently (bounded by MCP rate limit on search)
        eval_tasks = [
            self._evaluate_patent(
                query=query,
                patent=p,
                prior_research=self._matching_prior_research(p, recalled_context),
            )
            for p in candidate_patents
        ]
        finding_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        for patent, result in zip(candidate_patents, finding_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Evaluation failed for patent %s: %s", patent.get("patent_id"), result
                )
                continue

            finding = result
            findings.append(finding)

            # Step 5 — queue this evaluation for a single batched retain call
            retain_content = HindsightClient.format_retain_content(
                query=query,
                patent_id=finding.patent_id,
                patent_title=finding.title,
                verdict=finding.verdict,
                reasoning=finding.reasoning,
            )
            retain_items.append({
                "content": retain_content,
                "document_id": f"patent-{finding.patent_id}",
            })

        # Persist all evaluations in one batched retain call (single LLM
        # fact-extraction call on Hindsight's side, instead of one per item —
        # keeps us well under Groq's TPM rate limit).
        if retain_items:
            try:
                await self._memory.retain_batch(retain_items)
            except Exception as exc:
                logger.warning("Hindsight retain_batch failed: %s", exc)
        # ----------------------------------------------------------------
        # Build summary
        # ----------------------------------------------------------------
        accepted = sum(1 for f in findings if f.verdict == "ACCEPTED")
        rejected = len(findings) - accepted
        elapsed_ms = (time.monotonic() - start_ts) * 1000

        summary = (
            f"Research complete in {elapsed_ms:.0f}ms. "
            f"Evaluated {len(findings)} patent(s): "
            f"{accepted} accepted as prior art, {rejected} rejected. "
            f"Found {len(recalled_context)} relevant prior research memory/memories."
        )

        logger.info("Agent research done: %s", summary)

        return ResearchResponse(
            query=query,
            recalled_context=recalled_context,
            findings=findings,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _search_patents_via_mcp(
        self,
        query: str,
        cpc_class: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        max_results: int,
    ) -> List[dict]:
        """Call the search_patents MCP tool and return raw patent dicts."""
        args: dict = {
            "query": query,
            "max_results": max_results,
        }
        if cpc_class:
            args["cpc_class"] = cpc_class
        if date_from:
            args["date_from"] = date_from
        if date_to:
            args["date_to"] = date_to

        async with self._mcp_client:
            result = await self._mcp_client.call_tool("search_patents", args)

        # FastMCP returns a list of TextContent objects; parse the first one
        if not result or not result.content:
            return []

        raw_text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        try:
            patents = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse MCP search_patents response: %s", raw_text[:200])
            patents = []

        return patents if isinstance(patents, list) else []

    async def _evaluate_patent(
        self, query: str, patent: dict, prior_research: str = ""
    ) -> PatentFinding:
        """Run the LLM evaluation for a single patent dict."""
        evaluation = await self._evaluator.evaluate(
            query=query,
            patent_id=patent.get("patent_id", ""),
            title=patent.get("title", ""),
            abstract=patent.get("abstract", ""),
            claims=patent.get("claims", ""),
            assignee=patent.get("assignee"),
            cpc_codes=patent.get("cpc_codes", []),
            filing_date=patent.get("filing_date"),
            grant_date=patent.get("grant_date"),
            publication_date=patent.get("publication_date"),
            source=patent.get("source"),
            prior_research=prior_research,
        )

        return PatentFinding(
            patent_id=patent.get("patent_id", ""),
            title=patent.get("title", ""),
            abstract=patent.get("abstract", ""),
            assignee=patent.get("assignee"),
            cpc_codes=patent.get("cpc_codes", []),
            filing_date=patent.get("filing_date"),
            grant_date=patent.get("grant_date"),
            publication_date=patent.get("publication_date"),
            source=patent.get("source"),
            verdict=evaluation["verdict"],
            confidence=evaluation["confidence"],
            reasoning=evaluation["reasoning"],
        )

    @staticmethod
    def _matching_prior_research(patent: dict, memories: List[RecalledContext]) -> str:
        patent_id = "".join(
            char for char in str(patent.get("patent_id", "")).casefold() if char.isalnum()
        )
        if not patent_id:
            return ""
        matches = []
        for memory in memories:
            content = memory.content or ""
            canonical_content = "".join(char for char in content.casefold() if char.isalnum())
            if patent_id in canonical_content:
                matches.append(content)
        return "\n\n".join(matches)[:4000]
