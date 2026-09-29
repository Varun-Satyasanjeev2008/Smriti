"""
Hindsight memory client.

Wraps the Hindsight REST API (http://localhost:8888 by default) to provide
recall, retain, and reflect operations.

Hindsight organises memory in "banks". This module uses a single bank called
'patent-prior-art' for all agent memories.

Docs / reference:
  https://github.com/vectorize-io/hindsight
  https://vectorize.io/docs/hindsight
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# The default bank name used for all patent prior-art research memory
DEFAULT_BANK_ID = "patent-prior-art"


def _get_hindsight_url() -> str:
    return os.environ.get("HINDSIGHT_URL", "http://localhost:8888").rstrip("/")


class HindsightClient:
    """
    Async HTTP client for the Hindsight memory API.

    Hindsight exposes (all scoped under /v1/default/banks/{bank_id}/):
      POST /memories/recall    — semantic search over stored memories
      POST /memories           — retain (persist) new memory items
      POST /reflect            — synthesise / summarise stored memories

    This client wraps those three operations with typed helpers and
    sensible defaults for the patent prior-art use-case.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        bank_id: str = DEFAULT_BANK_ID,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url or _get_hindsight_url()
        self._bank_id = bank_id
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "HindsightClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search Hindsight for memories related to the query.

        Returns a normalised list of memory dicts, each with the keys:
          - 'content' : the memory text (mapped from Hindsight's 'text' field)
          - 'score'   : relevance score if available (None if absent)
          - 'metadata': dict of supplementary fields (context, document_id,
                        type, id, mentioned_at, entities, …)

        Hindsight's recall API returns each result with 'text' as the memory
        body.  The old code used mem.get('content', '') which always returned ''
        because Hindsight does not use a 'content' key in recall results.
        This method normalises the raw Hindsight response so the orchestrator
        receives a stable structure regardless of Hindsight version.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "max_tokens": 4096,
            "budget": "mid",
        }
        logger.debug("Hindsight recall: query=%r bank=%s top_k=%d", query, self._bank_id, top_k)

        try:
            response = await self._http.post(
                f"/v1/default/banks/{self._bank_id}/memories/recall", json=payload
            )
            response.raise_for_status()
            data = response.json()

            # --- DEBUG: log raw response structure (first result only) ---
            # Hindsight wraps results under 'results' (primary) or 'memories'
            raw_list: List[Dict] = (
                data.get("results")
                or data.get("memories")
                or []
            )
            if raw_list:
                logger.info(
                    "Hindsight recall raw result[0] keys=%s  sample=%r",
                    list(raw_list[0].keys()) if raw_list else [],
                    json.dumps(raw_list[0])[:300],
                )
            else:
                logger.info(
                    "Hindsight recall returned 0 results. Top-level keys: %s",
                    list(data.keys()),
                )

            logger.info("Hindsight recall: %d raw result(s) received", len(raw_list))
            return [self._normalise_memory(m) for m in raw_list[:top_k]]

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Hindsight recall failed (HTTP %d): %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return []
        except Exception as exc:
            logger.warning("Hindsight recall error: %s", exc)
            return []

    @staticmethod
    def _normalise_memory(raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a raw Hindsight recall result into a stable dict.

        Hindsight recall results use 'text' for the memory body, not 'content'.
        This normalises the field names so the orchestrator layer only needs
        to call mem.get('content') regardless of Hindsight version.

        Mapping
        -------
        raw['text']        -> normalised['content']   (primary memory text)
        raw['context']     -> normalised['metadata']['context']
        raw['document_id'] -> normalised['metadata']['document_id']
        raw['type']        -> normalised['metadata']['type']
        raw['id']          -> normalised['metadata']['id']
        raw['mentioned_at']-> normalised['metadata']['mentioned_at']
        raw['entities']    -> normalised['metadata']['entities']
        raw['score']       -> normalised['score']  (if Hindsight ever adds it)
        """
        # Hindsight stores the memory body in 'text'.
        # Fall back through 'content' / 'fact' / 'memory' / 'observation'
        # so the mapping stays robust if Hindsight changes field names.
        content: str = (
            raw.get("text")
            or raw.get("content")
            or raw.get("fact")
            or raw.get("memory")
            or raw.get("observation")
            or ""
        )

        # Score: Hindsight hybrid retrieval doesn't currently expose a
        # per-result numeric score in the recall response.
        score: Optional[float] = raw.get("score") or raw.get("relevance_score")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None

        # Collect supplementary metadata fields
        metadata: Dict[str, Any] = {}
        for field in (
            "context", "document_id", "type", "id",
            "mentioned_at", "occurred_start", "occurred_end",
            "entities", "tags", "chunk_id",
        ):
            value = raw.get(field)
            if value is not None:
                metadata[field] = value

        return {
            "content": content,
            "score": score,
            "metadata": metadata if metadata else {},
        }

    async def retain(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a single new memory in Hindsight.

        For multiple memories in one request (recommended — uses one LLM
        fact-extraction call instead of many), use retain_batch() instead.
        """
        item: Dict[str, Any] = {
            "content": content,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if metadata:
            if "document_id" in metadata:
                item["document_id"] = metadata["document_id"]
            if "context" in metadata:
                item["context"] = metadata["context"]

        result = await self.retain_batch([item])
        return result

    async def retain_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store multiple memory items in a single Hindsight retain call.

        This is far more TPM-efficient than calling retain() in a loop,
        since Hindsight's fact-extraction LLM call happens once per request,
        not once per item.

        Parameters
        ----------
        items:
            List of dicts, each with a required 'content' key, and optional
            'document_id', 'context', and 'timestamp' keys.
        """
        payload: Dict[str, Any] = {
            "async": False,
            "items": items,
        }

        logger.debug(
            "Hindsight retain_batch: bank=%s items=%d", self._bank_id, len(items)
        )

        try:
            response = await self._http.post(
                f"/v1/default/banks/{self._bank_id}/memories", json=payload
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                "Hindsight retain_batch succeeded: items_count=%s",
                result.get("items_count", ""),
            )
            return result
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Hindsight retain_batch failed (HTTP %d): %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise
        except Exception as exc:
            logger.error("Hindsight retain_batch error: %s", exc)
            raise

    async def reflect(self, query: Optional[str] = None) -> str:
        """
        Trigger Hindsight's reflect operation to synthesise stored memories.
        """
        reflection_query = query or (
            "Summarise patterns in the patent prior-art research decisions: "
            "which patent categories are commonly rejected and why, "
            "and which are commonly accepted as prior art."
        )

        payload: Dict[str, Any] = {"query": reflection_query}

        logger.debug("Hindsight reflect: bank=%s", self._bank_id)

        try:
            response = await self._http.post(
                f"/v1/default/banks/{self._bank_id}/reflect", json=payload
            )
            response.raise_for_status()
            data = response.json()
            reflection: str = (
                data.get("reflection")
                or data.get("content")
                or data.get("result")
                or ""
            )
            logger.info("Hindsight reflect returned %d chars", len(reflection))
            return reflection
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Hindsight reflect failed (HTTP %d): %s",
                exc.response.status_code,
                exc.response.text,
            )
            return ""
        except Exception as exc:
            logger.warning("Hindsight reflect error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Convenience helper
    # ------------------------------------------------------------------

    @staticmethod
    def format_retain_content(
        query: str,
        patent_id: str,
        patent_title: str,
        verdict: str,
        reasoning: str,
    ) -> str:
        """
        Format a patent evaluation result as a memory content string.
        """
        return (
            f"PATENT PRIOR ART RESEARCH\n"
            f"Research Query: {query}\n"
            f"Patent ID: {patent_id}\n"
            f"Patent Title: {patent_title}\n"
            f"Verdict: {verdict.upper()}\n"
            f"Reasoning: {reasoning}"
        )