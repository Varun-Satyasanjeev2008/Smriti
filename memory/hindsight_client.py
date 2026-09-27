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

        Returns a list of memory objects as returned by Hindsight.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "max_tokens": 4096,
            "budget": "mid",
        }
        logger.debug("Hindsight recall: query=%r bank=%s", query, self._bank_id)

        try:
            response = await self._http.post(
                f"/v1/default/banks/{self._bank_id}/memories/recall", json=payload
            )
            response.raise_for_status()
            data = response.json()
            # Hindsight wraps results in a 'memories' or 'results' key
            memories: List[Dict] = (
                data.get("memories") or data.get("results") or []
            )
            logger.info("Hindsight recall returned %d memories", len(memories))
            return memories[:top_k]
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Hindsight recall failed (HTTP %d): %s",
                exc.response.status_code,
                exc.response.text,
            )
            return []
        except Exception as exc:
            logger.warning("Hindsight recall error: %s", exc)
            return []

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