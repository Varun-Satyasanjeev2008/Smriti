"""
Patent search client.

Replaces the retired PatentsView Search API with the
patent-client-agents Google Patents client.

No PatentsView API key is required.

Public interface preserved:
    - PatentsViewClient.search(...)
    - PatentsViewClient.get_patent(...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import Patent

logger = logging.getLogger(__name__)


class PatentsViewClient:
    """
    Compatibility client for the existing MCP server.

    Internally uses Google Patents through patent-client-agents,
    while preserving the old PatentsViewClient interface.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._client = None

    async def _get_client(self):
        """
        Lazily create the Google Patents client.

        Lazy initialization keeps importing this module cheap and
        avoids opening a network client until it is actually needed.
        """
        if self._client is None:
            from patent_client_agents.google_patents import GooglePatentsClient

            self._client = GooglePatentsClient()

        return self._client

    async def aclose(self) -> None:
        """Close the underlying Google Patents client."""
        if self._client is not None:
            close = getattr(self._client, "aclose", None)

            if close is not None:
                result = close()

                if hasattr(result, "__await__"):
                    await result

            self._client = None

    async def __aenter__(self) -> "PatentsViewClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_value(
        obj: Any,
        *names: str,
        default: Any = None,
    ) -> Any:
        """
        Read a field from either a Pydantic/object result or a dict.
        """
        if obj is None:
            return default

        if isinstance(obj, dict):
            for name in names:
                if name in obj and obj[name] is not None:
                    return obj[name]
            return default

        for name in names:
            value = getattr(obj, name, None)

            if value is not None:
                return value

        return default

    @classmethod
    def _normalize(cls, raw: Any) -> Patent:
        """
        Convert a Google Patents result into the existing Patent model.

        The exact Google Patents result object can vary between versions
        of patent-client-agents, so this intentionally handles both
        dictionary-like and Pydantic/object-like responses.
        """

        patent_id = cls._get_value(
            raw,
            "publication_number",
            "patent_id",
            "publication_id",
            "patent_number",
            "number",
            default="",
        )

        title = cls._get_value(
            raw,
            "title",
            "patent_title",
            default="",
        )

        abstract = cls._get_value(
            raw,
            "abstract",
            "patent_abstract",
            default="",
        )

        claims = cls._get_value(
            raw,
            "claims",
            "claim_text",
            "claims_text",
            default="",
        )

        # --------------------------------------------------------------
        # Assignee
        # --------------------------------------------------------------

        assignee = cls._get_value(
            raw,
            "assignee",
            "assignee_name",
            "applicant",
            "applicant_name",
            default=None,
        )

        if isinstance(assignee, list):
            assignee = ", ".join(
                str(x)
                for x in assignee
                if x
            )

        # --------------------------------------------------------------
        # CPC codes
        # --------------------------------------------------------------

        cpc_codes = cls._get_value(
            raw,
            "cpc_codes",
            "cpc",
            "classifications",
            default=[],
        )

        if cpc_codes is None:
            cpc_codes = []

        if isinstance(cpc_codes, str):
            cpc_codes = [cpc_codes]

        normalized_cpcs: List[str] = []

        for code in cpc_codes:
            if isinstance(code, dict):
                code = (
                    code.get("code")
                    or code.get("cpc")
                    or code.get("classification")
                )

            if code:
                normalized_cpcs.append(str(code))

        # --------------------------------------------------------------
        # Dates
        # --------------------------------------------------------------

        filing_date = cls._get_value(
            raw,
            "filing_date",
            "application_date",
            "priority_date",
            default=None,
        )

        grant_date = cls._get_value(
            raw,
            "grant_date",
            "publication_date",
            "patent_date",
            "date",
            default=None,
        )

        # --------------------------------------------------------------
        # Claims may be a list
        # --------------------------------------------------------------

        if isinstance(claims, list):
            claim_texts: List[str] = []

            for claim in claims[:10]:
                if isinstance(claim, dict):
                    text = (
                        claim.get("text")
                        or claim.get("claim_text")
                        or ""
                    )
                else:
                    text = str(claim)

                if text:
                    claim_texts.append(text)

            claims = "\n\n".join(claim_texts)

        if claims is None:
            claims = ""

        # --------------------------------------------------------------
        # Convert everything to safe strings
        # --------------------------------------------------------------

        if patent_id is None:
            patent_id = ""

        if title is None:
            title = ""

        if abstract is None:
            abstract = ""

        return Patent(
            patent_id=str(patent_id),
            title=str(title),
            abstract=str(abstract),
            claims=str(claims),
            assignee=str(assignee) if assignee else None,
            cpc_codes=normalized_cpcs,
            filing_date=str(filing_date) if filing_date else None,
            grant_date=str(grant_date) if grant_date else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        cpc_class: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        size: int = 10,
    ) -> List[Patent]:
        """
        Search patents using Google Patents.

        Parameters
        ----------
        query:
            Free-text patent search query.

        cpc_class:
            Optional CPC classification filter.

        date_from:
            Optional lower date boundary.

        date_to:
            Optional upper date boundary.

        size:
            Maximum number of results.

        Returns
        -------
        List[Patent]
            Normalized patent records compatible with the existing MCP.
        """

        if not query or not query.strip():
            raise ValueError("Patent search query cannot be empty.")

        size = max(1, min(size, 100))

        client = await self._get_client()

        logger.info(
            "Searching Google Patents: query=%r size=%d",
            query,
            size,
        )

        # --------------------------------------------------------------
        # Build keywords.
        #
        # GooglePatentsClient accepts keyword-based searches.
        # Keep the query as one phrase so that the search engine can
        # interpret it naturally.
        # --------------------------------------------------------------

        keywords = [query.strip()]

        # --------------------------------------------------------------
        # Search.
        #
        # patent-client-agents exposes:
        #
        #     await client.search_patents(keywords=[...])
        #
        # The returned object contains a `results` collection.
        # --------------------------------------------------------------

        try:
            response = await client.search_patents(
                keywords=keywords,
            )
        except TypeError:
            # Compatibility fallback for versions where the method
            # accepts a positional query rather than keywords.
            response = await client.search_patents(
                query=query.strip(),
            )

        results = self._get_value(
            response,
            "results",
            "patents",
            "items",
            default=[],
        )

        if results is None:
            results = []

        normalized: List[Patent] = []

        for raw in list(results)[:size]:

            # ----------------------------------------------------------
            # Optional local filtering for CPC.
            # ----------------------------------------------------------

            patent = self._normalize(raw)

            if cpc_class:
                if patent.cpc_codes:
                    if not any(
                        str(code).startswith(cpc_class)
                        for code in patent.cpc_codes
                    ):
                        continue

            # ----------------------------------------------------------
            # Optional local date filtering.
            #
            # Google Patents may expose either publication_date or
            # grant_date depending on result type.
            # ----------------------------------------------------------

            if date_from and patent.grant_date:
                if str(patent.grant_date) < date_from:
                    continue

            if date_to and patent.grant_date:
                if str(patent.grant_date) > date_to:
                    continue

            normalized.append(patent)

            if len(normalized) >= size:
                break

        logger.info(
            "Google Patents search returned %d results",
            len(normalized),
        )

        return normalized

    async def get_patent(
        self,
        patent_id: str,
    ) -> Patent:
        """
        Fetch a single patent by publication number.

        Examples:
            US10123456B2
            US10123456
            EP1234567A1
        """

        if not patent_id or not patent_id.strip():
            raise ValueError("Patent ID cannot be empty.")

        patent_id = patent_id.strip()

        client = await self._get_client()

        logger.info(
            "Fetching patent details: %s",
            patent_id,
        )

        try:
            raw = await client.get_patent_data(
                patent_id,
            )
        except TypeError:
            # Some versions may require a keyword argument.
            raw = await client.get_patent_data(
                publication_number=patent_id,
            )

        if raw is None:
            raise ValueError(
                f"Patent {patent_id!r} was not found."
            )

        return self._normalize(raw)