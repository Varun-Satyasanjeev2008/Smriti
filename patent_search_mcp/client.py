
"""
Patent search client.

Replaces the retired PatentsView Search API with the
patent-client-agents Google Patents client.

No PatentsView API key is required.

Public interface preserved:
    - PatentsViewClient.search(...)
    - PatentsViewClient.get_patent(...)

Retrieval improvements (2026-09):
    - Natural-language query is decomposed into 3-6 focused sub-queries
      using a deterministic, local keyword-extraction strategy (no LLM call).
    - Each sub-query is sent to Google Patents independently.
    - Results from all sub-queries are merged and deduplicated by canonical
      publication number.
    - A concept-coverage relevance score is computed before the final
      max_results slice is returned to the Groq evaluator.
      The scorer dynamically derives concept groups from the query, weights
      domain-defining concepts higher than generic AI/ML terms, applies a
      multi-concept bonus for breadth, and penalises CNN/ML-only results
      that lack any domain-specific overlap.
    - Local corpus results supplement (not dominate) live Google results.
    - Google rate-limit conditions are visible in logs and handled gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import importlib.util
import os
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Patent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words for query expansion (common English + patent filler words)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset = frozenset(
    """a an the and or but not in on at of to for with by from as is are was
       were be been being have has had do does did will would could should may
       might shall can this that these those it its which who whom whose what
       where when why how all each every both either some any more most very
       just also only than then so such about up out if into through during
       before after above below between under again further once use used using
       based invention method system device apparatus comprising including
       wherein thereby thereof thereon thereto therefore however provided
       according present disclosure embodiment embodiments described related
       relate relates relating""".split()
)

# Technical phrase patterns that should be preserved as units in search queries.
# These are matched case-insensitively against the original query.
_TECH_PHRASES: List[str] = [
    "convolutional neural network",
    "deep neural network",
    "machine learning",
    "deep learning",
    "reinforcement learning",
    "computer vision",
    "object detection",
    "image classification",
    "natural language processing",
    "edge computing",
    "edge device",
    "real time",
    "real-time",
    "adaptive control",
    "adaptive signal",
    "traffic signal",
    "traffic control",
    "traffic flow",
    "vehicle detection",
    "vehicle density",
    "signal timing",
    "intersection control",
    "license plate",
    "autonomous vehicle",
    "self driving",
    "neural network",
    "support vector",
    "random forest",
    "gradient descent",
    "transfer learning",
    "generative adversarial",
    "attention mechanism",
    "transformer model",
    "recurrent neural",
    "long short term",
    "point cloud",
    "semantic segmentation",
    "instance segmentation",
    "optical flow",
    "simultaneous localization",
    "global positioning",
    "inertial measurement",
    "image processing",
    "signal processing",
    "power management",
    "battery management",
    "wireless communication",
    "internet of things",
    "smart city",
    "federated learning",
    "anomaly detection",
    "predictive maintenance",
    "demand forecasting",
]


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

        # Google Patents search results expose a lightweight `snippet` rather
        # than a full abstract.  Use it as a fallback so the evaluator still
        # receives evidence from live search results.
        abstract = cls._get_value(
            raw,
            "abstract",
            "patent_abstract",
            "snippet",
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
            "current_assignee",
            "original_assignee",
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
            "cpc_classifications",
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
            elif not isinstance(code, str):
                code = getattr(code, "code", None)

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
            "patent_date",
            default=None,
        )
        publication_date = cls._get_value(
            raw,
            "publication_date",
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
            publication_date=str(publication_date) if publication_date else None,
            source=str(cls._get_value(raw, "source", "source_name", default="Google Patents")),
        )

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tech_phrases(query: str) -> List[str]:
        """
        Extract multi-word technical phrases from query (case-insensitive).

        Returns phrases in the order they appear in _TECH_PHRASES list,
        deduplicated.  Longer/more-specific phrases are listed first in
        _TECH_PHRASES so they take precedence over their sub-phrases.
        """
        q_lower = query.lower()
        found: List[str] = []
        seen: set = set()
        for phrase in _TECH_PHRASES:
            if phrase in q_lower and phrase not in seen:
                found.append(phrase)
                seen.add(phrase)
        return found

    @staticmethod
    def _extract_content_words(query: str) -> List[str]:
        """
        Tokenise query and return non-stop content words, length >= 4.

        Strips punctuation and removes stop words defined in _STOP_WORDS.
        Returns words in order of first appearance, deduplicated.
        """
        # Replace hyphens/underscores/slashes with space for uniform tokenisation
        cleaned = re.sub(r"[-_/]", " ", query.lower())
        # Remove all remaining punctuation
        cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
        tokens = cleaned.split()
        seen: set = set()
        words: List[str] = []
        for tok in tokens:
            if tok not in _STOP_WORDS and len(tok) >= 4 and tok not in seen:
                words.append(tok)
                seen.add(tok)
        return words

    @classmethod
    def expand_query(cls, query: str, max_queries: int = 5) -> List[str]:
        """
        Convert a natural-language research question into a small set of
        focused patent-search queries.

        Strategy
        --------
        1. Extract multi-word technical phrases (highest priority).
        2. Extract remaining content words (filtered stop-words, len >= 4).
        3. Build the primary query from the top phrases + important words.
        4. Build 2-4 additional focused sub-queries from concept pairs and
           individual anchor phrases.
        5. Return at most ``max_queries`` distinct queries.

        This is intentionally deterministic and local -- no LLM call.

        Parameters
        ----------
        query:
            The original research / invention description.
        max_queries:
            Maximum number of search strings to return (3-6 recommended).

        Returns
        -------
        list[str]
            A deduplicated, ordered list of search strings.
        """
        phrases = cls._extract_tech_phrases(query)
        words = cls._extract_content_words(query)

        # Remove from words any word already contained in an extracted phrase
        phrase_words: set = set()
        for p in phrases:
            for w in p.split():
                phrase_words.add(w)

        extra_words = [w for w in words if w not in phrase_words]

        queries: List[str] = []

        # --- Primary query: first 2-3 phrases + up to 3 extra words ---
        primary_parts = phrases[:3] + extra_words[:3]
        if primary_parts:
            queries.append(" ".join(primary_parts))

        # --- Phrase-pair sub-queries (adjacent pairs) ---
        for i in range(len(phrases) - 1):
            subq = phrases[i] + " " + phrases[i + 1]
            if subq not in queries:
                queries.append(subq)

        # --- Anchor phrase + extra words ---
        for phrase in phrases[:4]:
            tail = [w for w in extra_words if w not in phrase][:3]
            if tail:
                subq = phrase + " " + " ".join(tail)
                if subq not in queries:
                    queries.append(subq)

        # --- Individual anchor phrases (fallback when pairs exhausted) ---
        for phrase in phrases:
            if phrase not in queries:
                queries.append(phrase)

        # --- Absolute fallback: use the raw query + content words ---
        if not queries:
            queries.append(query.strip())
            if extra_words:
                queries.append(" ".join(extra_words[:6]))

        # Deduplicate while preserving order, then cap at max_queries
        seen: set = set()
        unique: List[str] = []
        for q in queries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                unique.append(q)
            if len(unique) >= max_queries:
                break

        return unique

    # ------------------------------------------------------------------
    # Concept-Coverage Relevance Scoring
    # ------------------------------------------------------------------

    # Generic technical words that should carry LOW weight so they don't
    # dominate ranking when they appear many times in a patent.
    _GENERIC_NOISE_WORDS: frozenset = frozenset([
        "system", "method", "device", "image", "model", "processing",
        "detection", "technology", "network", "learning", "data",
        "algorithm", "feature", "layer", "output", "input", "module",
        "architecture", "performance", "accuracy", "training",
    ])

    # Concept groups that are purely generic ML/AI and carry a lower
    # base weight — they should NOT dominate a domain-specific query.
    _GENERIC_ML_CONCEPTS: List[str] = [
        "convolutional neural network",
        "deep neural network",
        "neural network",
        "deep learning",
        "machine learning",
        "image classification",
        "model optimization",
        "model pruning",
        "cnn inference",
        "error correction",
        "reinforcement learning",
        "transfer learning",
        "generative adversarial",
        "attention mechanism",
        "transformer model",
        "recurrent neural",
        "gradient descent",
        "federated learning",
    ]

    @classmethod
    def _build_concept_groups(
        cls,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Dynamically derive weighted concept groups from an arbitrary research query.

        Returns a list of concept-group dicts, each with:
          - "label"   : human-readable group name (for logging)
          - "terms"   : list of phrases/words to look for (all lowercase)
          - "weight"  : importance multiplier for this group
          - "generic" : True if this group is generic ML/AI (used for noise penalty)

        Strategy
        --------
        1.  Extract recognised multi-word technical phrases from the query
            using the same _TECH_PHRASES list used for query expansion.
        2.  Extract remaining content words (non-stop, length >= 4).
        3.  Cluster the extracted phrases into groups:
            a.  Phrases that appear in _GENERIC_ML_CONCEPTS -> weight 1.0,
                generic=True.
            b.  All other extracted phrases (domain-specific) -> weight 2.5,
                generic=False, ordered by length (longer/more-specific first).
        4.  Remaining content words form a domain-word group (weight 1.5).
        5.  A "generic ML" catch-all group (weight 0.5, generic=True) ensures
            CNN-heavy patents still get *some* score without dominating.

        Weights are relative and only meaningful for sorting within a
        single result set.  They are NOT legal-relevance scores.
        """
        # --- Extract phrases and words from the query ---
        all_phrases = cls._extract_tech_phrases(query)   # preserves _TECH_PHRASES order
        all_words   = cls._extract_content_words(query)

        # Separate domain phrases from generic-ML phrases
        generic_ml_phrases: List[str] = []
        domain_phrases: List[str]     = []
        for ph in all_phrases:
            if any(gml in ph for gml in cls._GENERIC_ML_CONCEPTS) or ph in cls._GENERIC_ML_CONCEPTS:
                generic_ml_phrases.append(ph)
            else:
                domain_phrases.append(ph)

        # Words not already covered by extracted phrases
        phrase_words: set = set()
        for p in all_phrases:
            for w in p.split():
                phrase_words.add(w)
        domain_words = [
            w for w in all_words
            if w not in phrase_words and w not in cls._GENERIC_NOISE_WORDS
        ]

        groups: List[Dict[str, Any]] = []

        # --- Group A: domain-specific phrases (VERY HIGH weight) ---
        # Each distinct domain phrase becomes its own group so that
        # multi-concept coverage can be measured independently.
        for ph in domain_phrases:
            # Build a small synonym set: the phrase itself plus any
            # shorter variant that is a strict sub-phrase.
            terms = [ph]
            for other in domain_phrases:
                if other != ph and other in ph:
                    terms.append(other)
            groups.append({
                "label":   ph,
                "terms":   list(dict.fromkeys(terms)),  # deduped, order preserved
                "weight":  2.5,
                "generic": False,
            })

        # --- Group B: domain content words (HIGH weight) ---
        if domain_words:
            groups.append({
                "label":   "domain keywords",
                "terms":   domain_words,
                "weight":  1.5,
                "generic": False,
            })

        # --- Group C: generic ML phrases from the query (MEDIUM weight) ---
        if generic_ml_phrases:
            groups.append({
                "label":   "ml/ai concepts",
                "terms":   generic_ml_phrases,
                "weight":  1.0,
                "generic": True,
            })

        # --- Group D: generic-ML catch-all (LOW weight) ---
        # Ensures CNN/NN patents still score non-zero even if the query
        # does not explicitly contain those words.
        catchall = [
            g for g in cls._GENERIC_ML_CONCEPTS
            if g not in generic_ml_phrases
        ]
        if catchall:
            groups.append({
                "label":   "generic ml catch-all",
                "terms":   catchall[:6],   # keep it short
                "weight":  0.5,
                "generic": True,
            })

        return groups

    @classmethod
    def _score_concept_coverage(
        cls,
        patent: Patent,
        query: str,
        phrases: List[str],
        words: List[str],
    ) -> Tuple[float, List[str]]:
        """
        Compute a concept-coverage relevance score and the list of matched
        concept-group labels (for logging).

        Scoring components
        ------------------
        For each concept group:
          - Title match    : weight * 3.0  per term (max one hit per group)
          - Abstract match : weight * 2.0  per term (max one hit per group)
          - Claims match   : weight * 1.2  per term (max one hit per group)

        Multi-concept bonus:
          bonus = 1.5 * (distinct_domain_groups_matched ** 1.5)
          This super-linearly rewards patents that cover more domain aspects.

        Generic-noise penalty:
          If NO domain group matched but >= 1 generic ML group matched,
          a penalty of -2.0 is applied to push CNN-only patents below
          domain-relevant ones.

        Returns
        -------
        (score, matched_group_labels)
        """
        title_l    = (patent.title    or "").lower()
        abstract_l = (patent.abstract or "").lower()
        claims_l   = (patent.claims   or "").lower()

        # Best available text (snippet lives in abstract field after _normalize)

        concept_groups = cls._build_concept_groups(query)

        score = 0.0
        matched_labels: List[str]  = []
        domain_groups_matched = 0
        generic_groups_matched = 0

        for group in concept_groups:
            label   = group["label"]
            terms   = group["terms"]
            weight  = group["weight"]
            is_generic = group["generic"]

            group_score = 0.0
            group_hit   = False

            for term in terms:
                t = term.lower()
                if t in title_l:
                    group_score = max(group_score, weight * 3.0)
                    group_hit = True
                elif t in abstract_l:
                    group_score = max(group_score, weight * 2.0)
                    group_hit = True
                elif t in claims_l:
                    group_score = max(group_score, weight * 1.2)
                    group_hit = True

            if group_hit:
                score += group_score
                matched_labels.append(label)
                if is_generic:
                    generic_groups_matched += 1
                else:
                    domain_groups_matched += 1

        # Multi-concept bonus (super-linear: rewards breadth)
        if domain_groups_matched > 1:
            bonus = 1.5 * (domain_groups_matched ** 1.5)
            score += bonus

        # Generic-noise penalty: CNN/ML-only patents that have no domain hit
        if domain_groups_matched == 0 and generic_groups_matched > 0:
            score -= 2.0

        return score, matched_labels

    @classmethod
    def _filter_local_fallback_candidates(
        cls,
        local_candidates: List[Patent],
        query: str,
        phrases: List[str],
        words: List[str],
        *,
        minimum_score: float = 4.5,
    ) -> Tuple[List[Patent], List[Tuple[Patent, float, List[str], str]]]:
        """
        Filter local-corpus fallback candidates before they can reach the
        evaluator.

        The local corpus is a small development dataset. When live Google
        results are unavailable, weak lexical matches (for example a patent
        that only happens to mention "vehicle") must not be presented as
        meaningful prior-art candidates.

        Policy:
        - If the query contains one or more domain-specific concept phrases,
          a local candidate must match at least one of those phrases AND meet
          the minimum retrieval score.
        - If the query has no explicit domain phrase, fall back to the score
          threshold alone so generic engineering queries still have a usable
          development-corpus fallback.
        """
        concept_groups = cls._build_concept_groups(query)

        domain_phrase_labels = {
            group["label"]
            for group in concept_groups
            if not group["generic"] and group["label"] != "domain keywords"
        }

        accepted: List[Patent] = []
        rejected: List[Tuple[Patent, float, List[str], str]] = []

        for patent in local_candidates:
            score, matched_groups = cls._score_concept_coverage(
                patent, query, phrases, words
            )

            matched_domain_phrases = sorted(
                domain_phrase_labels.intersection(matched_groups)
            )

            if domain_phrase_labels:
                if matched_domain_phrases and score >= minimum_score:
                    accepted.append(patent)
                    continue
                reason = (
                    "no domain-specific concept phrase matched"
                    if not matched_domain_phrases
                    else f"score {score:.2f} below threshold {minimum_score:.2f}"
                )
            else:
                if score >= minimum_score:
                    accepted.append(patent)
                    continue
                reason = f"score {score:.2f} below threshold {minimum_score:.2f}"

            rejected.append((patent, score, matched_groups, reason))

        return accepted, rejected

    @classmethod
    def _relevance_score(
        cls,
        patent: Patent,
        query: str,
        phrases: List[str],
        words: List[str],
    ) -> float:
        """
        Public-facing wrapper: returns the concept-coverage score.

        Signature extended with ``query`` so the scorer can derive concept
        groups dynamically.  ``phrases`` and ``words`` are kept for
        backward-compatibility and used as a tiebreaker.

        This is a retrieval relevance score, NOT a legal prior-art score.
        """
        score, _ = cls._score_concept_coverage(patent, query, phrases, words)
        return score

    # ------------------------------------------------------------------
    # Single focused Google search (internal helper)
    # ------------------------------------------------------------------

    async def _google_search_one(
        self,
        sub_query: str,
        upstream_size: int,
        cpc_class: Optional[str],
    ) -> Tuple[List[Any], bool]:
        """
        Execute one focused Google Patents search.

        Returns
        -------
        (raw_results, rate_limited)
            raw_results  : list of raw result objects from the client
            rate_limited : True if the error looks like a rate-limit hit
        """
        try:
            client = await self._get_client()
            search_kwargs: Dict[str, Any] = {
                "keywords": [sub_query.strip()],
                "page_size": upstream_size,
            }

            if cpc_class:
                search_kwargs["cpc_codes"] = [cpc_class]

            response = await client.search_patents(**search_kwargs)

        except TypeError:
            # Older patent-client-agents versions may not support all kwargs.
            try:
                client = await self._get_client()
                response = await client.search_patents(keywords=[sub_query.strip()])
            except Exception as exc:
                err_str = str(exc).lower()
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                is_rate_limit = (
                    status_code in (429, 503)
                    or "429" in err_str
                    or "503" in err_str
                    or "rate" in err_str
                    or "limit" in err_str
                    or "cooldown" in err_str
                    or "unavailable" in err_str
                    or "too many" in err_str
                )
                if is_rate_limit:
                    logger.warning(
                        "Google Patents rate limited (inner) for sub-query %r: %s",
                        sub_query,
                        exc,
                    )
                    return [], True
                logger.warning(
                    "Google Patents search failed for sub-query %r: %s",
                    sub_query,
                    exc,
                )
                return [], False

        except Exception as exc:
            err_str = str(exc).lower()
            # Treat as rate-limited if the error looks like a throttle response.
            # We check for 429 (Too Many Requests), 503 (Service Unavailable —
            # how patent-client-agents signals its internal cooldown period),
            # and common rate-limit keywords in the exception message.
            is_rate_limit = (
                "429" in err_str
                or "503" in err_str
                or "rate" in err_str
                or "limit" in err_str
                or "cooldown" in err_str
                or "unavailable" in err_str
                or "too many" in err_str
            )
            # Also check for HTTPStatusError with status 429 or 503
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in (429, 503):
                is_rate_limit = True

            if is_rate_limit:
                logger.warning(
                    "Google Patents rate limited (outer) for sub-query %r: %s",
                    sub_query,
                    exc,
                )
                return [], True
            logger.warning(
                "Google Patents search failed for sub-query %r: %s",
                sub_query,
                exc,
            )
            return [], False

        raw_list = self._get_value(
            response,
            "results",
            "patents",
            "items",
            default=[],
        )

        if raw_list is None:
            raw_list = []

        return list(raw_list), False

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
        Search Google Patents and supplement with the optional local corpus.

        Pipeline
        --------
        1. Expand the natural-language query into focused sub-queries using a
           deterministic, local strategy (no LLM call), but issue only ONE
           Google Patents request per research call to minimize throttling.
        2. Request only the final candidate-page size from Google Patents and
           stop immediately on a rate-limit response.
        3. Merge and deduplicate live results by canonical patent ID.
        4. Apply grant-date filtering (date_from / date_to refer to GRANT dates
           -- this preserves the documented MCP contract).
        5. Supplement with local corpus results that are not already present.
           Local results are passed through the fallback relevance gate and do
           not displace stronger live results.
        6. Score every candidate deterministically and return the top ``size``
           by relevance score.

        Parameters
        ----------
        query:
            Free-text research question or invention description.
        cpc_class:
            Optional CPC subgroup prefix (e.g. "G06N3/04").
            Sent to Google for upstream filtering when supplied.
        date_from, date_to:
            Optional GRANT-date boundaries (YYYY-MM-DD).
            Grant-date filtering is applied locally because Google Patents'
            own date parameters target filing / publication dates.
        size:
            Maximum number of results to return (1-100).
        """

        if not query or not query.strip():
            raise ValueError("Patent search query cannot be empty.")

        size = max(1, min(size, 100))

        logger.info(
            "=== Patent search START: query=%r  cpc=%r  date_from=%r  date_to=%r  size=%d",
            query,
            cpc_class,
            date_from,
            date_to,
            size,
        )

        # ----------------------------------------------------------------
        # Step 1 -- Query expansion
        # ----------------------------------------------------------------
        # Use exactly one focused sub-query per research call.
        # Repeated Google requests were consistently hitting the public
        # endpoint's throttling.  Keeping one request makes the live search
        # conservative while preserving the existing query-expansion/ranking
        # machinery.  We choose the first (primary) expanded query because it
        # contains the most information-dense combination of concepts.
        sub_queries = self.expand_query(query, max_queries=1) or [query.strip()]
        phrases = self._extract_tech_phrases(query)
        words = self._extract_content_words(query)
        # Words that are already covered by the extracted phrases
        phrase_words: set = set()
        for p in phrases:
            for w in p.split():
                phrase_words.add(w)
        extra_words = [w for w in words if w not in phrase_words]

        logger.info(
            "Query expansion produced %d sub-queries: %s",
            len(sub_queries),
            sub_queries,
        )

        # ----------------------------------------------------------------
        # Step 2 -- Run the single focused search
        # ----------------------------------------------------------------
        # Keep the upstream page small to reduce load on Google and avoid
        # requesting a much larger result set than the evaluator can consume.
        # Grant/date filtering and concept ranking are still applied locally.
        upstream_per_query = max(1, min(size, 5))

        all_raw: List[Any] = []
        any_rate_limited = False

        for i, sub_query in enumerate(sub_queries):
            logger.info(
                "Focused search [%d/%d]: %r  (upstream_size=%d)",
                i + 1,
                len(sub_queries),
                sub_query,
                upstream_per_query,
            )
            raw_results, rate_limited = await self._google_search_one(
                sub_query, upstream_per_query, cpc_class
            )
            logger.info(
                "Focused search [%d/%d] returned %d raw results",
                i + 1,
                len(sub_queries),
                len(raw_results),
            )
            all_raw.extend(raw_results)

            if rate_limited:
                any_rate_limited = True
                logger.warning(
                    "Google Patents rate limit hit on sub-query [%d/%d]. "
                    "No further Google requests will be made for this research call.",
                    i + 1,
                    len(sub_queries),
                )
                break  # Stop issuing more Google requests

            # Only one live Google request is issued per research call, so no
            # inter-request pause is necessary here.

        if any_rate_limited and not all_raw:
            logger.warning(
                "All Google searches were rate-limited; falling back to local corpus only."
            )

        # ----------------------------------------------------------------
        # Step 3 -- Normalise and deduplicate by canonical patent ID
        # ----------------------------------------------------------------
        seen_ids: set = set()
        deduplicated: List[Patent] = []

        for raw in all_raw:
            patent = self._normalize(raw)
            cid = self._canonical_id(patent.patent_id)
            if not cid:
                # No ID -- always keep (unusual but harmless)
                deduplicated.append(patent)
            elif cid not in seen_ids:
                seen_ids.add(cid)
                deduplicated.append(patent)

        logger.info(
            "Deduplication: %d unique candidates from %d raw results",
            len(deduplicated),
            len(all_raw),
        )

        # ----------------------------------------------------------------
        # Step 4 -- Grant-date filtering (MCP contract: date_from/date_to = grant dates)
        # ----------------------------------------------------------------
        date_filtered: List[Patent] = []
        for patent in deduplicated:
            if date_from and (
                not patent.grant_date or patent.grant_date < date_from
            ):
                continue
            if date_to and (
                not patent.grant_date or patent.grant_date > date_to
            ):
                continue
            date_filtered.append(patent)

        logger.info(
            "After grant-date filter: %d candidates remain",
            len(date_filtered),
        )

        # ----------------------------------------------------------------
        # Step 5 -- Supplement with optional local corpus
        # ----------------------------------------------------------------
        local_candidates = self._search_local_corpus(query, size)
        local_candidates_filtered: List[Patent] = []

        for patent in local_candidates:
            if cpc_class and (
                not patent.cpc_codes
                or not any(
                    str(code).startswith(cpc_class)
                    for code in patent.cpc_codes
                )
            ):
                continue
            if date_from and (
                not patent.grant_date or patent.grant_date < date_from
            ):
                continue
            if date_to and (
                not patent.grant_date or patent.grant_date > date_to
            ):
                continue
            local_candidates_filtered.append(patent)

        accepted_local, rejected_local = self._filter_local_fallback_candidates(
            local_candidates_filtered,
            query,
            phrases,
            extra_words,
            minimum_score=4.5,
        )

        local_added = 0
        for patent in accepted_local:
            cid = self._canonical_id(patent.patent_id)
            if cid not in seen_ids:
                seen_ids.add(cid)
                date_filtered.append(patent)
                local_added += 1

        logger.info(
            "Local fallback candidates: %d before relevance threshold, %d accepted, %d rejected",
            len(local_candidates_filtered),
            len(accepted_local),
            len(rejected_local),
        )

        for patent, sc, matched_groups, reason in rejected_local:
            logger.info(
                "Local fallback filtered out: ID=%s title=%r score=%.2f groups=%s reason=%s",
                patent.patent_id,
                (patent.title or "")[:100],
                sc,
                matched_groups if matched_groups else "(none)",
                reason,
            )

        if local_added:
            logger.info(
                "Local corpus supplemented with %d sufficiently relevant result(s)",
                local_added,
            )
        else:
            logger.info(
                "Local corpus contributed 0 sufficiently relevant result(s)"
            )

        # ----------------------------------------------------------------
        # Step 6 -- Concept-Coverage Relevance Ranking
        # ----------------------------------------------------------------
        scored: List[Tuple[float, Patent, List[str]]] = []
        for patent in date_filtered:
            sc, matched_groups = self._score_concept_coverage(
                patent, query, phrases, extra_words
            )
            scored.append((sc, patent, matched_groups))

        # Sort descending by score; Python sort is stable so Google relevance
        # ordering is preserved among patents with identical scores.
        scored.sort(key=lambda t: t[0], reverse=True)

        ranked = [patent for _, patent, _ in scored]

        logger.info(
            "\n=== Concept-Coverage Ranking: top %d of %d candidates ===",
            min(size, len(ranked)),
            len(ranked),
        )
        logger.info("Original query: %r", query)
        logger.info(
            "Expanded sub-queries: %s", sub_queries
        )
        for idx, (sc, patent, mgroups) in enumerate(scored[:size]):
            logger.info(
                "Candidate [%d/%d]\n"
                "  ID    : %s\n"
                "  Title : %s\n"
                "  Score : %.2f\n"
                "  Matched concept groups: %s",
                idx + 1,
                min(size, len(ranked)),
                patent.patent_id,
                (patent.title or "")[:100],
                sc,
                mgroups if mgroups else "(none)",
            )

        final = ranked[:size]

        logger.info(
            "=== Patent search END: returning %d result(s) to evaluator ===",
            len(final),
        )

        return final

    @staticmethod
    def _canonical_id(value: str) -> str:
        """Return a canonical (lower-case, alphanumeric-only) patent ID."""
        return "".join(ch for ch in value.casefold() if ch.isalnum())

    @staticmethod
    def _search_local_corpus(query: str, size: int) -> List[Patent]:
        # Primary: explicit env var (always set inside Docker via docker-compose.yml)
        configured_path = os.environ.get("PATENT_DATA_SEARCH_PATH")

        if configured_path:
            corpus_dir = Path(configured_path)
        else:
            # Development fallback (no Docker).
            # patent_search_mcp/client.py is two levels below the repo root;
            # Patent-Data_Search/ is a sibling of patent_search_mcp/.
            # Resolve: client.py -> patent_search_mcp/ -> repo root -> Patent-Data_Search/
            corpus_dir = Path(__file__).resolve().parent.parent / "Patent-Data_Search"

        module_path = corpus_dir / "patent_search.py"
        logger.debug("Local corpus path resolved to: %s (exists=%s)", module_path, module_path.is_file())

        if not module_path.is_file():
            logger.info("Local patent corpus is unavailable at %s", module_path)
            return []

        spec = importlib.util.spec_from_file_location("smriti_local_patent_search", module_path)
        if spec is None or spec.loader is None:
            logger.warning("Could not load local patent search module at %s", module_path)
            return []
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            records = module.search_patents(query=query, top_k=size)
            patents: List[Patent] = []
            for record in records:
                patents.append(Patent(
                    patent_id=str(record.get("patent_id") or ""),
                    title=str(record.get("title") or ""),
                    abstract=str(record.get("abstract") or ""),
                    claims=str(record.get("claims") or ""),
                    assignee=record.get("assignee"),
                    cpc_codes=record.get("cpc") or [],
                    publication_date=record.get("publication_date"),
                    # This is an application-provided development fallback,
                    # not a live Google Patents result. Keep provenance explicit.
                    source="Local Patent Corpus",
                ))
            return patents
        except Exception:
            logger.exception("Local patent corpus search failed at %s", module_path)
            return []

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
