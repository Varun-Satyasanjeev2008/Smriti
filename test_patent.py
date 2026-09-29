"""
Unit tests for the patent_search_mcp.client module.

Tests cover:
  - Query expansion (expand_query, _extract_tech_phrases, _extract_content_words)
  - Deduplication (_canonical_id, dedup logic via search() stub)
  - Relevance ranking (_relevance_score)
  - CPC handling in _normalize
  - Date handling (grant-date MCP contract)
  - Patent ID normalization (_normalize)

Run with:
    pytest test_patent.py -v
"""

from __future__ import annotations

import asyncio
import sys
import os
import unittest
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from patent_search_mcp.client import PatentsViewClient, _TECH_PHRASES, _STOP_WORDS
from patent_search_mcp.models import Patent


# ===========================================================================
# Helpers
# ===========================================================================

def _make_patent(**kwargs) -> Patent:
    defaults = dict(
        patent_id="US12345678",
        title="",
        abstract="",
        claims="",
        assignee=None,
        cpc_codes=[],
        filing_date=None,
        grant_date=None,
        publication_date=None,
        source="test",
    )
    defaults.update(kwargs)
    return Patent(**defaults)


class _FakeResponse:
    """Minimal fake response object that _get_value can introspect."""

    def __init__(self, results):
        self.results = results


# ===========================================================================
# Query expansion tests
# ===========================================================================


class TestExtractTechPhrases(unittest.TestCase):
    """Tests for _extract_tech_phrases."""

    def test_traffic_signal_query(self):
        query = (
            "A smart traffic signal control system using computer vision "
            "to detect vehicle density in real time, with a convolutional "
            "neural network running on an edge device and adaptive signal "
            "timing based on the detected traffic conditions."
        )
        phrases = PatentsViewClient._extract_tech_phrases(query)
        # All of these must be found
        for expected in [
            "convolutional neural network",
            "computer vision",
            "vehicle density",
            "traffic signal",
            "edge device",
            "adaptive signal",
        ]:
            self.assertIn(expected, phrases, f"Expected phrase {expected!r} not found")

    def test_no_phrases_when_unrelated(self):
        query = "A system for sorting apples by colour in a fruit warehouse."
        phrases = PatentsViewClient._extract_tech_phrases(query)
        self.assertEqual(phrases, [])

    def test_deduplication(self):
        # Phrase appearing twice should only appear once
        query = "machine learning and machine learning again"
        phrases = PatentsViewClient._extract_tech_phrases(query)
        self.assertEqual(phrases.count("machine learning"), 1)

    def test_single_phrase(self):
        query = "deep learning for natural language processing tasks"
        phrases = PatentsViewClient._extract_tech_phrases(query)
        self.assertIn("deep learning", phrases)
        self.assertIn("natural language processing", phrases)


class TestExtractContentWords(unittest.TestCase):
    """Tests for _extract_content_words."""

    def test_stop_words_removed(self):
        query = "the system is a device that uses machine learning"
        words = PatentsViewClient._extract_content_words(query)
        for sw in _STOP_WORDS:
            self.assertNotIn(sw, words)

    def test_short_tokens_removed(self):
        # 'system' and 'device' are stop words in this implementation;
        # 'object' and 'detection' are not.
        query = "AI ML DL system for object detection pipeline"
        words = PatentsViewClient._extract_content_words(query)
        # ai, ml, dl are < 4 chars so they are excluded
        self.assertNotIn("ai", words)
        self.assertNotIn("ml", words)
        self.assertNotIn("dl", words)
        # 'system' is a stop word in this implementation
        self.assertNotIn("system", words)
        self.assertIn("object", words)
        self.assertIn("detection", words)
        self.assertIn("pipeline", words)

    def test_hyphen_normalised(self):
        query = "real-time camera processing pipeline"
        words = PatentsViewClient._extract_content_words(query)
        # Hyphens replaced by spaces
        self.assertIn("real", words)
        self.assertIn("time", words)
        self.assertIn("camera", words)
        self.assertIn("processing", words)
        self.assertIn("pipeline", words)
        # 'device' is a stop word so it would be excluded; don't assert it

    def test_deduplicated(self):
        query = "traffic signal traffic control traffic flow"
        words = PatentsViewClient._extract_content_words(query)
        self.assertEqual(words.count("traffic"), 1)


class TestExpandQuery(unittest.TestCase):
    """Tests for expand_query."""

    TRAFFIC_QUERY = (
        "A smart traffic signal control system using computer vision "
        "to detect vehicle density in real time, with a convolutional "
        "neural network running on an edge device and adaptive signal "
        "timing based on the detected traffic conditions."
    )

    def test_count_in_range(self):
        queries = PatentsViewClient.expand_query(self.TRAFFIC_QUERY)
        self.assertGreaterEqual(len(queries), 1)
        self.assertLessEqual(len(queries), 6)

    def test_max_queries_respected(self):
        for limit in [1, 3, 5]:
            queries = PatentsViewClient.expand_query(self.TRAFFIC_QUERY, max_queries=limit)
            self.assertLessEqual(len(queries), limit)

    def test_no_duplicates(self):
        queries = PatentsViewClient.expand_query(self.TRAFFIC_QUERY)
        self.assertEqual(len(queries), len(set(queries)))

    def test_traffic_signal_concepts_present(self):
        queries = PatentsViewClient.expand_query(self.TRAFFIC_QUERY)
        combined = " ".join(queries).lower()
        # CNN-related phrases must appear (highest priority phrases)
        self.assertIn("convolutional neural network", combined)
        self.assertIn("computer vision", combined)
        # At least one traffic/signal-related concept should appear
        # (traffic signal, adaptive signal, vehicle density, real time, or edge device)
        traffic_concepts = ["adaptive signal", "traffic signal", "vehicle density", "real time", "edge device"]
        found = [c for c in traffic_concepts if c in combined]
        self.assertGreater(
            len(found), 0,
            f"Expected at least one of {traffic_concepts} in expanded queries, got: {combined[:200]}",
        )

    def test_fallback_on_plain_query(self):
        # A query with no recognised tech phrases should still produce output
        query = "sorting apples by weight in a warehouse"
        queries = PatentsViewClient.expand_query(query)
        self.assertGreater(len(queries), 0)
        # The raw query (or the top content words) must appear
        combined = " ".join(queries).lower()
        self.assertIn("sorting", combined)

    def test_no_empty_queries(self):
        queries = PatentsViewClient.expand_query(self.TRAFFIC_QUERY)
        for q in queries:
            self.assertTrue(q.strip(), "Empty query string in expansion result")

    def test_nn_query(self):
        # This query does NOT contain "convolutional neural network" as a phrase
        # (it has "convolutional layers" not "convolutional neural network"),
        # so we only assert that 'neural network' and 'image classification' appear.
        query = "Neural network image classification using convolutional layers and residual skip connections"
        queries = PatentsViewClient.expand_query(query)
        combined = " ".join(queries).lower()
        self.assertIn("neural network", combined)
        self.assertIn("image classification", combined)
        # The word 'convolutional' should appear as a content word
        self.assertIn("convolutional", combined)


# ===========================================================================
# Normalization tests
# ===========================================================================


class TestNormalize(unittest.TestCase):
    """Tests for _normalize."""

    def test_dict_input_basic(self):
        raw = {
            "publication_number": "US10000001B2",
            "title": "A test patent",
            "snippet": "Abstract text from snippet",
        }
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.patent_id, "US10000001B2")
        self.assertEqual(patent.title, "A test patent")
        self.assertEqual(patent.abstract, "Abstract text from snippet")

    def test_snippet_fallback_when_no_abstract(self):
        raw = {"publication_number": "US1", "snippet": "snippet text"}
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.abstract, "snippet text")

    def test_abstract_preferred_over_snippet(self):
        raw = {
            "publication_number": "US2",
            "abstract": "full abstract",
            "snippet": "snippet only",
        }
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.abstract, "full abstract")

    def test_cpc_codes_list(self):
        raw = {
            "publication_number": "US3",
            "cpc_codes": ["G06N3/04", "G06V10/82"],
        }
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.cpc_codes, ["G06N3/04", "G06V10/82"])

    def test_cpc_codes_dict_objects(self):
        raw = {
            "publication_number": "US4",
            "cpc_codes": [{"code": "G06N3/04"}, {"cpc": "H04L29/06"}],
        }
        patent = PatentsViewClient._normalize(raw)
        self.assertIn("G06N3/04", patent.cpc_codes)
        self.assertIn("H04L29/06", patent.cpc_codes)

    def test_cpc_codes_string(self):
        raw = {"publication_number": "US5", "cpc_codes": "G06N3/04"}
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.cpc_codes, ["G06N3/04"])

    def test_assignee_list_joined(self):
        raw = {
            "publication_number": "US6",
            "assignee": ["Acme Corp", "Widget Inc"],
        }
        patent = PatentsViewClient._normalize(raw)
        self.assertIn("Acme Corp", patent.assignee)
        self.assertIn("Widget Inc", patent.assignee)

    def test_claims_list_joined(self):
        raw = {
            "publication_number": "US7",
            "claims": [
                {"text": "1. A method comprising..."},
                {"text": "2. The method of claim 1..."},
            ],
        }
        patent = PatentsViewClient._normalize(raw)
        self.assertIn("1. A method comprising", patent.claims)
        self.assertIn("2. The method of claim 1", patent.claims)

    def test_dates_extracted(self):
        raw = {
            "publication_number": "US8",
            "filing_date": "2020-01-15",
            "grant_date": "2022-06-01",
            "publication_date": "2021-03-10",
        }
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.filing_date, "2020-01-15")
        self.assertEqual(patent.grant_date, "2022-06-01")
        self.assertEqual(patent.publication_date, "2021-03-10")

    def test_none_fields_safe(self):
        raw = {"publication_number": None}
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.patent_id, "")
        self.assertEqual(patent.title, "")
        self.assertEqual(patent.abstract, "")

    def test_cn_patent_id_preserved(self):
        """Non-US IDs must not be altered during normalisation."""
        raw = {"publication_number": "CN104268596A", "title": "Test"}
        patent = PatentsViewClient._normalize(raw)
        self.assertEqual(patent.patent_id, "CN104268596A")

    def test_object_style_result(self):
        """_normalize must also work on Pydantic-style attribute objects."""

        class FakeResult:
            publication_number = "EP1234567A1"
            title = "EP patent"
            abstract = "EP abstract"
            claims = ""
            assignee = None
            cpc_codes = []
            filing_date = "2019-01-01"
            grant_date = None
            publication_date = None
            source = "Google Patents"

        patent = PatentsViewClient._normalize(FakeResult())
        self.assertEqual(patent.patent_id, "EP1234567A1")
        self.assertEqual(patent.title, "EP patent")


# ===========================================================================
# Canonical ID tests
# ===========================================================================


class TestCanonicalId(unittest.TestCase):
    """Tests for _canonical_id."""

    def test_strips_spaces(self):
        self.assertEqual(
            PatentsViewClient._canonical_id("US 10000001 B2"),
            PatentsViewClient._canonical_id("US10000001B2"),
        )

    def test_case_insensitive(self):
        self.assertEqual(
            PatentsViewClient._canonical_id("us10000001b2"),
            PatentsViewClient._canonical_id("US10000001B2"),
        )

    def test_cn_id(self):
        cid = PatentsViewClient._canonical_id("CN104268596A")
        self.assertEqual(cid, "cn104268596a")

    def test_empty(self):
        self.assertEqual(PatentsViewClient._canonical_id(""), "")


# ===========================================================================
# Relevance ranking tests
# ===========================================================================


class TestRelevanceScore(unittest.TestCase):
    """Tests for _relevance_score."""

    TRAFFIC_PHRASES = [
        "convolutional neural network",
        "computer vision",
        "vehicle detection",
        "traffic signal",
        "adaptive signal",
    ]
    TRAFFIC_WORDS = ["density", "timing", "edge", "intersection", "detect"]

    # Build a synthetic query from the phrase list so the new scorer
    # (which needs a query string) still works with the legacy test data.
    _SYNTHETIC_QUERY = (
        "A smart traffic signal control system using computer vision "
        "to detect vehicle density in real time with a convolutional "
        "neural network running on an edge device and adaptive signal timing."
    )

    def _score(self, patent: Patent) -> float:
        return PatentsViewClient._relevance_score(
            patent, self._SYNTHETIC_QUERY, self.TRAFFIC_PHRASES, self.TRAFFIC_WORDS
        )

    def test_title_phrase_match_beats_abstract_match(self):
        # For a title match to beat an abstract match, the title must match more
        # phrases than the abstract alternative. Here title_match has 2 phrases
        # in title vs abstract_match which has 3 phrases in abstract.
        title_match = _make_patent(
            patent_id="US1",
            title="Adaptive traffic signal control using convolutional neural network",
            abstract="An unrelated abstract.",
        )
        # Only one phrase match in abstract -- fewer than title_match's two phrases
        abstract_match = _make_patent(
            patent_id="US2",
            title="An unrelated title",
            abstract="Vehicle detection system.",
        )
        self.assertGreater(self._score(title_match), self._score(abstract_match))

    def test_multi_phrase_beats_single_phrase(self):
        multi = _make_patent(
            patent_id="US3",
            title="Traffic signal control with convolutional neural network",
            abstract="Computer vision and vehicle detection for adaptive signal timing.",
        )
        single = _make_patent(
            patent_id="US4",
            title="Traffic management",
            abstract="A system for controlling traffic.",
        )
        self.assertGreater(self._score(multi), self._score(single))

    def test_irrelevant_patent_scores_zero_or_low(self):
        irrelevant = _make_patent(
            patent_id="US5",
            title="Method for sorting fruit in a warehouse",
            abstract="Apples and oranges are sorted by weight.",
        )
        score = self._score(irrelevant)
        self.assertLess(score, 1.0)

    def test_coverage_bonus_increases_score(self):
        """A patent covering more distinct phrases should rank higher."""
        covers_all = _make_patent(
            patent_id="US6",
            title="Convolutional neural network for traffic signal control",
            abstract="Computer vision system for vehicle detection using adaptive signal timing.",
        )
        covers_one = _make_patent(
            patent_id="US7",
            title="Traffic signal timing method",
            abstract="An unrelated abstract.",
        )
        self.assertGreater(self._score(covers_all), self._score(covers_one))


# ===========================================================================
# Date handling tests (MCP contract: date_from/date_to = grant dates)
# ===========================================================================


class TestDateHandling(unittest.IsolatedAsyncioTestCase):
    """Tests that date_from/date_to are applied as GRANT-date boundaries."""

    async def _run_search_with_fake_results(
        self,
        patents: List[Patent],
        date_from=None,
        date_to=None,
    ) -> List[Patent]:
        """
        Run PatentsViewClient.search() with Google stubbed to return given patents.
        """
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            # Return the test patents as raw dicts for the first sub-query only
            return [p.model_dump() for p in patents], False

        with patch.object(client, "_google_search_one", side_effect=fake_google_one):
            with patch.object(client, "_search_local_corpus", return_value=[]):
                result = await client.search(
                    query="test query",
                    date_from=date_from,
                    date_to=date_to,
                    size=20,
                )
        return result

    async def test_date_from_filters_old_patents(self):
        patents = [
            _make_patent(patent_id="US1", grant_date="2010-01-01"),
            _make_patent(patent_id="US2", grant_date="2022-06-15"),
        ]
        result = await self._run_search_with_fake_results(patents, date_from="2015-01-01")
        ids = [p.patent_id for p in result]
        self.assertNotIn("US1", ids)
        self.assertIn("US2", ids)

    async def test_date_to_filters_new_patents(self):
        patents = [
            _make_patent(patent_id="US1", grant_date="2010-01-01"),
            _make_patent(patent_id="US2", grant_date="2022-06-15"),
        ]
        result = await self._run_search_with_fake_results(patents, date_to="2015-01-01")
        ids = [p.patent_id for p in result]
        self.assertIn("US1", ids)
        self.assertNotIn("US2", ids)

    async def test_patent_with_no_grant_date_excluded_when_date_filter_set(self):
        patents = [
            _make_patent(patent_id="US1", grant_date=None),
            _make_patent(patent_id="US2", grant_date="2022-06-15"),
        ]
        result = await self._run_search_with_fake_results(patents, date_from="2015-01-01")
        ids = [p.patent_id for p in result]
        self.assertNotIn("US1", ids)
        self.assertIn("US2", ids)

    async def test_no_date_filter_returns_all(self):
        patents = [
            _make_patent(patent_id="US1", grant_date="2010-01-01"),
            _make_patent(patent_id="US2", grant_date="2022-06-15"),
            _make_patent(patent_id="US3", grant_date=None),
        ]
        result = await self._run_search_with_fake_results(patents)
        self.assertEqual(len(result), 3)


# ===========================================================================
# Deduplication tests
# ===========================================================================


class TestDeduplication(unittest.IsolatedAsyncioTestCase):
    """Tests that duplicates across sub-queries are removed."""

    async def test_duplicate_ids_deduplicated(self):
        """
        Simulate two sub-queries returning the same patent; only one should
        appear in the final result.
        """
        p = _make_patent(patent_id="US10000001B2", title="Traffic signal CNN")
        call_count = 0

        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            nonlocal call_count
            call_count += 1
            # Both sub-queries return the same patent
            return [p.model_dump()], False

        with patch.object(client, "_google_search_one", side_effect=fake_google_one):
            with patch.object(client, "_search_local_corpus", return_value=[]):
                result = await client.search(
                    query=(
                        "traffic signal convolutional neural network computer vision "
                        "vehicle density adaptive signal"
                    ),
                    size=10,
                )

        ids = [r.patent_id for r in result]
        self.assertEqual(ids.count("US10000001B2"), 1)

    async def test_local_corpus_does_not_duplicate_live(self):
        p = _make_patent(patent_id="US20240001", title="Traffic signal CNN")
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            return [p.model_dump()], False

        def fake_local(query, size):
            # Local corpus has the same patent
            return [p]

        with patch.object(client, "_google_search_one", side_effect=fake_google_one):
            with patch.object(client, "_search_local_corpus", side_effect=fake_local):
                result = await client.search(query="traffic signal control", size=10)

        ids = [r.patent_id for r in result]
        self.assertEqual(ids.count("US20240001"), 1)

    async def test_local_corpus_supplements_live(self):
        live = _make_patent(patent_id="US1", title="Live patent")
        local = _make_patent(patent_id="US2", title="Local patent")
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            return [live.model_dump()], False

        def fake_local(query, size):
            return [local]

        with patch.object(client, "_google_search_one", side_effect=fake_google_one):
            with patch.object(client, "_search_local_corpus", side_effect=fake_local):
                result = await client.search(query="traffic signal", size=10)

        ids = [r.patent_id for r in result]
        self.assertIn("US1", ids)
        self.assertIn("US2", ids)


# ===========================================================================
# Rate-limit handling tests
# ===========================================================================


class TestRateLimitHandling(unittest.IsolatedAsyncioTestCase):
    """Tests that rate-limit events stop further Google requests."""

    async def test_rate_limit_stops_further_requests(self):
        call_count = 0
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            nonlocal call_count
            call_count += 1
            # Simulate rate-limit on the very first call
            return [], True

        with patch.object(client, "_google_search_one", side_effect=fake_google_one):
            with patch.object(client, "_search_local_corpus", return_value=[]):
                await client.search(
                    query=(
                        "traffic signal convolutional neural network computer vision"
                    ),
                    size=5,
                )

        # Must have stopped after the first rate-limited call
        self.assertEqual(call_count, 1)

    async def test_partial_results_returned_after_rate_limit(self):
        results_by_call = [
            ([_make_patent(patent_id="US1").model_dump()], False),
            ([], True),  # rate-limited on second call
        ]
        call_iter = iter(results_by_call)
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            return next(call_iter)

        with patch.object(client, "_google_search_one", side_effect=fake_google_one):
            with patch.object(client, "_search_local_corpus", return_value=[]):
                result = await client.search(
                    query=(
                        "traffic signal convolutional neural network computer vision"
                    ),
                    size=5,
                )

        # Should still return the one result from the first successful call
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].patent_id, "US1")


# ===========================================================================
# CPC handling tests
# ===========================================================================


class TestCpcHandling(unittest.IsolatedAsyncioTestCase):
    """Tests for CPC filtering of local corpus results."""

    async def test_local_corpus_cpc_filtered(self):
        """Local corpus results with wrong CPC must be excluded when cpc_class is set."""
        wrong_cpc = _make_patent(patent_id="LC1", cpc_codes=["H04L29/06"])
        right_cpc = _make_patent(patent_id="LC2", cpc_codes=["G06N3/04"])
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            return [], False

        def fake_local(query, size):
            return [wrong_cpc, right_cpc]

        with patch.object(client, "_google_search_one", side_effect=fake_google_one):
            with patch.object(client, "_search_local_corpus", side_effect=fake_local):
                result = await client.search(
                    query="neural network traffic", cpc_class="G06N", size=10
                )

        ids = [r.patent_id for r in result]
        self.assertNotIn("LC1", ids)
        self.assertIn("LC2", ids)



# ===========================================================================
# Concept-Coverage Relevance Scorer tests (new scorer)
# ===========================================================================


TRAFFIC_QUERY = (
    "A smart traffic signal control system using computer vision "
    "to detect vehicle density in real time, with a convolutional "
    "neural network running on an edge device and adaptive signal "
    "timing based on the detected traffic conditions."
)


class TestBuildConceptGroups(unittest.TestCase):
    """Tests for _build_concept_groups."""

    def test_traffic_query_has_domain_groups(self):
        groups = PatentsViewClient._build_concept_groups(TRAFFIC_QUERY)
        labels = [g["label"] for g in groups]
        domain_labels = " ".join(labels)
        self.assertIn("traffic signal", domain_labels)
        self.assertIn("vehicle density", domain_labels)
        self.assertIn("adaptive signal", domain_labels)

    def test_cnn_phrase_is_generic(self):
        groups = PatentsViewClient._build_concept_groups(TRAFFIC_QUERY)
        cnn_groups = [g for g in groups if "convolutional neural network" in " ".join(g["terms"])]
        self.assertTrue(len(cnn_groups) > 0)
        for g in cnn_groups:
            self.assertTrue(g["generic"], f"CNN group not marked generic: {g}")

    def test_domain_groups_have_higher_weight_than_generic(self):
        groups = PatentsViewClient._build_concept_groups(TRAFFIC_QUERY)
        domain_weights  = [g["weight"] for g in groups if not g["generic"]]
        generic_weights = [g["weight"] for g in groups if g["generic"]]
        self.assertTrue(len(domain_weights) > 0)
        self.assertTrue(len(generic_weights) > 0)
        self.assertGreater(min(domain_weights), max(generic_weights))

    def test_plain_query_produces_groups(self):
        query = "Wireless power transfer system for implantable medical devices using resonant coupling"
        groups = PatentsViewClient._build_concept_groups(query)
        self.assertTrue(len(groups) > 0)

    def test_all_generic_query_has_no_high_weight_phrase_groups(self):
        query = "convolutional neural network deep learning image classification"
        groups = PatentsViewClient._build_concept_groups(query)
        phrase_groups = [g for g in groups if not g["generic"] and g["weight"] >= 2.5]
        self.assertEqual(len(phrase_groups), 0,
                         "No high-weight domain phrase groups expected for a pure-ML query")


class TestConceptCoverageScore(unittest.TestCase):
    """Tests for _score_concept_coverage against the 7 required scenarios."""

    def _score(self, patent, query=TRAFFIC_QUERY):
        return PatentsViewClient._score_concept_coverage(patent, query, [], [])

    # Required test 1: traffic-signal + vehicle-density ranks above generic CNN
    def test_traffic_signal_ranks_above_generic_cnn(self):
        traffic_patent = _make_patent(
            patent_id="TRAFFIC1",
            title="Adaptive traffic signal control using vehicle density estimation",
            abstract=(
                "A system for adaptive signal timing at intersections using "
                "computer vision to measure vehicle density and traffic flow "
                "from roadside cameras. Signal timing is dynamically adjusted "
                "based on detected congestion."
            ),
        )
        cnn_only_patent = _make_patent(
            patent_id="CNN1",
            title="Convolutional neural network model pruning for edge inference",
            abstract=(
                "A method for pruning convolutional neural network weights to "
                "reduce computation in CNN inference. The pruned model is deployed "
                "on an edge device. No traffic or vehicle content."
            ),
        )
        traffic_score, traffic_groups = self._score(traffic_patent)
        cnn_score,     cnn_groups     = self._score(cnn_only_patent)
        self.assertGreater(
            traffic_score, cnn_score,
            f"Expected traffic ({traffic_score:.2f}, {traffic_groups}) "
            f"> CNN-only ({cnn_score:.2f}, {cnn_groups})"
        )

    # Required test 2: vehicle-detection + traffic ranks above generic image-classification
    def test_vehicle_detection_ranks_above_image_classification(self):
        vehicle_patent = _make_patent(
            patent_id="VD1",
            title="Vehicle detection and traffic flow estimation at intersections",
            abstract=(
                "Computer vision algorithms detect and count vehicles at "
                "road intersections. Traffic flow and vehicle density are "
                "estimated per lane. Results feed an adaptive traffic control system."
            ),
        )
        imgcls_patent = _make_patent(
            patent_id="IMG1",
            title="Weakly supervised fine-grained image classification method",
            abstract=(
                "A weakly supervised neural network method for fine-grained "
                "image classification using a convolutional neural network "
                "trained on weakly labelled data."
            ),
        )
        vd_score, vd_groups   = self._score(vehicle_patent)
        img_score, img_groups = self._score(imgcls_patent)
        self.assertGreater(
            vd_score, img_score,
            f"Vehicle+traffic ({vd_score:.2f}, {vd_groups}) should beat "
            f"image-cls ({img_score:.2f}, {img_groups})"
        )

    # Required test 3: multi-concept bonus for patents covering many groups
    def test_multi_concept_bonus(self):
        multi = _make_patent(
            patent_id="MULTI1",
            title="Smart traffic signal control with adaptive timing based on vehicle density",
            abstract=(
                "Traffic signal control uses vehicle density detection from cameras. "
                "A convolutional neural network on an edge device enables real-time "
                "vehicle detection. Adaptive signal timing optimises traffic flow "
                "at congested intersections."
            ),
        )
        single_domain = _make_patent(
            patent_id="SINGLE1",
            title="Adaptive traffic signal controller",
            abstract="Controls traffic signals at intersections.",
        )
        multi_score,  multi_groups  = self._score(multi)
        single_score, single_groups = self._score(single_domain)
        self.assertGreater(
            multi_score, single_score,
            f"Multi-concept ({multi_score:.2f}) should beat single ({single_score:.2f})"
        )

    # Required test 4: generic words alone do not dominate ranking
    def test_generic_words_do_not_dominate(self):
        generic_filler = _make_patent(
            patent_id="GEN1",
            title="System and method for a device",
            abstract=(
                "A system comprising a method implemented on a device. "
                "The system uses a model for processing data. "
                "The technology provides output based on input features."
            ),
        )
        domain_patent = _make_patent(
            patent_id="DOMAIN1",
            title="Traffic signal control at road intersections",
            abstract="Adaptive traffic control based on vehicle density measurements.",
        )
        gen_score, gen_groups = self._score(generic_filler)
        dom_score, dom_groups = self._score(domain_patent)
        self.assertGreater(
            dom_score, gen_score,
            f"Domain ({dom_score:.2f}) should beat filler ({gen_score:.2f})"
        )

    # Required test 5: missing claims do not discard a relevant patent
    def test_missing_claims_does_not_discard_relevant_patent(self):
        no_claims = _make_patent(
            patent_id="NOCLAIMS1",
            title="Adaptive traffic signal control with vehicle density estimation",
            abstract="Traffic flow managed via vehicle detection and signal timing.",
            claims="",
        )
        score, groups = self._score(no_claims)
        self.assertGreater(score, 0.0,
                           f"Relevant no-claims patent should score > 0; got {score}")
        self.assertTrue(len(groups) > 0, "Should match at least one concept group")

    # Required test 6: duplicate IDs are still removed after deduplication
    def test_canonical_id_deduplication_works(self):
        cid1 = PatentsViewClient._canonical_id("US12387505B2")
        cid2 = PatentsViewClient._canonical_id("US12387505B2")
        self.assertEqual(cid1, cid2)
        cid3 = PatentsViewClient._canonical_id("us 12387505 b2")
        self.assertEqual(cid1, cid3)

    # Required test 7: max_results limits the final returned list
    def test_max_results_limits_output(self):
        import asyncio
        from unittest.mock import patch

        patents = [
            _make_patent(
                patent_id=f"US{i:08d}",
                title=f"Adaptive traffic signal control patent {i}",
                abstract="Vehicle density detection and traffic signal timing.",
            )
            for i in range(10)
        ]
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            return [p.model_dump() for p in patents], False

        async def run():
            with patch.object(client, "_google_search_one", side_effect=fake_google_one):
                with patch.object(client, "_search_local_corpus", return_value=[]):
                    return await client.search(query=TRAFFIC_QUERY, size=3)

        result = asyncio.run(run())
        self.assertLessEqual(len(result), 3,
                             f"max_results=3 should limit output; got {len(result)}")


# ===========================================================================
# End-to-end ranking integration test (simulated candidates pool)
# ===========================================================================


class TestConceptCoverageRankingIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Mix domain-relevant and CNN-only patents in the candidate pool and verify
    that concept-coverage scorer puts domain patents at the top.
    """

    MIXED_PATENTS = [
        _make_patent(
            patent_id="HIGH1",
            title="Smart traffic signal control with adaptive timing based on vehicle density",
            abstract=(
                "Computer vision cameras capture real-time video at road intersections. "
                "A vehicle detection module counts vehicles per lane to estimate traffic density. "
                "An adaptive signal timing algorithm adjusts green/red phases based on detected "
                "congestion to minimise intersection delay."
            ),
        ),
        _make_patent(
            patent_id="HIGH2",
            title="Vehicle detection and traffic flow estimation using roadside cameras",
            abstract=(
                "A computer vision system detects and tracks vehicles at intersections. "
                "Traffic flow rates and vehicle density are estimated per lane. "
                "The system integrates with existing traffic signal control infrastructure."
            ),
        ),
        _make_patent(
            patent_id="LOW1",
            title="Convolutional neural network model pruning for edge inference optimization",
            abstract=(
                "A method for pruning convolutional neural network weights. "
                "The pruned CNN model achieves faster inference on edge devices. "
                "No traffic, signal control, or vehicle content."
            ),
        ),
        _make_patent(
            patent_id="LOW2",
            title="Weakly supervised fine-grained image classification using CNN",
            abstract=(
                "A convolutional neural network is trained with weakly supervised labels "
                "for fine-grained image classification. Deep learning improves accuracy."
            ),
        ),
        _make_patent(
            patent_id="LOW3",
            title="Error correction in convolutional neural networks",
            abstract=(
                "Methods for detecting and correcting errors in CNN computations. "
                "Applies to CNN inference acceleration and model optimisation."
            ),
        ),
    ]

    async def _run_search(self, patents, size=5):
        client = PatentsViewClient()

        async def fake_google_one(sub_query, upstream_size, cpc_class):
            return [p.model_dump() for p in patents], False

        with __import__("unittest.mock", fromlist=["patch"]).patch.object(
            client, "_google_search_one", side_effect=fake_google_one
        ):
            with __import__("unittest.mock", fromlist=["patch"]).patch.object(
                client, "_search_local_corpus", return_value=[]
            ):
                return await client.search(query=TRAFFIC_QUERY, size=size)

    async def test_domain_patents_rank_above_cnn_only(self):
        result = await self._run_search(self.MIXED_PATENTS)
        ids = [p.patent_id for p in result]
        high1_pos = ids.index("HIGH1") if "HIGH1" in ids else 999
        high2_pos = ids.index("HIGH2") if "HIGH2" in ids else 999
        low1_pos  = ids.index("LOW1")  if "LOW1"  in ids else 999
        low2_pos  = ids.index("LOW2")  if "LOW2"  in ids else 999
        low3_pos  = ids.index("LOW3")  if "LOW3"  in ids else 999
        self.assertLess(
            min(high1_pos, high2_pos),
            max(low1_pos, low2_pos, low3_pos),
            f"Domain patents not ranked above CNN-only patents. Order: {ids}"
        )

    async def test_cnn_pruning_ranks_below_domain_patents(self):
        """CNN pruning (LOW1) must rank below both domain patents (HIGH1, HIGH2).
        LOW1 mentions 'edge devices' so can legitimately reach position 3;
        what matters is it never outranks the domain-specific patents."""
        result = await self._run_search(self.MIXED_PATENTS, size=5)
        ids = [p.patent_id for p in result]
        low1_pos  = ids.index("LOW1")  if "LOW1"  in ids else 999
        high1_pos = ids.index("HIGH1") if "HIGH1" in ids else 999
        high2_pos = ids.index("HIGH2") if "HIGH2" in ids else 999
        self.assertGreater(
            low1_pos, high1_pos,
            f"LOW1 (pos {low1_pos}) should rank below HIGH1 (pos {high1_pos}); order: {ids}"
        )
        self.assertGreater(
            low1_pos, high2_pos,
            f"LOW1 (pos {low1_pos}) should rank below HIGH2 (pos {high2_pos}); order: {ids}"
        )

    async def test_image_classification_not_in_top3(self):
        result = await self._run_search(self.MIXED_PATENTS, size=3)
        top3 = [p.patent_id for p in result]
        self.assertNotIn("LOW2", top3,
                         f"Image-classification should NOT be in top-3; got {top3}")

    async def test_score_logging_returns_concept_groups(self):
        domain_patent = self.MIXED_PATENTS[0]
        score, groups = PatentsViewClient._score_concept_coverage(
            domain_patent, TRAFFIC_QUERY, [], []
        )
        self.assertGreater(score, 0.0)
        self.assertTrue(len(groups) > 0,
                        "Domain patent must match at least one concept group")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
