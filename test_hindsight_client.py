"""
Unit tests for memory.hindsight_client.HindsightClient._normalise_memory

These tests cover the core bug fix: Hindsight returns 'text' not 'content'
as the memory body field.

Run with:
    pytest test_hindsight_client.py -v
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from memory.hindsight_client import HindsightClient


class TestNormaliseMemory(unittest.TestCase):
    """Tests for HindsightClient._normalise_memory — the field-mapping fix."""

    def _norm(self, raw: dict) -> dict:
        return HindsightClient._normalise_memory(raw)

    # ------------------------------------------------------------------
    # Core bug fix: 'text' -> 'content'
    # ------------------------------------------------------------------

    def test_text_field_mapped_to_content(self):
        """Hindsight returns 'text'; must appear as 'content' after normalisation."""
        raw = {
            "id": "abc123",
            "text": "PATENT PRIOR ART RESEARCH\nResearch Query: traffic signal\nVerdict: ACCEPTED",
            "type": "experience",
            "context": "patent-prior-art",
            "document_id": "patent-US12345678",
        }
        result = self._norm(raw)
        self.assertEqual(
            result["content"],
            "PATENT PRIOR ART RESEARCH\nResearch Query: traffic signal\nVerdict: ACCEPTED",
            "Memory text must be mapped from 'text' -> 'content'",
        )

    def test_content_key_still_works_as_fallback(self):
        """If Hindsight ever uses 'content' directly, it should still work."""
        raw = {"content": "some memory text"}
        result = self._norm(raw)
        self.assertEqual(result["content"], "some memory text")

    def test_text_preferred_over_content(self):
        """'text' takes priority over 'content' when both are present."""
        raw = {"text": "correct text", "content": "wrong content"}
        result = self._norm(raw)
        self.assertEqual(result["content"], "correct text")

    def test_fact_fallback(self):
        raw = {"fact": "This is a fact memory"}
        result = self._norm(raw)
        self.assertEqual(result["content"], "This is a fact memory")

    def test_empty_raw_gives_empty_content(self):
        """A completely empty raw result should produce empty content, not crash."""
        raw = {}
        result = self._norm(raw)
        self.assertEqual(result["content"], "")
        self.assertIsNone(result["score"])
        self.assertEqual(result["metadata"], {})

    # ------------------------------------------------------------------
    # Score handling
    # ------------------------------------------------------------------

    def test_score_none_when_absent(self):
        """Hindsight doesn't expose per-result scores — must be None, not 0."""
        raw = {"text": "some memory"}
        result = self._norm(raw)
        self.assertIsNone(result["score"])

    def test_score_mapped_when_present(self):
        raw = {"text": "some memory", "score": 0.92}
        result = self._norm(raw)
        self.assertAlmostEqual(result["score"], 0.92)

    def test_relevance_score_alias(self):
        raw = {"text": "memory", "relevance_score": 0.75}
        result = self._norm(raw)
        self.assertAlmostEqual(result["score"], 0.75)

    def test_invalid_score_becomes_none(self):
        raw = {"text": "memory", "score": "not-a-number"}
        result = self._norm(raw)
        self.assertIsNone(result["score"])

    # ------------------------------------------------------------------
    # Metadata population
    # ------------------------------------------------------------------

    def test_context_in_metadata(self):
        raw = {"text": "memory", "context": "patent-prior-art"}
        result = self._norm(raw)
        self.assertEqual(result["metadata"]["context"], "patent-prior-art")

    def test_document_id_in_metadata(self):
        raw = {"text": "memory", "document_id": "patent-US12387505B2"}
        result = self._norm(raw)
        self.assertEqual(result["metadata"]["document_id"], "patent-US12387505B2")

    def test_type_in_metadata(self):
        raw = {"text": "memory", "type": "experience"}
        result = self._norm(raw)
        self.assertEqual(result["metadata"]["type"], "experience")

    def test_id_in_metadata(self):
        raw = {"text": "memory", "id": "mem-uuid-001"}
        result = self._norm(raw)
        self.assertEqual(result["metadata"]["id"], "mem-uuid-001")

    def test_entities_in_metadata(self):
        raw = {"text": "memory", "entities": ["traffic signal", "CNN"]}
        result = self._norm(raw)
        self.assertEqual(result["metadata"]["entities"], ["traffic signal", "CNN"])

    def test_unknown_fields_not_in_metadata(self):
        """Random unknown fields should not leak into metadata."""
        raw = {"text": "memory", "unknown_field": "should not appear"}
        result = self._norm(raw)
        self.assertNotIn("unknown_field", result["metadata"])

    def test_none_metadata_fields_excluded(self):
        """Fields with None values must not populate metadata."""
        raw = {"text": "memory", "context": None, "document_id": None}
        result = self._norm(raw)
        self.assertNotIn("context", result["metadata"])
        self.assertNotIn("document_id", result["metadata"])

    def test_empty_metadata_when_no_metadata_fields(self):
        raw = {"text": "memory only, no metadata"}
        result = self._norm(raw)
        self.assertEqual(result["metadata"], {})

    # ------------------------------------------------------------------
    # Full realistic Hindsight response shape
    # ------------------------------------------------------------------

    def test_realistic_hindsight_result(self):
        """Simulate a real Hindsight recall result as described in the docs."""
        raw = {
            "id": "fact-uuid-001",
            "text": (
                "PATENT PRIOR ART RESEARCH\n"
                "Research Query: traffic signal control computer vision\n"
                "Patent ID: US12387505B2\n"
                "Patent Title: Lane violation detection using CNNs\n"
                "Verdict: REJECTED\n"
                "Reasoning: Focuses on lane violation detection, not adaptive signal timing."
            ),
            "type": "experience",
            "context": "patent-prior-art",
            "document_id": "patent-US12387505B2",
            "mentioned_at": "2026-09-28T17:00:00Z",
            "entities": ["US12387505B2", "CNN", "lane violation"],
            "tags": ["patent", "rejected"],
            "chunk_id": "chunk-001",
        }
        result = self._norm(raw)

        # Content must be the full text, not empty
        self.assertIn("PATENT PRIOR ART RESEARCH", result["content"])
        self.assertIn("REJECTED", result["content"])
        self.assertGreater(len(result["content"]), 50)

        # Score is None (Hindsight doesn't expose it in this format)
        self.assertIsNone(result["score"])

        # Metadata must be populated
        self.assertEqual(result["metadata"]["context"], "patent-prior-art")
        self.assertEqual(result["metadata"]["document_id"], "patent-US12387505B2")
        self.assertEqual(result["metadata"]["type"], "experience")
        self.assertEqual(result["metadata"]["id"], "fact-uuid-001")
        self.assertEqual(result["metadata"]["mentioned_at"], "2026-09-28T17:00:00Z")
        self.assertIn("US12387505B2", result["metadata"]["entities"])


class TestRecallResponseParsing(unittest.TestCase):
    """
    Tests that verify the recall() method correctly picks the right top-level
    key from Hindsight's response ('results' preferred over 'memories').
    These use a minimal synchronous simulation.
    """

    def test_normalise_list_from_results_key(self):
        """Simulate parsing 'results' key (Hindsight primary)."""
        data = {
            "results": [
                {"text": "memory one", "type": "experience"},
                {"text": "memory two", "type": "observation"},
            ]
        }
        raw_list = data.get("results") or data.get("memories") or []
        normalised = [HindsightClient._normalise_memory(m) for m in raw_list]

        self.assertEqual(len(normalised), 2)
        self.assertEqual(normalised[0]["content"], "memory one")
        self.assertEqual(normalised[1]["content"], "memory two")

    def test_normalise_falls_back_to_memories_key(self):
        """If 'results' is absent, 'memories' should be used."""
        data = {
            "memories": [
                {"text": "fallback memory"},
            ]
        }
        raw_list = data.get("results") or data.get("memories") or []
        normalised = [HindsightClient._normalise_memory(m) for m in raw_list]

        self.assertEqual(len(normalised), 1)
        self.assertEqual(normalised[0]["content"], "fallback memory")

    def test_empty_response_returns_empty_list(self):
        """Empty response should yield no crash and no results."""
        data = {}
        raw_list = data.get("results") or data.get("memories") or []
        normalised = [HindsightClient._normalise_memory(m) for m in raw_list]
        self.assertEqual(normalised, [])

    def test_old_content_key_still_works(self):
        """
        If a future Hindsight version sends 'content' instead of 'text',
        the normaliser must still work correctly.
        """
        data = {"results": [{"content": "old-style content field"}]}
        raw_list = data.get("results") or []
        result = HindsightClient._normalise_memory(raw_list[0])
        self.assertEqual(result["content"], "old-style content field")


if __name__ == "__main__":
    unittest.main(verbosity=2)
