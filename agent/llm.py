"""
Groq LLM client for patent relevance reasoning.

Uses the OpenAI-compatible endpoint exposed by Groq to evaluate whether
a patent constitutes relevant prior art for a given research query.

Environment variables:
  GROQ_API_KEY   — Required. Your Groq API key.
  GROQ_MODEL     — Optional. Model to use (default: openai/gpt-oss-120b).

Note: llama-3.3-70b-versatile was deprecated by Groq on 2026-08-16.
Current default is openai/gpt-oss-120b (Groq's recommended replacement).
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from openai import AsyncOpenAI, APIError

logger = logging.getLogger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "openai/gpt-oss-120b"

# System prompt that stays constant across all evaluations.
_SYSTEM_PROMPT = """You are an expert patent attorney and prior-art analyst with deep knowledge \
of USPTO patent law. Your task is to evaluate whether a given patent constitutes relevant \
prior art for a research query.

Guidelines:
- Prior art is relevant if it anticipates or makes obvious the claimed invention.
- Consider the patent's title, abstract, and claims carefully.
- Base your verdict ONLY on the information provided.
- Be precise and cite specific claim language or abstract text when available.
- Your response MUST follow the exact JSON format requested."""

_USER_PROMPT_TEMPLATE = """Research Query:
{query}

Patent Under Evaluation:
Patent ID: {patent_id}
Title: {title}
Abstract: {abstract}
Claims (excerpt): {claims}
Assignee: {assignee}
CPC Codes: {cpc_codes}
Filing Date: {filing_date}
Grant Date: {grant_date}
Publication Date: {publication_date}
Record Source: {source}
Earlier Hindsight research for this exact patent (historical context):
{prior_research}

Evaluate whether this patent is relevant prior art for the research query above.
Compare any earlier evaluation with the current query and patent evidence. Treat the earlier evaluation as context, not as a verdict to copy; explain changed conclusions when relevant.

Respond with a JSON object in EXACTLY this format (no markdown fences):
{{
  "verdict": "ACCEPTED" | "REJECTED",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "<2-4 sentences explaining the verdict, citing specific patent content>"
}}"""


def _get_groq_client() -> AsyncOpenAI:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Obtain a key at https://console.groq.com"
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=_GROQ_BASE_URL,
    )


class PatentRelevanceEvaluator:
    """
    Evaluates patent relevance for a research query using the Groq LLM.

    Each call to `evaluate` makes one chat completion request and returns
    a structured verdict dict.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or os.environ.get("GROQ_MODEL", _DEFAULT_MODEL)
        self._groq = _get_groq_client()

    async def evaluate(
        self,
        query: str,
        patent_id: str,
        title: str,
        abstract: str,
        claims: str,
        assignee: Optional[str] = None,
        cpc_codes: Optional[list] = None,
        filing_date: Optional[str] = None,
        grant_date: Optional[str] = None,
        publication_date: Optional[str] = None,
        source: Optional[str] = None,
        prior_research: str = "",
    ) -> dict:
        """
        Ask the LLM to evaluate whether the patent is relevant prior art.

        Returns a dict with keys: verdict, confidence, reasoning.

        Parameters
        ----------
        query:
            The original research / invention description.
        patent_id, title, abstract, claims, assignee, cpc_codes, filing_date, grant_date:
            Patent data fields from PatentsView.
        """
        user_message = _USER_PROMPT_TEMPLATE.format(
            query=query,
            patent_id=patent_id,
            title=title,
            abstract=abstract[:2000],      # cap to avoid token limits
            claims=claims[:1500],
            assignee=assignee or "Unknown",
            cpc_codes=", ".join(cpc_codes or []) or "N/A",
            filing_date=filing_date or "Unknown",
            grant_date=grant_date or "Unknown",
            publication_date=publication_date or "Unknown",
            source=source or "Unknown",
            prior_research=prior_research[:4000] or "No prior evaluation for this patent was recalled.",
        )

        logger.debug(
            "Groq evaluate: patent_id=%s model=%s", patent_id, self._model
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await self._groq.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,   # low temperature for deterministic legal reasoning
                max_tokens=512,
                response_format={"type": "json_object"},
            )
        except APIError as exc:
            # Some Groq-hosted models reject response_format={"type": "json_object"}.
            # Fall back to a plain completion and rely on prompt instructions + manual parsing.
            logger.warning(
                "Groq structured JSON mode failed for model %s (%s); "
                "retrying without response_format",
                self._model, exc,
            )
            response = await self._groq.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )

        raw_content = response.choices[0].message.content or "{}"
        logger.debug("Groq raw response: %s", raw_content)

        # Strip markdown code fences if the model added them despite instructions
        stripped = raw_content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            raw_content = stripped.strip()

        import json
        try:
            result = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.warning(
                "Groq returned non-JSON for patent %s: %s", patent_id, raw_content
            )
            result = {
                "verdict": "REJECTED",
                "confidence": "LOW",
                "reasoning": f"Unable to parse LLM response: {raw_content[:200]}",
            }

        # Normalise verdict to expected values
        verdict = str(result.get("verdict", "REJECTED")).upper()
        if verdict not in ("ACCEPTED", "REJECTED"):
            verdict = "REJECTED"

        return {
            "verdict": verdict,
            "confidence": result.get("confidence", "LOW"),
            "reasoning": result.get("reasoning", "No reasoning provided."),
        }
