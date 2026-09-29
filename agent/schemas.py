"""
Pydantic models for the agent's REST API (FastAPI layer).

These models define the shape of POST /research request/response payloads.
They are intentionally separate from patent_search_mcp.models so that the
agent API surface can evolve independently.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Incoming request for the POST /research endpoint."""

    query: str = Field(
        ...,
        min_length=5,
        description="The prior-art research question or invention description.",
        examples=["neural network based image classification using convolutional layers"],
    )
    cpc_class: Optional[str] = Field(
        default=None,
        description="Optional CPC classification filter (e.g. 'G06N3/04').",
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Optional lower bound for patent grant date (YYYY-MM-DD).",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Optional upper bound for patent grant date (YYYY-MM-DD).",
    )
    max_patents: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of candidate patents to evaluate (1-20).",
    )


class RecalledContext(BaseModel):
    """A previously stored research memory surfaced by Hindsight recall."""

    content: str = Field(..., description="The stored memory text.")
    score: Optional[float] = Field(
        default=None, description="Relevance score returned by Hindsight."
    )
    metadata: Optional[dict] = Field(
        default=None, description="Metadata attached to the memory."
    )


class PatentFinding(BaseModel):
    """The agent's evaluation of a single candidate patent."""

    patent_id: str
    title: str
    abstract: str
    assignee: Optional[str]
    cpc_codes: List[str]
    filing_date: Optional[str]
    grant_date: Optional[str]
    publication_date: Optional[str] = None
    source: Optional[str] = None
    verdict: str = Field(
        ..., description="ACCEPTED (relevant prior art) or REJECTED (not relevant)."
    )
    confidence: str = Field(..., description="HIGH, MEDIUM, or LOW.")
    reasoning: str = Field(..., description="LLM-generated explanation for the verdict.")


class ResearchResponse(BaseModel):
    """Response returned by POST /research."""

    query: str
    recalled_context: List[RecalledContext] = Field(
        default_factory=list,
        description="Prior research memories retrieved from Hindsight before new search.",
    )
    findings: List[PatentFinding] = Field(
        default_factory=list,
        description="New patent evaluations performed during this request.",
    )
    summary: str = Field(
        default="",
        description="Brief summary of results: how many accepted vs. rejected.",
    )
