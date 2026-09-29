"""
Shared Pydantic models for patent data.
These are the canonical data structures used across all modules.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Patent(BaseModel):
    """Canonical representation of a USPTO patent record."""

    patent_id: str = Field(..., description="USPTO patent ID (e.g. '10,123,456')")
    title: str = Field(default="", description="Patent title")
    abstract: str = Field(default="", description="Patent abstract")
    claims: str = Field(
        default="",
        description="Full claims text (first claim or concatenated claims)",
    )
    assignee: Optional[str] = Field(
        default=None, description="Primary assignee organization name"
    )
    cpc_codes: List[str] = Field(
        default_factory=list, description="CPC classification codes"
    )
    filing_date: Optional[str] = Field(
        default=None, description="Application filing date (YYYY-MM-DD)"
    )
    grant_date: Optional[str] = Field(
        default=None, description="Patent grant date (YYYY-MM-DD)"
    )
    publication_date: Optional[str] = Field(
        default=None, description="Publication date when no grant date is available (YYYY-MM-DD)"
    )
    source: Optional[str] = Field(
        default=None, description="Origin of the patent record"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patent_id": "11,049,013",
                "title": "Neural network patent classification system",
                "abstract": "A system for classifying patents using neural networks...",
                "claims": "1. A method comprising: ...",
                "assignee": "Acme Corp",
                "cpc_codes": ["G06N3/04", "G06F16/903"],
                "filing_date": "2020-01-15",
                "grant_date": "2021-06-29",
            }
        }
