"""
FastAPI application — HTTP layer for the patent prior-art research agent.

Endpoints:
  GET  /health            — Liveness check
  POST /research          — Run prior-art research pipeline
  GET  /reflect           — Trigger Hindsight reflect for cross-research patterns
  GET  /logs              — Return recent request logs from app Postgres

Run:
  uvicorn agent.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from agent.database import PatentCache, RequestLog, close_db, get_session, init_db
from agent.orchestrator import PriorArtResearchAgent
from agent.schemas import ResearchRequest, ResearchResponse
from memory.hindsight_client import HindsightClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_agent: PriorArtResearchAgent | None = None
_memory: HindsightClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise long-lived resources on startup, clean up on shutdown."""
    global _agent, _memory

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting Patent Prior-Art Research Agent...")

    # Initialise app-level database
    await init_db()

    # Initialise shared agent and memory client
    _agent = PriorArtResearchAgent()
    _memory = HindsightClient()

    logger.info("Agent ready.")
    yield

    # Cleanup
    if _agent:
        await _agent.aclose()
    if _memory:
        await _memory.aclose()
    await close_db()
    logger.info("Agent shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Patent Prior-Art Research Agent",
    description=(
        "An AI agent that searches public patent records, evaluates prior-art relevance "
        "using Groq LLM reasoning, and remembers past research via Hindsight memory."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_agent() -> PriorArtResearchAgent:
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialised yet.",
        )
    return _agent


def _get_memory() -> HindsightClient:
    if _memory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory client not initialised yet.",
        )
    return _memory


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health_check():
    """Liveness probe. Returns 200 if the service is running."""
    return {"status": "ok", "service": "patent-prior-art-agent"}


@app.post(
    "/research",
    response_model=ResearchResponse,
    tags=["research"],
    summary="Run patent prior-art research",
    description=(
        "Accepts a research query, checks Hindsight for prior research, "
        "searches Google Patents and the local development corpus for candidates, evaluates each via Groq LLM, "
        "stores findings in Hindsight, and returns a structured result."
    ),
)
async def research(
    request: ResearchRequest,
    agent: PriorArtResearchAgent = Depends(_get_agent),
    db: AsyncSession = Depends(get_session),
) -> ResearchResponse:
    start = time.monotonic()

    try:
        response = await agent.research(request)
    except Exception as exc:
        logger.exception("Research pipeline failed for query %r", request.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research pipeline error: {exc}",
        ) from exc

    elapsed_ms = (time.monotonic() - start) * 1000

    # Persist audit log to app Postgres
    try:
        accepted = sum(1 for f in response.findings if f.verdict == "ACCEPTED")
        log_entry = RequestLog(
            query=request.query,
            cpc_class=request.cpc_class,
            date_from=request.date_from,
            date_to=request.date_to,
            patents_searched=len(response.findings),
            patents_accepted=accepted,
            patents_rejected=len(response.findings) - accepted,
            recalled_memories=len(response.recalled_context),
            latency_ms=elapsed_ms,
        )
        db.add(log_entry)
        await db.commit()
    except Exception as db_exc:
        logger.warning("Failed to write audit log: %s", db_exc)
        # Non-fatal: don't fail the request because of logging

    return response


@app.get(
    "/reflect",
    tags=["research"],
    summary="Synthesise cross-research patterns",
    description=(
        "Triggers Hindsight's reflect operation to surface patterns across "
        "all accumulated patent research (e.g. recurring rejection reasons)."
    ),
)
async def reflect(
    query: str | None = None,
    memory: HindsightClient = Depends(_get_memory),
):
    """
    Ask Hindsight to synthesise patterns from accumulated research memories.

    Optionally provide a `query` parameter to focus the reflection.
    """
    reflection = await memory.reflect(query=query)
    return {"reflection": reflection or "No patterns found yet."}


@app.get(
    "/logs",
    tags=["ops"],
    summary="Recent request audit logs",
    description="Return the most recent research request logs from the app database.",
)
async def get_logs(
    limit: int = 20,
    db: AsyncSession = Depends(get_session),
):
    """
    Return the most recent request log entries (default 20).
    """
    try:
        stmt = (
            select(RequestLog)
            .order_by(desc(RequestLog.created_at))
            .limit(min(limit, 100))
        )
        result = await db.execute(stmt)
        logs = result.scalars().all()
        return {
            "count": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "query": log.query,
                    "cpc_class": log.cpc_class,
                    "date_from": log.date_from,
                    "date_to": log.date_to,
                    "patents_searched": log.patents_searched,
                    "patents_accepted": log.patents_accepted,
                    "patents_rejected": log.patents_rejected,
                    "recalled_memories": log.recalled_memories,
                    "latency_ms": log.latency_ms,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ],
        }
    except Exception as exc:
        logger.exception("Failed to fetch request logs")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {exc}",
        ) from exc
