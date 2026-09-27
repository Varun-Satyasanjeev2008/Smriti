"""
Database module — app-level PostgreSQL with pgvector.

This module provides:
  1. SQLAlchemy async engine / session setup.
  2. ORM models for caching raw PatentsView responses and logging requests.
  3. init_db() to create tables on startup.

This is SEPARATE from Hindsight's internal Postgres; do not share the
connection string or the database instance with Hindsight.

Tables:
  patent_cache  — cached raw API responses keyed by patent_id
  request_log   — per-request audit log
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

from sqlalchemy import String, Text, DateTime, Integer, Float, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)


def _get_database_url() -> str:
    url = os.environ.get("APP_DATABASE_URL", "")
    if not url:
        raise ValueError(
            "APP_DATABASE_URL environment variable is not set.\n"
            "Expected format: postgresql+asyncpg://user:pass@host:port/dbname"
        )
    # Ensure we use the asyncpg driver
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


# ---------------------------------------------------------------------------
# SQLAlchemy setup
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# Created lazily on first import of get_session()
_engine = None
_session_factory = None


def _build_engine():
    global _engine, _session_factory
    if _engine is not None:
        return
    db_url = _get_database_url()
    _engine = create_async_engine(
        db_url,
        echo=os.environ.get("SQL_ECHO", "false").lower() == "true",
        pool_size=5,
        max_overflow=10,
    )
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )


async def get_session() -> AsyncSession:  # type: ignore[return]
    """Dependency / async generator for FastAPI route injection."""
    _build_engine()
    assert _session_factory is not None
    async with _session_factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables (idempotent). Called once at FastAPI startup."""
    _build_engine()
    assert _engine is not None
    async with _engine.begin() as conn:
        # Enable pgvector extension (safe to run if already exists)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised successfully.")


async def close_db() -> None:
    """Dispose the engine. Called at FastAPI shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine disposed.")


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class PatentCache(Base):
    """
    Caches raw PatentsView patent data to reduce repeated API calls.

    The 'raw_json' column stores the full JSON response as text.
    """

    __tablename__ = "patent_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patent_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    assignee: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False)
    cached_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )


class RequestLog(Base):
    """
    Audit log for every POST /research request.

    Records: the query, number of patents found, number accepted, latency.
    """

    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    cpc_class: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    date_from: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    date_to: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    patents_searched: Mapped[int] = mapped_column(Integer, default=0)
    patents_accepted: Mapped[int] = mapped_column(Integer, default=0)
    patents_rejected: Mapped[int] = mapped_column(Integer, default=0)
    recalled_memories: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )
