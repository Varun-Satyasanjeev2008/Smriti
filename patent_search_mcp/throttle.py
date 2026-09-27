"""
Token-bucket rate limiter for the PatentsView API.

PatentsView allows 45 requests per minute. This module implements a simple
async token-bucket limiter that is safe for use in a single-process async
server. For multi-process deployments, swap the asyncio.Lock for a Redis-
backed counter.
"""

from __future__ import annotations

import asyncio
import time
import logging

logger = logging.getLogger(__name__)

# PatentsView rate limit: 45 requests per 60 seconds
_RATE_LIMIT_REQUESTS: int = 45
_RATE_LIMIT_WINDOW_SECS: float = 60.0


class AsyncRateLimiter:
    """
    A sliding-window rate limiter using an asyncio.Lock.

    Usage::

        limiter = AsyncRateLimiter(max_calls=45, period=60)
        async with limiter:
            response = await httpx_client.get(...)
    """

    def __init__(self, max_calls: int = _RATE_LIMIT_REQUESTS, period: float = _RATE_LIMIT_WINDOW_SECS) -> None:
        self._max_calls = max_calls
        self._period = period
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncRateLimiter":
        async with self._lock:
            now = time.monotonic()
            # Purge timestamps older than the window
            self._calls = [t for t in self._calls if now - t < self._period]

            if len(self._calls) >= self._max_calls:
                # Wait until the oldest call falls out of the window
                oldest = self._calls[0]
                sleep_for = self._period - (now - oldest) + 0.05  # small buffer
                logger.debug("PatentsView rate limit reached; sleeping %.2fs", sleep_for)
                await asyncio.sleep(sleep_for)
                # Re-purge after sleeping
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._period]

            self._calls.append(time.monotonic())
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


# Module-level singleton — shared by all PatentsView client calls
patentsview_limiter = AsyncRateLimiter(
    max_calls=_RATE_LIMIT_REQUESTS,
    period=_RATE_LIMIT_WINDOW_SECS,
)
