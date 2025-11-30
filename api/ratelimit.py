"""In-memory token bucket rate limiter per bearer token."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

from fastapi import HTTPException, Request, status

RATE_LIMIT_PER_MINUTE = 60
BURST_CAPACITY = 10
REFILL_RATE_PER_SEC = RATE_LIMIT_PER_MINUTE / 60.0

RATE_LIMIT_LIMIT_HEADER = "X-RateLimit-Limit"
RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"


@dataclass
class RateLimitState:
    tokens: float
    updated_at: float


@dataclass
class RateLimitInfo:
    limit: int
    remaining: int


class InMemoryRateLimiter:
    """Async-friendly token bucket limiter backed by local memory."""

    def __init__(self, limit_per_minute: int = RATE_LIMIT_PER_MINUTE, burst: int = BURST_CAPACITY):
        self.limit = limit_per_minute
        self.burst = burst
        self.refill_rate = limit_per_minute / 60.0
        self._state: Dict[str, RateLimitState] = {}
        self._lock = threading.Lock()

    def _consume(self, token_key: str) -> Tuple[bool, RateLimitInfo]:
        if token_key is None:
            return False, RateLimitInfo(limit=self.limit, remaining=0)
        now = time.monotonic()
        with self._lock:
            state = self._state.get(token_key)
            if state is None:
                state = RateLimitState(tokens=self.burst, updated_at=now)
                self._state[token_key] = state

            elapsed = max(0.0, now - state.updated_at)
            if elapsed > 0:
                state.tokens = min(self.burst, state.tokens + elapsed * self.refill_rate)
                state.updated_at = now

            allowed = state.tokens >= 1.0
            if allowed:
                state.tokens -= 1.0
            remaining = int(state.tokens) if allowed else 0
            return allowed, RateLimitInfo(limit=self.limit, remaining=remaining)

    async def check_and_consume(self, token_key: str) -> Tuple[bool, RateLimitInfo]:
        return self._consume(token_key)

    def check_and_consume_sync(self, token_key: str) -> Tuple[bool, RateLimitInfo]:
        return self._consume(token_key)

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


MEMORY_LIMITER = InMemoryRateLimiter()


def reset_rate_limits() -> None:
    MEMORY_LIMITER.reset()


def enforce_rate_limit(request: Request) -> RateLimitInfo:
    token_hash = getattr(request.state, "token_hash", None)
    allowed, info = MEMORY_LIMITER.check_and_consume_sync(token_hash)
    if not allowed:
        headers = {
            RATE_LIMIT_LIMIT_HEADER: str(info.limit),
            RATE_LIMIT_REMAINING_HEADER: "0",
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "rate_limited"},
            headers=headers,
        )
    return info


def apply_rate_limit_headers(response, info: RateLimitInfo) -> None:
    response.headers[RATE_LIMIT_LIMIT_HEADER] = str(info.limit)
    response.headers[RATE_LIMIT_REMAINING_HEADER] = str(max(info.remaining, 0))
