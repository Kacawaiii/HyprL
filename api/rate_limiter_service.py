"""Shared accessors for the active rate limiter."""

from __future__ import annotations

from typing import Union

from api.ratelimit import InMemoryRateLimiter
from api.ratelimit_redis import RedisRateLimiter

RateLimiterType = Union[InMemoryRateLimiter, RedisRateLimiter]

_rate_limiter: RateLimiterType = InMemoryRateLimiter()


def get_rate_limiter() -> RateLimiterType:
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiterType) -> None:
    global _rate_limiter
    _rate_limiter = limiter
