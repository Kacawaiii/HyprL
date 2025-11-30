"""Redis-backed rate limiter implementation."""

from __future__ import annotations

import asyncio
import time
from typing import Tuple

import redis.asyncio as redis_async
from redis.exceptions import RedisError, WatchError

from api.ratelimit import RateLimitInfo

KEY_PREFIX = "hyprl:rl:"


class RedisRateLimiter:
    """Token bucket limiter stored in Redis."""

    def __init__(
        self,
        client: redis_async.Redis,
        limit_per_minute: int = 60,
        burst: int = 10,
    ) -> None:
        self.client = client
        self.limit = limit_per_minute
        self.burst = burst
        self.refill_rate = limit_per_minute / 60.0

    async def check_and_consume(self, token_key: str) -> Tuple[bool, RateLimitInfo]:
        if not token_key:
            return False, RateLimitInfo(limit=self.limit, remaining=0)

        key = f"{KEY_PREFIX}{token_key}"
        attempts = 0
        while True:
            attempts += 1
            try:
                async with self.client.pipeline(transaction=True) as pipe:
                    await pipe.watch(key)
                    data = await pipe.hgetall(key)
                    now = time.monotonic()
                    tokens = float(data.get("tokens", self.burst))
                    last_ts = float(data.get("ts", now))
                    elapsed = max(0.0, now - last_ts)
                    tokens = min(self.burst, tokens + elapsed * self.refill_rate)
                    allowed = tokens >= 1.0
                    if allowed:
                        tokens -= 1.0
                    remaining = int(tokens) if allowed else 0
                    pipe.multi()
                    pipe.hset(key, mapping={"tokens": tokens, "ts": now})
                    pipe.expire(key, 120)
                    await pipe.execute()
                    return allowed, RateLimitInfo(limit=self.limit, remaining=remaining)
            except WatchError:
                if attempts >= 5:
                    await asyncio.sleep(0)
                continue
            except RedisError:
                raise
