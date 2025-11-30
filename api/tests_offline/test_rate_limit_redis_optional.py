import asyncio
import os

import pytest

from api.ratelimit import RateLimitInfo
from api.ratelimit_redis import RedisRateLimiter


def test_rate_limit_redis_optional():
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        pytest.skip("Redis not available")
    try:
        import redis.asyncio as redis_async
    except ImportError:  # pragma: no cover - optional dependency missing
        pytest.skip("Redis client not installed")

    client = redis_async.from_url(redis_url, encoding="utf-8", decode_responses=True)

    try:
        asyncio.run(client.ping())
    except Exception:
        pytest.skip("Redis not reachable")

    async def _run():
        limiter = RedisRateLimiter(client)
        token = "redis_test_token"
        await client.delete(f"hyprl:rl:{token}")
        last_info: RateLimitInfo | None = None
        for _ in range(60):
            allowed, info = await limiter.check_and_consume(token)
            assert allowed
            last_info = info
        assert last_info is not None
        allowed, info = await limiter.check_and_consume(token)
        assert not allowed
        assert info.remaining == 0
        await client.close()

    asyncio.run(_run())
