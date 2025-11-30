"""HyprL V2 FastAPI skeleton with auth/rate-limit layers."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Tuple

import redis.asyncio as redis_async
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from api import rate_limiter_service
from api.auth import bootstrap_admin
from api.db import init_db
from api.ratelimit import InMemoryRateLimiter
from api.ratelimit_redis import RedisRateLimiter
from api.routers.v2_autorank import router as autorank_router
from api.routers.v2_predict import router as predict_router
from api.routers.v2_sessions import router as sessions_router
from api.routers.v2_tokens import router as tokens_router
from api.routers.v2_usage import router as usage_router
from api.session_manager import session_manager

APP_VERSION_HEADER = "X-HYPRL-Version"
APP_VERSION_VALUE = "2"
logger = logging.getLogger(__name__)


app = FastAPI(title="HyprL V2 API", version="2.0")

_default_limiter = InMemoryRateLimiter()
rate_limiter_service.set_rate_limiter(_default_limiter)
_redis_client: Optional[redis_async.Redis] = None


@app.middleware("http")
async def add_version_header(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers[APP_VERSION_HEADER] = APP_VERSION_VALUE
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict):
        content = detail
    else:
        content = {"detail": detail}
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


@app.get("/health", response_model=dict)
async def healthcheck() -> dict:
    """Simple readiness endpoint."""
    return {"ok": True}


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Placeholder metrics endpoint satisfying monitoring probes."""
    return JSONResponse({"status": "ok"})


app.include_router(tokens_router)
app.include_router(predict_router)
app.include_router(usage_router)
app.include_router(sessions_router)
app.include_router(autorank_router)


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    bootstrap_admin()
    await _init_rate_limiter()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
    await session_manager.shutdown()


async def _init_rate_limiter() -> None:
    global _redis_client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        rate_limiter_service.set_rate_limiter(InMemoryRateLimiter())
        return
    try:
        client = redis_async.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        await client.ping()
    except Exception as exc:  # pragma: no cover - environment guard
        logger.warning("Redis unavailable (%s); falling back to in-memory limiter", exc)
        rate_limiter_service.set_rate_limiter(InMemoryRateLimiter())
        _redis_client = None
        return
    _redis_client = client
    rate_limiter_service.set_rate_limiter(RedisRateLimiter(client))
    logger.info("Redis rate limiter enabled at %s", redis_url)
