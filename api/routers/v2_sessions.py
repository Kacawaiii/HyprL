"""/v2/sessions endpoints handling realtime paper sessions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from api import repo
from api.auth import AuthContext, require_any_scope, require_scopes
from api.db import get_db
from api.ratelimit import (
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    apply_rate_limit_headers,
)
from api.rate_limiter_service import get_rate_limiter
from api.schemas import (
    SessionCounters,
    SessionReportMetrics,
    SessionReportResponse,
    SessionStatusMetrics,
    SessionStatusResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from api.session_manager import (
    SESSION_COST,
    SessionAccessError,
    SessionNotFoundError,
    session_manager,
)

router = APIRouter(prefix="/v2", tags=["sessions"])


async def _enforce_rate_limit(request: Request):
    limiter = get_rate_limiter()
    token_key = getattr(request.state, "token_id", None)
    allowed, info = await limiter.check_and_consume(token_key)
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


@router.post("/sessions", response_model=StartSessionResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    payload: StartSessionRequest,
    request: Request,
    response: Response,
    auth_ctx: AuthContext = Depends(require_scopes({"write:session"})),
    db: Session = Depends(get_db),
) -> StartSessionResponse:
    rate_info = await _enforce_rate_limit(request)
    try:
        result = await session_manager.start_session(
            payload,
            account_id=auth_ctx.account_id,
            token_id=auth_ctx.token_id,
            db=db,
        )
    except repo.InsufficientCreditsError:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={"error": "insufficient_credits"},
        ) from None

    apply_rate_limit_headers(response, rate_info)
    return StartSessionResponse(
        session_id=result["session_id"],
        log_dir=result["session_dir"],
        impl=result["impl"],
        meta={"debited": SESSION_COST},
    )


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def session_status(
    session_id: str,
    request: Request,
    response: Response,
    auth_ctx: AuthContext = Depends(require_any_scope({"read:usage", "read:session"})),
) -> SessionStatusResponse:
    rate_info = await _enforce_rate_limit(request)
    try:
        payload = await session_manager.get_status(
            session_id,
            account_id=auth_ctx.account_id,
            scopes=auth_ctx.scopes,
        )
    except (SessionNotFoundError, SessionAccessError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "session_not_found"}) from None

    counters = SessionCounters(**payload["counters"])
    metrics_data = payload.get("metrics")
    metrics = SessionStatusMetrics(**metrics_data) if metrics_data else None
    apply_rate_limit_headers(response, rate_info)
    return SessionStatusResponse(
        session_id=payload["session_id"],
        status=payload["status"],
        last_event_ts=payload["last_event_ts"],
        counters=counters,
        kill_switch_triggered=payload["kill_switch_triggered"],
        metrics=metrics,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    request: Request,
    response: Response,
    auth_ctx: AuthContext = Depends(require_scopes({"write:session"})),
) -> Response:
    rate_info = await _enforce_rate_limit(request)
    try:
        await session_manager.stop_session(
            session_id,
            account_id=auth_ctx.account_id,
            scopes=auth_ctx.scopes,
        )
    except (SessionNotFoundError, SessionAccessError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "session_not_found"}) from None
    response.status_code = status.HTTP_204_NO_CONTENT
    apply_rate_limit_headers(response, rate_info)
    return response


@router.get("/sessions/{session_id}/report", response_model=SessionReportResponse)
async def session_report(
    session_id: str,
    request: Request,
    response: Response,
    auth_ctx: AuthContext = Depends(require_any_scope({"read:usage", "read:session"})),
) -> SessionReportResponse:
    rate_info = await _enforce_rate_limit(request)
    try:
        payload = await session_manager.build_report(
            session_id,
            account_id=auth_ctx.account_id,
            scopes=auth_ctx.scopes,
        )
    except (SessionNotFoundError, SessionAccessError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "session_not_found"}) from None

    metrics = SessionReportMetrics(**payload["metrics"])
    apply_rate_limit_headers(response, rate_info)
    return SessionReportResponse(
        session_id=payload["session_id"],
        metrics=metrics,
        top_rejections=payload["top_rejections"],
        duration_s=payload["duration_s"],
    )
