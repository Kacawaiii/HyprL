"""Autorank → sessions orchestration endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from api import repo
from api.auth import AuthContext, require_any_scope, require_scopes
from api.autorank_manager import (
    AutorankJobExistsError,
    AutorankJobNotFoundError,
    AutorankPathError,
    autorank_manager,
)
from api.db import get_db
from api.ratelimit import (
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    apply_rate_limit_headers,
)
from api.rate_limiter_service import get_rate_limiter
from api.schemas import (
    AutorankSessionLaunch,
    AutorankSessionStatus,
    AutorankStartRequest,
    AutorankStartResponse,
    AutorankStatusResponse,
)
from api.session_manager import SessionAccessError, SessionNotFoundError, session_manager

router = APIRouter(prefix="/v2", tags=["autorank"])


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


@router.post("/autorank/start", response_model=AutorankStartResponse, status_code=status.HTTP_201_CREATED)
async def autorank_start(
    payload: AutorankStartRequest,
    request: Request,
    response: Response,
    auth_ctx: AuthContext = Depends(require_scopes({"write:session"})),
    db: Session = Depends(get_db),
) -> AutorankStartResponse:
    rate_info = await _enforce_rate_limit(request)
    try:
        job = await autorank_manager.start_autorank(
            payload,
            account_id=auth_ctx.account_id,
            token_id=auth_ctx.token_id,
            db=db,
        )
    except AutorankJobExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "autorank_exists", "job_id": exc.job_id},
        ) from None
    except AutorankPathError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "invalid_path", "detail": str(exc)}) from None
    except repo.InsufficientCreditsError:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail={"error": "insufficient_credits"}) from None

    apply_rate_limit_headers(response, rate_info)
    sessions_payload = [
        AutorankSessionLaunch(
            rank=record.rank,
            session_id=record.session_id,
            source_csv=record.source_csv,
            config_index=record.config_index,
        )
        for record in job.sessions
    ]
    return AutorankStartResponse(
        autorank_id=job.job_id,
        artifacts_dir=str(job.artifacts_dir),
        autoranked_csv=str(job.autoranked_csv),
        summary_txt=str(job.summary_path),
        sessions=sessions_payload,
        debited_credits=job.debited_credits,
    )


@router.get("/autorank/{job_id}", response_model=AutorankStatusResponse)
async def autorank_status(
    job_id: str,
    request: Request,
    response: Response,
    auth_ctx: AuthContext = Depends(require_any_scope({"read:usage", "read:session"})),
) -> AutorankStatusResponse:
    rate_info = await _enforce_rate_limit(request)
    try:
        job = await autorank_manager.get_status(job_id, account_id=auth_ctx.account_id)
    except AutorankJobNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "autorank_not_found"}) from None

    session_states: list[AutorankSessionStatus] = []
    for record in job.sessions:
        status_label = "pending" if not record.session_id else "unknown"
        if record.session_id:
            try:
                sess_payload = await session_manager.get_status(
                    record.session_id,
                    account_id=auth_ctx.account_id,
                    scopes=auth_ctx.scopes,
                )
                status_label = sess_payload["status"]
            except (SessionNotFoundError, SessionAccessError):
                status_label = "unknown"
        session_states.append(
            AutorankSessionStatus(
                session_id=record.session_id or f"rank_{record.rank}",
                status=status_label,
            )
        )

    apply_rate_limit_headers(response, rate_info)
    return AutorankStatusResponse(
        autorank_id=job.job_id,
        status=job.status,
        sessions=session_states,
        artifacts={
            "autoranked_csv": str(job.autoranked_csv),
            "summary_txt": str(job.summary_path),
        },
    )
