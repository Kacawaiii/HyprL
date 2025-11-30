"""Predict endpoint with auth, quotas, rate limiting, and outcome tracking."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from api import repo
from api.auth import AuthContext, require_scopes
from api.db import get_db
from api.ratelimit import (
    RateLimitInfo,
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    apply_rate_limit_headers,
)
from api.rate_limiter_service import get_rate_limiter
from api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictSummaryResponse,
    PredictionOutcomeUpdateRequest,
)
from api.models import Prediction
from core_bridge import service as predict_service

router = APIRouter(prefix="/v2", tags=["predict"])

ENDPOINT_KEY = "v2/predict"


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    request: Request,
    response: Response,
    auth_ctx: AuthContext = Depends(require_scopes({"read:predict"})),
    db: Session = Depends(get_db),
) -> PredictResponse:
    token_key = getattr(request.state, "token_id", None)
    limiter = get_rate_limiter()
    allowed, rate_info = await limiter.check_and_consume(token_key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "rate_limited"},
            headers={
                RATE_LIMIT_LIMIT_HEADER: str(rate_info.limit),
                RATE_LIMIT_REMAINING_HEADER: "0",
            },
        )
    cost = len(payload.symbols)
    try:
        repo.debit_credits(db, auth_ctx.account_id, cost)
    except repo.InsufficientCreditsError:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={"error": "insufficient_credits"},
        ) from None

    repo.record_usage_event(
        db,
        account_id=auth_ctx.account_id,
        token_id=auth_ctx.token_id,
        endpoint=ENDPOINT_KEY,
        cost=cost,
    )
    db.commit()

    apply_rate_limit_headers(response, rate_info)

    results = predict_service.predict_batch(
        symbols=[s.upper() for s in payload.symbols],
        interval=payload.interval,
        features=payload.features,
        threshold=payload.threshold,
        risk_pct=payload.risk_pct,
    )
    for result in results:
        result.setdefault("risk_pct", payload.risk_pct)

    prediction_records = repo.create_predictions(
        db,
        account_id=auth_ctx.account_id,
        token_id=auth_ctx.token_id,
        entries=results,
        default_risk_pct=payload.risk_pct,
    )
    impl = os.environ.get("HYPRL_PREDICT_IMPL", "stub").lower()
    if impl != "real":
        repo.apply_stub_outcomes(prediction_records)

    db.commit()

    meta = {"model": "prob_bridge_v2", "version": 2}
    enriched = [
        _serialize_prediction(record, base_payload=result)
        for record, result in zip(prediction_records, results, strict=False)
    ]
    return PredictResponse(results=enriched, meta=meta)


@router.post("/predict/{prediction_id}/outcome", response_model=PredictResponse)
async def update_prediction_outcome(
    prediction_id: str,
    payload: PredictionOutcomeUpdateRequest,
    auth_ctx: AuthContext = Depends(require_scopes({"read:predict"})),
    db: Session = Depends(get_db),
) -> PredictResponse:
    record = repo.update_prediction_outcome(
        db,
        prediction_id,
        closed=payload.closed,
        pnl=payload.pnl,
        outcome=payload.outcome,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "prediction_not_found"},
        )
    db.commit()
    return PredictResponse(results=[_serialize_prediction(record)], meta={"updated": True})


@router.get("/predict/summary", response_model=PredictSummaryResponse)
async def predict_summary(
    auth_ctx: AuthContext = Depends(require_scopes({"read:predict"})),
    db: Session = Depends(get_db),
) -> PredictSummaryResponse:
    summary = repo.prediction_summary(db)
    return PredictSummaryResponse(**summary)


def _serialize_prediction(record: Prediction, *, base_payload: dict | None = None) -> dict:
    payload = {
        "prediction_id": record.id,
        "symbol": record.symbol,
        "prob_up": record.prob_up,
        "direction": record.direction,
        "threshold": record.threshold,
        "risk_pct": record.risk_pct,
        "tp": record.tp,
        "sl": record.sl,
        "closed": record.closed,
        "outcome": record.outcome,
        "pnl": record.pnl,
        "created_at": record.created_at.isoformat(),
        "closed_at": record.closed_at.isoformat() if record.closed_at else None,
    }
    if base_payload:
        payload = {**base_payload, **payload}
    return payload
