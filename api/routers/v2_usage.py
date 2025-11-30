"""Usage endpoint exposing credit balances."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api import repo
from api.auth import AuthContext, require_scopes
from api.db import get_db
from api.schemas import UsageResponse

router = APIRouter(prefix="/v2", tags=["usage"])


@router.get("/usage", response_model=UsageResponse)
async def usage(
    auth_ctx: AuthContext = Depends(require_scopes({"read:usage"})),
    db: Session = Depends(get_db),
) -> UsageResponse:
    account = repo.get_account(db, auth_ctx.account_id)
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail={"error": "unauthorized"}
        )
    return UsageResponse(
        account_id=account.id,
        credits_total=account.credits_total,
        credits_remaining=account.credits_remaining,
        by_endpoint=repo.usage_by_endpoint(db, account.id),
    )
