"""Administrative token management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from api import repo
from api.auth import AuthContext, require_scopes
from api.db import get_db
from api.schemas import TokenCreateRequest, TokenCreateResponse

router = APIRouter(prefix="/v2", tags=["tokens"])


@router.post("/tokens", response_model=TokenCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_token(
    payload: TokenCreateRequest,
    db: Session = Depends(get_db),
    _: AuthContext = Depends(require_scopes({"admin:*"})),
) -> TokenCreateResponse:
    token, token_plain = repo.create_token(
        db,
        account_id=payload.account_id,
        scopes=payload.scopes,
        label=payload.label,
        credits_total=payload.credits_total,
        expires_at=payload.expires_at,
    )
    db.commit()
    return TokenCreateResponse(
        token_id=token.id,
        token_plain=token_plain,
        scopes=payload.scopes,
    )


@router.delete("/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_token(
    token_id: str,
    db: Session = Depends(get_db),
    _: AuthContext = Depends(require_scopes({"admin:*"})),
) -> Response:
    repo.revoke_token(db, token_id)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
