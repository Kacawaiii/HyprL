"""FastAPI dependencies for token authentication and quota tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from hyprl_api.db import get_db
from hyprl_api.models import ApiToken, UsageDaily, User
from hyprl_api.security import hash_token_secret


@dataclass(slots=True)
class ApiAuthContext:
    user: User
    token: ApiToken
    usage: UsageDaily


def _http_error(status_code: int, code: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": code})


def _extract_token(
    authorization: str | None,
    api_key: str | None,
) -> str | None:
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    if api_key:
        return api_key.strip()
    return None


def _parse_token(raw_token: str) -> tuple[str, str]:
    if "." not in raw_token:
        raise _http_error(status.HTTP_401_UNAUTHORIZED, "invalid_token")
    prefix, secret = raw_token.split(".", 1)
    if not prefix or not secret:
        raise _http_error(status.HTTP_401_UNAUTHORIZED, "invalid_token")
    return prefix, secret


def _load_usage(db: Session, token: ApiToken, *, cost: int) -> UsageDaily:
    today = date.today()
    stmt = select(UsageDaily).where(
        UsageDaily.token_id == token.id,
        UsageDaily.day == today,
    )
    usage = db.execute(stmt).scalar_one_or_none()
    if usage is None:
        usage = UsageDaily(
            token_id=token.id,
            day=today,
            calls=0,
            daily_quota=token.daily_quota,
        )
        db.add(usage)
        db.flush()
    if usage.daily_quota != token.daily_quota:
        usage.daily_quota = token.daily_quota
    if usage.calls + cost > usage.daily_quota:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "quota_exceeded", "daily_quota": usage.daily_quota},
        )
    usage.calls += cost
    token.last_used_at = datetime.now(tz=UTC)
    db.flush()
    return usage


def get_token_and_user(*, cost: int = 1):
    """Return a dependency enforcing token auth + quota updates."""

    async def dependency(
        request: Request,
        authorization: str | None = Header(default=None, alias="Authorization"),
        api_key: str | None = Header(default=None, alias="X-API-Key"),
        db: Session = Depends(get_db),
    ) -> ApiAuthContext:
        raw_token = _extract_token(authorization, api_key)
        if not raw_token:
            raise _http_error(status.HTTP_401_UNAUTHORIZED, "missing_token")
        prefix, secret = _parse_token(raw_token)
        stmt = select(ApiToken).where(ApiToken.token_prefix == prefix)
        token = db.execute(stmt).scalar_one_or_none()
        if token is None or not token.is_active:
            raise _http_error(status.HTTP_401_UNAUTHORIZED, "invalid_token")
        hashed = hash_token_secret(secret)
        if hashed != token.token_hash:
            raise _http_error(status.HTTP_401_UNAUTHORIZED, "invalid_token")
        user = token.user
        if user is None:
            raise _http_error(status.HTTP_401_UNAUTHORIZED, "invalid_token")
        usage = _load_usage(db, token, cost=cost)
        db.commit()
        return ApiAuthContext(user=user, token=token, usage=usage)

    return dependency