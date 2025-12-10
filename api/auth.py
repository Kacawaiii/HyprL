"""Bearer authentication helpers backed by SQLite."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional, Set

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.db import get_db, session_scope
from api import repo
from api.utils.crypto import verify_token
from api.settings import settings

# Admin bootstrap constants
HYPRL_ADMIN_TOKEN_ENV = "HYPRL_ADMIN_TOKEN"
FALLBACK_ADMIN_SECRET = "hyprl_admin_dev_123"


@dataclass
class AuthContext:
    account_id: str
    token_id: str
    scopes: Set[str]
    tier: str = "standard"
    presets: list[str] | None = None
    credits_total: int = 0
    credits_used: int = 0


def _http_error(status_code: int, code: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": code})


def _extract_bearer(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    parts = header_value.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _now() -> datetime:
    return datetime.now(tz=UTC)


async def _authenticate_request(request: Request, db: Session) -> AuthContext:
    header_value = request.headers.get("authorization")
    token_plain = _extract_bearer(header_value)
    if not token_plain:
        raise _http_error(status.HTTP_401_UNAUTHORIZED, "unauthorized")

    # DEV BYPASS: Allow DEMO-KEY for free access as requested
    if settings.allow_free_token and token_plain == settings.demo_key:
        return AuthContext(
            account_id="demo-account",
            token_id="demo-token",
            scopes={"*"},
            tier="free",
            presets=["research_minimal", "api_candidate"],
            credits_total=10_000,
            credits_used=0,
        )

    token_id, secret = repo.split_token(token_plain)
    token = repo.find_token_by_id(db, token_id)
    if token is None:
        token = repo.find_token_by_secret(db, secret)
        if token is None:
            raise _http_error(status.HTTP_401_UNAUTHORIZED, "unauthorized")
        token_id = token.id

    if token.revoked_at is not None:
        raise _http_error(status.HTTP_401_UNAUTHORIZED, "unauthorized")
    if token.expires_at and token.expires_at <= _now():
        raise _http_error(status.HTTP_401_UNAUTHORIZED, "unauthorized")
    if not verify_token(secret, token.hash):
        raise _http_error(status.HTTP_401_UNAUTHORIZED, "unauthorized")

    scopes = repo.scopes_for_token(token)
    account = token.account
    if account is None:
        raise _http_error(status.HTTP_401_UNAUTHORIZED, "unauthorized")

    request.state.account_id = account.id
    request.state.token_id = token_id
    return AuthContext(account_id=account.id, token_id=token_id, scopes=scopes)


def require_scopes(required: set[str]):
    """FastAPI dependency enforcing bearer auth + scope checks."""

    async def dependency(
        request: Request,
        db: Session = Depends(get_db),
    ) -> AuthContext:
        auth_ctx = await _authenticate_request(request, db)
        if not required.issubset(auth_ctx.scopes):
            raise _http_error(status.HTTP_403_FORBIDDEN, "forbidden")
        return auth_ctx

    return dependency


def require_any_scope(options: set[str]):
    """Dependency requiring at least one scope from the provided set."""

    async def dependency(
        request: Request,
        db: Session = Depends(get_db),
    ) -> AuthContext:
        auth_ctx = await _authenticate_request(request, db)
        if options and not (options & auth_ctx.scopes):
            raise _http_error(status.HTTP_403_FORBIDDEN, "forbidden")
        return auth_ctx

    return dependency


def bootstrap_admin(token_plain: Optional[str] = None) -> None:
    """Ensure an admin token exists using the provided or fallback secret."""
    provided = token_plain or os.environ.get(HYPRL_ADMIN_TOKEN_ENV)
    generated = False
    if not provided:
        provided = FALLBACK_ADMIN_SECRET
        generated = True

    normalized = _normalize_admin_token(provided)
    with session_scope() as db:
        repo.ensure_admin_token(db, normalized)

    if generated:
        print(
            "[HyprL] Using fallback admin token hyprl_admin_dev_123 "
            "(development use only)."
        )


def _normalize_admin_token(token_plain: str) -> str:
    if "." in token_plain:
        return token_plain
    return f"tok_admin.{token_plain}"
