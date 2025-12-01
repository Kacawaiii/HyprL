from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, status


@dataclass(slots=True)
class ApiToken:
    key: str
    tier: str
    daily_quota: int
    used_today: int
    rpm_limit: int


def _lookup_token(api_key: str) -> Optional[ApiToken]:
    """Temporary in-memory token lookup until DB-backed auth is wired."""

    demo_key = "DEMO-KEY"
    if api_key == demo_key:
        return ApiToken(
            key=demo_key,
            tier="demo",
            daily_quota=100,
            used_today=0,
            rpm_limit=5,
        )
    return None


def get_current_token(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> ApiToken:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
        )
    token = _lookup_token(x_api_key.strip())
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    # TODO: enforce RPM / quota counters once DB available.
    return token
