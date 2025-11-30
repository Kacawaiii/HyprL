"""Persistence helpers for HyprL V2 API."""

from __future__ import annotations

import json
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any, Iterable, Optional, Sequence, Tuple

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from api.models import Account, Prediction, Token, UsageEvent
from api.utils.crypto import hash_token, verify_token

ADMIN_ACCOUNT_ID = "acc_admin"
ADMIN_DEFAULT_CREDITS = 10**9
ADMIN_SCOPES = {"admin:*", "read:predict", "read:usage", "read:session", "write:session"}
AUTORANK_ENDPOINT = "v2/autorank.start"


class InsufficientCreditsError(Exception):
    """Raised when an account lacks credits for an operation."""


class TokenRevokedError(Exception):
    """Raised when attempting to use a revoked token."""


def _serialize_scopes(scopes: Iterable[str]) -> str:
    return json.dumps(sorted(set(scopes)))


def _deserialize_scopes(scopes_str: str) -> set[str]:
    try:
        raw = json.loads(scopes_str)
    except json.JSONDecodeError:
        return set()
    return {str(scope) for scope in raw}


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def ensure_account(
    db: Session,
    account_id: str,
    *,
    credits_total: Optional[int] = None,
    label_defaults: Optional[dict[str, str]] = None,
) -> Account:
    account = db.get(Account, account_id)
    if account:
        return account
    total = credits_total if credits_total is not None else 0
    account = Account(
        id=account_id,
        credits_total=total,
        credits_remaining=total,
    )
    if label_defaults:
        account.email = label_defaults.get("email")
        account.plan = label_defaults.get("plan")
    db.add(account)
    return account


def create_token(
    db: Session,
    *,
    account_id: str,
    scopes: Sequence[str],
    label: str | None,
    credits_total: Optional[int] = None,
    expires_at: Optional[datetime] = None,
    secret_override: str | None = None,
    token_id: str | None = None,
) -> Tuple[Token, str]:
    account = ensure_account(db, account_id, credits_total=credits_total)
    secret = secret_override or secrets.token_urlsafe(32)
    token_identifier = token_id or f"tok_{uuid.uuid4().hex[:12]}"
    token_plain = f"{token_identifier}.{secret}"
    token = Token(
        id=token_identifier,
        account_id=account.id,
        hash=hash_token(secret),
        scopes=_serialize_scopes(scopes),
        label=label,
        expires_at=expires_at,
    )
    db.add(token)
    return token, token_plain


def revoke_token(db: Session, token_id: str) -> bool:
    token = db.get(Token, token_id)
    if not token:
        return False
    token.revoked_at = _utcnow()
    return True


def find_token_by_id(db: Session, token_id: str) -> Optional[Token]:
    return db.get(Token, token_id)


def find_token_by_secret(db: Session, secret: str) -> Optional[Token]:
    stmt = select(Token).where(Token.revoked_at.is_(None))
    for token in db.execute(stmt).scalars():
        if verify_token(secret, token.hash):
            return token
    return None


def ensure_admin_token(db: Session, token_plain: str) -> None:
    token_id, secret = split_token(token_plain, default_token_id="tok_admin")
    ensure_account(
        db,
        ADMIN_ACCOUNT_ID,
        credits_total=ADMIN_DEFAULT_CREDITS,
        label_defaults={"plan": "admin"},
    )
    existing = db.get(Token, token_id)
    hashed = hash_token(secret)
    if existing:
        existing.hash = hashed
        existing.account_id = ADMIN_ACCOUNT_ID
        existing.scopes = _serialize_scopes(ADMIN_SCOPES)
        existing.revoked_at = None
        existing.expires_at = None
        existing.label = existing.label or "admin_bootstrap"
    else:
        token = Token(
            id=token_id,
            account_id=ADMIN_ACCOUNT_ID,
            hash=hashed,
            scopes=_serialize_scopes(ADMIN_SCOPES),
            label="admin_bootstrap",
        )
        db.add(token)


def debit_credits(db: Session, account_id: str, cost: int) -> Account:
    account = db.get(Account, account_id)
    if account is None:
        raise ValueError(f"Account {account_id} not found")
    if account.credits_remaining < cost:
        raise InsufficientCreditsError(f"{account_id} lacks credits")
    account.credits_remaining -= cost
    return account


def debit_for_autorank(
    db: Session,
    *,
    account_id: str,
    token_id: str,
    total_cost: int,
) -> Account:
    account = db.get(Account, account_id)
    if account is None:
        raise ValueError(f"Account {account_id} not found")
    if account.credits_remaining < total_cost:
        raise InsufficientCreditsError(f"{account_id} lacks credits for autorank")
    account.credits_remaining -= total_cost
    record_usage_event(
        db,
        account_id=account_id,
        token_id=token_id,
        endpoint=AUTORANK_ENDPOINT,
        cost=total_cost,
    )
    return account


def record_usage_event(
    db: Session,
    *,
    account_id: str,
    token_id: str,
    endpoint: str,
    cost: int,
) -> UsageEvent:
    event = UsageEvent(
        id=f"usage_{uuid.uuid4().hex}",
        account_id=account_id,
        token_id=token_id,
        endpoint=endpoint,
        cost=cost,
        created_at=_utcnow(),
    )
    db.add(event)
    return event


def create_predictions(
    db: Session,
    *,
    account_id: str,
    token_id: str,
    entries: Sequence[dict[str, Any]],
    default_risk_pct: float,
) -> list[Prediction]:
    records: list[Prediction] = []
    for data in entries:
        record = Prediction(
            id=f"pred_{uuid.uuid4().hex[:12]}",
            account_id=account_id,
            token_id=token_id,
            symbol=str(data.get("symbol", "")).upper(),
            prob_up=float(data.get("prob_up", 0.0) or 0.0),
            direction=str(data.get("direction", "UNKNOWN")).upper(),
            threshold=float(data.get("threshold", 0.0) or 0.0),
            risk_pct=float(data.get("risk_pct", default_risk_pct) or 0.0),
            tp=_coerce_optional_float(data.get("tp")),
            sl=_coerce_optional_float(data.get("sl")),
            closed=False,
        )
        db.add(record)
        records.append(record)
    return records


def apply_stub_outcomes(predictions: Sequence[Prediction]) -> None:
    """Deterministically close predictions for stub mode/testing."""
    for record in predictions:
        if record.closed:
            continue
        outcome = "WIN" if record.prob_up >= 0.55 else "LOSS"
        record.closed = True
        record.closed_at = _utcnow()
        record.outcome = outcome
        record.pnl = 1.0 if outcome == "WIN" else -1.0


def update_prediction_outcome(
    db: Session,
    prediction_id: str,
    *,
    closed: bool,
    pnl: float | None,
    outcome: str | None,
) -> Prediction | None:
    record = db.get(Prediction, prediction_id)
    if record is None:
        return None
    record.closed = closed
    record.closed_at = _utcnow() if closed else None
    record.pnl = pnl
    record.outcome = outcome
    return record


def prediction_summary(db: Session) -> dict[str, float | int | None]:
    total = int(db.scalar(select(func.count(Prediction.id))) or 0)
    closed = int(
        db.scalar(select(func.count()).where(Prediction.closed.is_(True))) or 0
    )
    pnl_total_raw = db.scalar(
        select(func.coalesce(func.sum(Prediction.pnl), 0.0))
    )
    pnl_total = float(pnl_total_raw or 0.0)
    wins = int(
        db.scalar(
            select(func.count()).where(
                Prediction.closed.is_(True),
                func.upper(Prediction.outcome) == "WIN",
            )
        )
        or 0
    )
    winrate = (wins / closed) if closed else None
    avg_pnl = (pnl_total / closed) if closed else None
    return {
        "total_predictions": total,
        "closed_predictions": closed,
        "pending_predictions": total - closed,
        "win_predictions": wins,
        "winrate_real": winrate,
        "avg_pnl": avg_pnl,
        "pnl_total": pnl_total,
    }


def usage_by_endpoint(db: Session, account_id: str) -> dict[str, int]:
    stmt = (
        select(UsageEvent.endpoint, func.sum(UsageEvent.cost))
        .where(UsageEvent.account_id == account_id)
        .group_by(UsageEvent.endpoint)
    )
    return {endpoint: int(total or 0) for endpoint, total in db.execute(stmt)}


def get_account(db: Session, account_id: str) -> Optional[Account]:
    return db.get(Account, account_id)


def split_token(token_plain: str, *, default_token_id: str | None = None) -> tuple[str, str]:
    if "." in token_plain:
        token_id, secret = token_plain.split(".", 1)
        return token_id, secret
    if default_token_id:
        return default_token_id, token_plain
    return f"tok_{uuid.uuid4().hex[:6]}", token_plain


def scopes_for_token(token: Token) -> set[str]:
    return _deserialize_scopes(token.scopes)
