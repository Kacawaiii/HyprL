"""SQLAlchemy models for HyprL V2 API."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db import Base


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    plan: Mapped[str | None] = mapped_column(String, nullable=True)
    credits_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    credits_remaining: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    tokens: Mapped[list["Token"]] = relationship(
        "Token", back_populates="account", cascade="all, delete-orphan"
    )
    usage_events: Mapped[list["UsageEvent"]] = relationship(
        "UsageEvent", back_populates="account", cascade="all, delete-orphan"
    )


class Token(Base):
    __tablename__ = "tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id"), nullable=False, index=True
    )
    hash: Mapped[str] = mapped_column(String, nullable=False)
    scopes: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    account: Mapped[Account] = relationship("Account", back_populates="tokens")


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id"), nullable=False, index=True
    )
    token_id: Mapped[str] = mapped_column(
        String, ForeignKey("tokens.id"), nullable=False, index=True
    )
    endpoint: Mapped[str] = mapped_column(String, nullable=False)
    cost: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    account: Mapped[Account] = relationship("Account", back_populates="usage_events")
    token: Mapped[Token] = relationship("Token")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id"), nullable=False, index=True
    )
    token_id: Mapped[str] = mapped_column(
        String, ForeignKey("tokens.id"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String, nullable=False, index=True)
    prob_up: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    risk_pct: Mapped[float] = mapped_column(Float, nullable=False)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    closed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    account: Mapped[Account] = relationship("Account")
    token: Mapped[Token] = relationship("Token")
