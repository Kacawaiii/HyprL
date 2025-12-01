"""SQLAlchemy models for HyprL public API."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hyprl_api.db import Base


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""

    return datetime.now(tz=UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    discord_id: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    plan: Mapped[str] = mapped_column(String(32), default="free", nullable=False)
    default_daily_quota: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    tokens: Mapped[list["ApiToken"]] = relationship(
        "ApiToken",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class ApiToken(Base):
    __tablename__ = "api_tokens"
    __table_args__ = (UniqueConstraint("token_prefix", name="uq_token_prefix"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tier: Mapped[str] = mapped_column(String(32), default="demo", nullable=False)
    daily_quota: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    rpm_limit: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    rpd_limit: Mapped[int] = mapped_column(Integer, default=500, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="tokens")
    usage_records: Mapped[list["UsageDaily"]] = relationship(
        "UsageDaily",
        back_populates="token",
        cascade="all, delete-orphan",
    )


class UsageDaily(Base):
    __tablename__ = "usage_daily"
    __table_args__ = (UniqueConstraint("token_id", "day", name="uq_usage_token_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_id: Mapped[int] = mapped_column(Integer, ForeignKey("api_tokens.id"), nullable=False, index=True)
    day: Mapped[date] = mapped_column(Date, nullable=False)
    calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    daily_quota: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    token: Mapped[ApiToken] = relationship("ApiToken", back_populates="usage_records")

