"""v2 router exposing predict/usage/token endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from hyprl.backtest import runner as backtest_runner
from hyprl.backtest.runner import BacktestConfig, prepare_supercalc_dataset
from hyprl.configs import (
    get_risk_settings,
    get_supersearch_preset,
    load_long_threshold,
    load_short_threshold,
    load_ticker_settings,
)
from hyprl.native.supercalc import (
    native_available as native_engine_available,
    run_backtest_native,
)
from hyprl.risk.manager import RiskConfig
from hyprl.supercalc import _build_signal_series

from hyprl_api.config import get_settings
from hyprl_api.deps import ApiAuthContext, get_token_and_user
from hyprl_api.db import get_db
from hyprl_api.models import ApiToken, User
from hyprl_api.security import (
    format_token,
    generate_token_prefix,
    generate_token_secret,
    hash_token_secret,
)

router = APIRouter(prefix="/v2", tags=["v2"])
settings = get_settings()
SAFE_PRESETS = {"research_minimal", "api_candidate"}


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    period: str = Field("1y", description="Lookback period (e.g., 6mo, 1y)")
    interval: str = Field("1h", description="Data interval (e.g., 1h, 1d)")
    preset: str = Field("api_candidate", description="Preset name", pattern=r"^[A-Za-z0-9_]+$")
    engine: Literal["auto", "native", "python"] = Field("auto")
    initial_balance: float = Field(10_000.0, ge=1_000.0, le=1_000_000.0)
    trailing_stop_activation: float | None = Field(None, ge=0.0, le=3.0)
    trailing_stop_distance: float | None = Field(None, ge=0.0, le=2.0)


class PredictSummary(BaseModel):
    ticker: str
    period: str
    interval: str
    preset: str
    engine_used: Literal["native", "python"]
    final_balance: float
    profit_factor: float | None
    sharpe: float | None
    max_drawdown_pct: float | None
    risk_of_ruin: float | None
    robustness_score: float | None
    pnl_10k: float


class UsageResponse(BaseModel):
    daily_calls: int
    daily_quota: int


class TokenResolveRequest(BaseModel):
    discord_id: str = Field(..., min_length=3, max_length=64)
    label: str | None = Field(default=None, max_length=128)
    plan: str | None = Field(default=None, max_length=32)


class TokenResolveResponse(BaseModel):
    token: str
    user_id: int
    daily_quota: int


@router.post("/predict", response_model=PredictSummary)
def predict_summary(
    payload: PredictRequest,
    auth_ctx: ApiAuthContext = Depends(get_token_and_user(cost=1)),
) -> PredictSummary:
    _ = auth_ctx  # ensure dependency executes even if unused downstream
    preset = payload.preset.strip()
    if preset not in SAFE_PRESETS:
        raise HTTPException(status_code=400, detail={"error": "invalid_preset"})
    try:
        _ = get_supersearch_preset(preset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"error": str(exc)}) from exc

    cfg = _build_backtest_config(payload)
    dataset = prepare_supercalc_dataset(cfg)

    engine_mode = payload.engine
    if engine_mode in {"native", "auto"} and native_engine_available():
        signal, _, _ = _build_signal_series(dataset, cfg)
        native_result = run_backtest_native(dataset.prices, signal, cfg)
        metrics = _summarize_native(native_result)
        engine_used: Literal["native", "python"] = "native"
    else:
        stats = backtest_runner.simulate_from_dataset(dataset, cfg)
        metrics = _summarize_python(stats)
        engine_used = "python"

    final_balance = float(metrics.get("final_balance") or cfg.initial_balance)
    pnl_10k = float(final_balance - cfg.initial_balance)

    return PredictSummary(
        ticker=payload.ticker.upper(),
        period=payload.period,
        interval=payload.interval,
        preset=preset,
        engine_used=engine_used,
        final_balance=final_balance,
        profit_factor=_coerce_optional(metrics.get("profit_factor")),
        sharpe=_coerce_optional(metrics.get("sharpe")),
        max_drawdown_pct=_coerce_optional(metrics.get("max_drawdown_pct")),
        risk_of_ruin=_coerce_optional(metrics.get("risk_of_ruin")),
        robustness_score=_coerce_optional(metrics.get("robustness_score")),
        pnl_10k=pnl_10k,
    )


@router.get("/usage", response_model=UsageResponse)
def usage_summary(auth_ctx: ApiAuthContext = Depends(get_token_and_user(cost=0))) -> UsageResponse:
    return UsageResponse(
        daily_calls=auth_ctx.usage.calls,
        daily_quota=auth_ctx.usage.daily_quota,
    )


@router.post("/token/resolve-discord", response_model=TokenResolveResponse)
def resolve_discord_token(
    payload: TokenResolveRequest,
    discord_secret: str | None = Header(default=None, alias="X-Discord-Secret"),
    db: Session = Depends(get_db),
) -> TokenResolveResponse:
    if discord_secret != settings.discord_registration_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"error": "forbidden"})
    discord_id = payload.discord_id.strip()
    stmt = select(User).where(User.discord_id == discord_id)
    user = db.execute(stmt).scalar_one_or_none()
    if user is None:
        user = User(
            discord_id=discord_id,
            plan=payload.plan or "discord",
            default_daily_quota=settings.default_daily_credits,
        )
        db.add(user)
        db.flush()
    elif payload.plan:
        user.plan = payload.plan

    active_stmt = select(ApiToken).where(ApiToken.user_id == user.id, ApiToken.is_active.is_(True))
    for token in db.execute(active_stmt).scalars():
        token.is_active = False

    token, token_plain = _create_token(db, user, label=payload.label)
    db.commit()
    return TokenResolveResponse(token=token_plain, user_id=user.id, daily_quota=token.daily_quota)


def _create_token(db: Session, user: User, *, label: str | None) -> tuple[ApiToken, str]:
    prefix = _generate_unique_prefix(db)
    secret = generate_token_secret()
    hashed = hash_token_secret(secret)
    token = ApiToken(
        user_id=user.id,
        token_prefix=prefix,
        token_hash=hashed,
        label=label or f"discord:{user.discord_id}",
        tier=user.plan,
        daily_quota=user.default_daily_quota or settings.default_daily_credits,
        rpm_limit=settings.default_max_rpm,
        rpd_limit=settings.default_max_rpd,
        is_active=True,
    )
    db.add(token)
    db.flush()
    token_plain = format_token(prefix, secret)
    return token, token_plain


def _generate_unique_prefix(db: Session) -> str:
    while True:
        candidate = generate_token_prefix()
        stmt = select(ApiToken.token_prefix).where(ApiToken.token_prefix == candidate)
        existing = db.execute(stmt).scalar_one_or_none()
        if existing is None:
            return candidate


def _build_backtest_config(req: PredictRequest) -> BacktestConfig:
    settings_map = load_ticker_settings(req.ticker, req.interval)
    long_threshold = load_long_threshold(settings_map, default=0.6, location=f"{req.ticker}-{req.interval}")
    short_threshold = load_short_threshold(settings_map, default=0.4, location=f"{req.ticker}-{req.interval}")
    risk_settings = get_risk_settings(settings_map)
    risk_pct = min(float(risk_settings.get("risk_pct", 0.02)), 0.02)
    trailing_activation = _normalize_trailing(req.trailing_stop_activation, cap=3.0)
    trailing_distance = _normalize_trailing(req.trailing_stop_distance, cap=2.0)

    risk_cfg = RiskConfig(
        balance=req.initial_balance,
        risk_pct=risk_pct,
        atr_multiplier=float(risk_settings.get("atr_multiplier", 2.0)),
        reward_multiple=float(risk_settings.get("reward_multiple", 2.0)),
        min_position_size=float(risk_settings.get("min_position_size", 1.0)),
        trailing_stop_activation=trailing_activation,
        trailing_stop_distance=trailing_distance,
    )

    return BacktestConfig(
        ticker=req.ticker.upper(),
        period=req.period,
        interval=req.interval,
        initial_balance=req.initial_balance,
        long_threshold=float(long_threshold),
        short_threshold=float(short_threshold),
        risk=risk_cfg,
        enable_trend_filter=bool(settings_map.get("enable_trend_filter", False)),
        sentiment_min=float(settings_map.get("sentiment_min", -1.0)),
        sentiment_max=float(settings_map.get("sentiment_max", 1.0)),
        sentiment_regime=str(settings_map.get("sentiment_regime", "off")),
    )


def _normalize_trailing(value: float | None, cap: float) -> float | None:
    if value is None or value <= 0.0:
        return None
    return float(min(value, cap))


def _summarize_native(result) -> dict[str, float | None]:
    native_metrics = result.native_metrics or {}
    max_dd_pct = None
    if result.max_drawdown is not None:
        max_dd_pct = result.max_drawdown * 100.0
    return {
        "final_balance": float(result.final_balance),
        "profit_factor": native_metrics.get("profit_factor", result.profit_factor),
        "sharpe": native_metrics.get("sharpe_ratio", result.sharpe_ratio),
        "max_drawdown_pct": native_metrics.get("max_drawdown_pct", max_dd_pct),
        "risk_of_ruin": native_metrics.get("risk_of_ruin", result.risk_of_ruin),
        "robustness_score": native_metrics.get("robustness_score", result.robustness_score),
    }


def _summarize_python(stats) -> dict[str, float | None]:
    return {
        "final_balance": float(stats.final_balance),
        "profit_factor": stats.profit_factor,
        "sharpe": stats.sharpe_ratio,
        "max_drawdown_pct": stats.max_drawdown_pct,
        "risk_of_ruin": stats.risk_of_ruin,
        "robustness_score": stats.robustness_score,
    }


def _coerce_optional(value):
    if value is None:
        return None
    return float(value)