"""Pure ETH paper-strategy and news-overlay logic.

The module deliberately contains no broker or network code.  Both Alpaca paper
accounts import the same base signal; the news account may only reduce the
resulting exposure.  This keeps the A/B comparison auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EthStrategyConfig:
    """Frozen discovery configuration for the ETH long/cash paper test."""

    trend_windows: tuple[int, ...] = (64, 128, 256)
    momentum_windows: tuple[int, ...] = (30, 90, 180)
    volatility_window: int = 30
    annualization_days: float = 365.25
    target_volatility: float = 0.20
    volatility_floor: float = 0.10
    max_exposure: float = 0.75
    exposure_step: float = 0.05
    min_trade_pct: float = 0.02
    min_trade_usd: float = 250.0
    max_drawdown: float = 0.25
    max_daily_loss: float = 0.05


@dataclass(frozen=True)
class BaseSignal:
    as_of: datetime
    price: float
    trend_score: float
    momentum_score: float
    composite_score: float
    realized_volatility: float
    volatility_scale: float
    raw_exposure: float
    target_exposure: float


@dataclass(frozen=True)
class NewsRisk:
    multiplier: float
    level: str
    reasons: tuple[str, ...]
    article_count: int
    critical_count: int
    policy_count: int
    event_sha256: str


@dataclass(frozen=True)
class RebalancePlan:
    action: str
    target_value: float
    current_value: float
    delta_value: float
    order_notional: float = 0.0
    order_qty: float = 0.0
    reason: str = ""


CRITICAL_PHRASES = (
    "hack",
    "hacked",
    "hacking",
    "exploit",
    "exploited",
    "vulnerability",
    "security breach",
    "drained",
    "stolen",
    "theft",
    "outage",
    "network halt",
    "chain halt",
    "reorg",
    "insolvent",
    "insolvency",
    "bankruptcy",
    "liquidation cascade",
    "emergency shutdown",
)

POLICY_PHRASES = (
    "ban",
    "bans",
    "banned",
    "crackdown",
    "lawsuit",
    "sues",
    "charges",
    "charged",
    "investigation",
    "probe",
    "sanction",
    "seizure",
    "rejects",
    "rejected",
    "delist",
    "restrict",
)

POLITICAL_ENTITIES = (
    "trump",
    "white house",
    "federal reserve",
    "the fed",
    "treasury",
    "sec",
    "cftc",
)

POLITICAL_RISK_PHRASES = (
    "tariff",
    "trade war",
    "sanction",
    "ban",
    "crackdown",
    "rate hike",
    "higher rates",
    "emergency",
    "war",
)

NEGATING_PHRASES = (
    "not hacked",
    "no hack",
    "no exploit",
    "denies hack",
    "denies breach",
    "dismisses report",
    "false report",
)

CRYPTO_SYMBOLS = frozenset(("ETHUSD", "BTCUSD"))


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = pd.Timestamp(value).to_pydatetime()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_symbol(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _article_symbols(article: Mapping[str, object]) -> set[str]:
    raw = article.get("symbols") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {_normalize_symbol(value) for value in raw}


def _contains(text: str, phrases: Sequence[str]) -> bool:
    return any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
            text,
        )
        is not None
        for phrase in phrases
    )


def quantize_exposure(
    exposure: float,
    *,
    step: float,
    maximum: float,
) -> float:
    if not math.isfinite(exposure):
        return 0.0
    clipped = min(maximum, max(0.0, exposure))
    quantized = math.floor((clipped / step) + 0.5) * step
    return round(min(maximum, max(0.0, quantized)), 10)


def compute_base_signal(
    closes: pd.Series,
    *,
    as_of: datetime | None = None,
    config: EthStrategyConfig | None = None,
) -> BaseSignal:
    """Return the frozen long/cash target using completed daily closes only."""

    cfg = config or EthStrategyConfig()
    values = pd.Series(closes, copy=True).astype(float).dropna()
    if not values.index.is_monotonic_increasing:
        values = values.sort_index()
    if values.index.has_duplicates:
        values = values[~values.index.duplicated(keep="last")]

    required = max(
        max(cfg.trend_windows),
        max(cfg.momentum_windows) + 1,
        cfg.volatility_window + 1,
    )
    if len(values) < required:
        raise ValueError(f"need at least {required} completed daily closes")
    if (values <= 0).any():
        raise ValueError("ETH closes must be positive")

    price = float(values.iloc[-1])
    trend_flags = [
        float(price > float(values.rolling(window).mean().iloc[-1]))
        for window in cfg.trend_windows
    ]
    momentum_flags = [
        float(price / float(values.iloc[-window - 1]) - 1.0 > 0.0)
        for window in cfg.momentum_windows
    ]
    trend_score = float(np.mean(trend_flags))
    momentum_score = float(np.mean(momentum_flags))
    composite = float(np.mean(trend_flags + momentum_flags))

    returns = values.pct_change()
    realized_volatility = float(
        returns.iloc[-cfg.volatility_window :].std(ddof=0)
        * math.sqrt(cfg.annualization_days)
    )
    if not math.isfinite(realized_volatility) or realized_volatility <= 0:
        volatility_scale = 0.0
    else:
        volatility_scale = min(
            1.0,
            cfg.target_volatility
            / max(realized_volatility, cfg.volatility_floor),
        )
    raw_exposure = min(
        cfg.max_exposure,
        max(0.0, composite * volatility_scale),
    )
    target = quantize_exposure(
        raw_exposure,
        step=cfg.exposure_step,
        maximum=cfg.max_exposure,
    )

    signal_time = as_of
    if signal_time is None:
        try:
            signal_time = _utc(values.index[-1])
        except Exception:
            signal_time = datetime.now(timezone.utc)

    return BaseSignal(
        as_of=_utc(signal_time),
        price=price,
        trend_score=trend_score,
        momentum_score=momentum_score,
        composite_score=composite,
        realized_volatility=realized_volatility,
        volatility_scale=volatility_scale,
        raw_exposure=raw_exposure,
        target_exposure=target,
    )


def apply_account_risk_caps(
    target_exposure: float,
    *,
    current_equity: float,
    peak_equity: float,
    previous_day_equity: float,
    config: EthStrategyConfig | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Only reduce exposure when account-level loss limits are breached."""

    cfg = config or EthStrategyConfig()
    reasons: list[str] = []
    target = max(0.0, min(cfg.max_exposure, float(target_exposure)))

    if peak_equity > 0:
        drawdown = current_equity / peak_equity - 1.0
        if drawdown <= -cfg.max_drawdown:
            target = 0.0
            reasons.append("max_drawdown")
    if previous_day_equity > 0:
        daily_return = current_equity / previous_day_equity - 1.0
        if daily_return <= -cfg.max_daily_loss:
            target = 0.0
            reasons.append("daily_loss")
    return target, tuple(reasons)


def evaluate_news_risk(
    articles: Iterable[Mapping[str, object]],
    *,
    now: datetime,
    return_1h: float,
    return_6h: float,
    lookback: timedelta = timedelta(hours=12),
) -> NewsRisk:
    """Classify recent tagged headlines; news can never increase exposure."""

    reference = _utc(now)
    cutoff = reference - lookback
    unique: dict[str, Mapping[str, object]] = {}
    for article in articles:
        try:
            created = _utc(article.get("created_at") or article.get("ts"))
        except Exception:
            continue
        if created < cutoff or created > reference:
            continue
        if not (_article_symbols(article) & CRYPTO_SYMBOLS):
            continue
        identifier = str(article.get("id") or "")
        if not identifier:
            identifier = hashlib.sha256(
                f"{created.isoformat()}:{article.get('headline', '')}".encode()
            ).hexdigest()
        unique[identifier] = article

    critical_ids: list[str] = []
    policy_ids: list[str] = []
    for identifier, article in unique.items():
        text = " ".join(
            (
                str(article.get("headline") or ""),
                str(article.get("summary") or ""),
            )
        ).lower()
        negated = _contains(text, NEGATING_PHRASES)
        critical = not negated and _contains(text, CRITICAL_PHRASES)
        policy = not negated and _contains(text, POLICY_PHRASES)
        political_risk = (
            _contains(text, POLITICAL_ENTITIES)
            and _contains(text, POLITICAL_RISK_PHRASES)
        )
        if critical:
            critical_ids.append(identifier)
        if policy or political_risk:
            policy_ids.append(identifier)

    market_stress = return_1h <= -0.015 or return_6h <= -0.03
    reasons: list[str] = []
    multiplier = 1.0
    level = "normal"

    if critical_ids:
        multiplier = 0.0 if market_stress else 0.5
        level = "critical_confirmed" if market_stress else "critical_unconfirmed"
        reasons.append("critical_headline")
    elif policy_ids and market_stress:
        multiplier = 0.5
        level = "policy_confirmed"
        reasons.append("policy_headline")
    elif len(unique) >= 4 and market_stress:
        multiplier = 0.5
        level = "attention_confirmed"
        reasons.append("attention_burst")

    if market_stress:
        reasons.append("market_stress")

    event_ids = sorted(unique)
    digest = hashlib.sha256("\n".join(event_ids).encode()).hexdigest()
    return NewsRisk(
        multiplier=multiplier,
        level=level,
        reasons=tuple(reasons),
        article_count=len(unique),
        critical_count=len(critical_ids),
        policy_count=len(policy_ids),
        event_sha256=digest,
    )


def plan_rebalance(
    *,
    equity: float,
    target_exposure: float,
    current_qty: float,
    current_price: float,
    available_cash: float,
    config: EthStrategyConfig | None = None,
) -> RebalancePlan:
    """Create a bounded spot-only rebalance plan."""

    cfg = config or EthStrategyConfig()
    if equity <= 0 or current_price <= 0 or current_qty < 0:
        raise ValueError("invalid account or position values")
    exposure = min(cfg.max_exposure, max(0.0, target_exposure))
    target_value = equity * exposure
    current_value = current_qty * current_price
    delta = target_value - current_value
    threshold = max(cfg.min_trade_usd, cfg.min_trade_pct * equity)

    if abs(delta) < threshold:
        return RebalancePlan(
            action="hold",
            target_value=target_value,
            current_value=current_value,
            delta_value=delta,
            reason="inside_rebalance_band",
        )
    if delta > 0:
        notional = min(delta, max(0.0, available_cash) * 0.98)
        if notional < max(10.0, cfg.min_trade_usd):
            return RebalancePlan(
                action="hold",
                target_value=target_value,
                current_value=current_value,
                delta_value=delta,
                reason="insufficient_cash_or_minimum",
            )
        return RebalancePlan(
            action="buy",
            target_value=target_value,
            current_value=current_value,
            delta_value=delta,
            order_notional=round(notional, 2),
            reason="increase_to_target",
        )

    sell_value = min(abs(delta), current_value)
    qty = min(current_qty, sell_value / current_price)
    if sell_value < 10.0 or qty <= 0:
        return RebalancePlan(
            action="hold",
            target_value=target_value,
            current_value=current_value,
            delta_value=delta,
            reason="below_sell_minimum",
        )
    return RebalancePlan(
        action="sell",
        target_value=target_value,
        current_value=current_value,
        delta_value=delta,
        order_qty=math.floor(qty * 1_000_000_000) / 1_000_000_000,
        reason="reduce_to_target",
    )
