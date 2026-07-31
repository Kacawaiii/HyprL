from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.eth_paper.strategy import (
    EthStrategyConfig,
    apply_account_risk_caps,
    compute_base_signal,
    evaluate_news_risk,
    plan_rebalance,
    quantize_exposure,
)


NOW = datetime(2026, 7, 31, 12, tzinfo=timezone.utc)


def close_series(values) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=len(values), freq="D", tz="UTC"),
    )


def article(
    headline: str,
    *,
    hours_ago: int = 1,
    symbols=("ETHUSD",),
    article_id: str = "a1",
) -> dict:
    return {
        "id": article_id,
        "created_at": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "headline": headline,
        "summary": "",
        "symbols": list(symbols),
    }


def test_rising_market_produces_bounded_quantized_long_target() -> None:
    closes = close_series(np.geomspace(1_000.0, 4_000.0, 400))
    signal = compute_base_signal(closes, as_of=NOW)

    assert signal.trend_score == 1.0
    assert signal.momentum_score == 1.0
    assert 0.0 < signal.target_exposure <= EthStrategyConfig().max_exposure
    assert signal.target_exposure * 20 == pytest.approx(
        round(signal.target_exposure * 20)
    )


def test_falling_market_moves_fully_to_cash() -> None:
    closes = close_series(np.geomspace(4_000.0, 500.0, 400))
    signal = compute_base_signal(closes, as_of=NOW)

    assert signal.composite_score == 0.0
    assert signal.target_exposure == 0.0


def test_signal_has_no_future_dependency() -> None:
    base = close_series(np.geomspace(1_000.0, 2_000.0, 420))
    cutoff = 390
    expected = compute_base_signal(base.iloc[:cutoff], as_of=NOW)
    mutated = base.copy()
    mutated.iloc[cutoff:] *= 50
    actual = compute_base_signal(mutated.iloc[:cutoff], as_of=NOW)

    assert actual == expected


def test_signal_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="need at least"):
        compute_base_signal(close_series(np.arange(100.0, 200.0)), as_of=NOW)


def test_exposure_quantization_is_bounded() -> None:
    assert quantize_exposure(0.126, step=0.05, maximum=0.75) == 0.15
    assert quantize_exposure(-1.0, step=0.05, maximum=0.75) == 0.0
    assert quantize_exposure(2.0, step=0.05, maximum=0.75) == 0.75


def test_account_risk_caps_only_reduce_exposure() -> None:
    target, reasons = apply_account_risk_caps(
        0.4,
        current_equity=74_000,
        peak_equity=100_000,
        previous_day_equity=80_000,
    )
    assert target == 0.0
    assert set(reasons) == {"max_drawdown", "daily_loss"}

    target, reasons = apply_account_risk_caps(
        0.4,
        current_equity=101_000,
        peak_equity=101_000,
        previous_day_equity=100_000,
    )
    assert target == 0.4
    assert reasons == ()


def test_critical_news_halves_risk_until_market_confirms() -> None:
    risk = evaluate_news_risk(
        [article("Ethereum protocol hacked in major exploit")],
        now=NOW,
        return_1h=0.002,
        return_6h=-0.005,
    )
    assert risk.multiplier == 0.5
    assert risk.level == "critical_unconfirmed"

    confirmed = evaluate_news_risk(
        [article("Ethereum protocol hacked in major exploit")],
        now=NOW,
        return_1h=-0.02,
        return_6h=-0.04,
    )
    assert confirmed.multiplier == 0.0
    assert confirmed.level == "critical_confirmed"


def test_political_headline_requires_risk_content_and_market_confirmation() -> None:
    positive = evaluate_news_risk(
        [article("Trump praises Ethereum builders")],
        now=NOW,
        return_1h=-0.02,
        return_6h=-0.04,
    )
    assert positive.multiplier == 1.0

    risk = evaluate_news_risk(
        [article("Trump tariff plan triggers crypto trade war fears")],
        now=NOW,
        return_1h=-0.02,
        return_6h=-0.04,
    )
    assert risk.multiplier == 0.5
    assert risk.level == "policy_confirmed"


def test_news_filter_ignores_untagged_future_and_negated_reports() -> None:
    future = article("Ethereum hacked", article_id="future")
    future["created_at"] = (NOW + timedelta(minutes=1)).isoformat()
    risk = evaluate_news_risk(
        [
            article("Ethereum hacked", symbols=("AAPL",), article_id="stock"),
            article("Foundation says Ethereum was not hacked", article_id="negated"),
            article("Ethereum hackathon opens in Paris", article_id="hackathon"),
            future,
        ],
        now=NOW,
        return_1h=-0.02,
        return_6h=-0.04,
    )
    assert risk.article_count == 2
    assert risk.critical_count == 0
    assert risk.multiplier == 1.0


def test_rebalance_plan_is_spot_only_and_respects_cash_and_band() -> None:
    buy = plan_rebalance(
        equity=100_000,
        target_exposure=0.20,
        current_qty=0.0,
        current_price=2_000,
        available_cash=100_000,
    )
    assert buy.action == "buy"
    assert buy.order_notional == 20_000

    hold = plan_rebalance(
        equity=100_000,
        target_exposure=0.20,
        current_qty=9.5,
        current_price=2_000,
        available_cash=80_000,
    )
    assert hold.action == "hold"

    sell = plan_rebalance(
        equity=100_000,
        target_exposure=0.0,
        current_qty=10.0,
        current_price=2_000,
        available_cash=80_000,
    )
    assert sell.action == "sell"
    assert sell.order_qty == 10.0

    with pytest.raises(ValueError):
        plan_rebalance(
            equity=100_000,
            target_exposure=0.2,
            current_qty=-1.0,
            current_price=2_000,
            available_cash=100_000,
        )
