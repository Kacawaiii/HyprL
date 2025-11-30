from __future__ import annotations

import pandas as pd
import pytest

from hyprl.portfolio.core import (
    build_portfolio_equity,
    compute_portfolio_stats,
    compute_portfolio_weights,
)


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_compute_weights_equal_three_tickers() -> None:
    equities = {
        "AAA": _series([100, 101, 102]),
        "BBB": _series([200, 202, 204]),
        "CCC": _series([300, 303, 306]),
    }
    weights = compute_portfolio_weights(equities, scheme="equal")
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0
    for value in weights.values():
        assert pytest.approx(value, rel=1e-9) == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_compute_weights_inv_vol_prefers_low_volatility() -> None:
    smooth = _series(list(range(100, 150)))
    choppy = _series([100 + ((-1) ** i) * 5 for i in range(50)])
    equities = {"LOW": smooth, "HIGH": choppy}
    weights = compute_portfolio_weights(equities, scheme="inv_vol", vol_window=20)
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0
    assert weights["LOW"] > weights["HIGH"]


def test_compute_weights_inv_vol_fallback_equal_when_vol_invalid() -> None:
    flat = _series([100.0, 100.0, 100.0])
    equities = {"AAA": flat, "BBB": flat}
    weights = compute_portfolio_weights(equities, scheme="inv_vol")
    assert pytest.approx(weights["AAA"], rel=1e-9) == 0.5
    assert pytest.approx(weights["BBB"], rel=1e-9) == 0.5


def test_compute_weights_single_ticker_is_full_weight() -> None:
    equities = {"AAA": _series([100.0, 101.0, 102.0])}
    weights = compute_portfolio_weights(equities, scheme="inv_vol")
    assert weights == {"AAA": 1.0}


def test_inv_vol_weighting_improves_portfolio_sharpe() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    low_vol = pd.Series(100.0 + 0.3 * (idx - idx[0]).days, index=idx)
    alt = pd.Series([(-1) ** i for i in range(len(idx))], index=idx)
    high_vol = pd.Series(100.0 + 0.3 * (idx - idx[0]).days + 5.0 * alt, index=idx)
    equities = {"LOW": low_vol, "HIGH": high_vol}

    equal_equity = build_portfolio_equity(equities, total_capital=10_000.0)
    equal_stats = compute_portfolio_stats(equal_equity, initial_balance=10_000.0, seed=7, bootstrap_runs=64)

    inv_weights = compute_portfolio_weights(equities, scheme="inv_vol", vol_window=20)
    inv_equity = build_portfolio_equity(equities, total_capital=10_000.0, weights=inv_weights)
    inv_stats = compute_portfolio_stats(inv_equity, initial_balance=10_000.0, seed=7, bootstrap_runs=64)

    assert inv_stats["sharpe"] >= equal_stats["sharpe"] - 1e-9
