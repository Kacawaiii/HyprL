"""Synthetic causal/accounting regressions; never contacts a provider."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.research.crypto_hunter.data import features, forward_returns, from_frames, protocol
from scripts.research.crypto_hunter.engine import simulate


def market(days=800):
    index = pd.date_range("2019-01-01", periods=days)
    close = 100 * np.exp(np.arange(days) * 0.003 + np.sin(np.arange(days)) * 0.005)
    frames = {s: pd.DataFrame({"open": close, "high": close * 1.1, "low": close * 0.9,
                               "close": close, "volume": np.full(days, 100000)}, index=index)
              for s in ("BTC-USD", "ALT-USD")}
    return from_frames(frames)


def tiny(prices):
    index = pd.date_range("2020-12-30", periods=len(prices))
    m = from_frames({"BTC-USD": pd.DataFrame(prices, index=index,
                                             columns=["open", "high", "low", "close", "volume"])})
    frame = lambda v: pd.DataFrame(v, index=index, columns=m.close.columns)
    f = {"eligible": frame(True), "btc_regime": pd.Series(True, index=index),
         "return90": frame(1.0), "high_distance": frame(0.1), "surge": frame(2.0),
         "volatility60": frame(0.5), "above200": frame(True),
         "breadth7": pd.Series(0.6, index=index)}
    return m, f


def test_all_features_are_prefix_invariant():
    full = market()
    short = deepcopy(full)
    for attr in ("open", "high", "low", "close", "volume"):
        setattr(short, attr, getattr(short, attr).iloc[:600])
    a, b = features(full), features(short)
    for key in a:
        left = a[key].iloc[:600]
        if isinstance(left, pd.Series):
            pd.testing.assert_series_equal(left, b[key])
        else:
            pd.testing.assert_frame_equal(left, b[key])


def test_future_perturbations_do_not_change_past_portfolio():
    m = market()
    other = deepcopy(m)
    for attr in ("open", "high", "low", "close", "volume"):
        getattr(other, attr).iloc[600:] *= 50
    a = simulate(m, features(m), "momentum", {"n": 3}, "2020-02-01", "2020-08-01")
    b = simulate(other, features(other), "momentum", {"n": 3}, "2020-02-01", "2020-08-01")
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.cycles == b.cycles


def test_next_open_execution_and_round_trip_cost():
    m, f = tiny([[100, 100, 100, 100, 10], [100, 100, 100, 100, 10],
                 [200, 200, 200, 200, 10], [200, 300, 200, 300, 10]])
    r = simulate(m, f, "btc", {}, "2021-01-01", "2021-01-02")
    assert r.equity.iloc[-1] == pytest.approx(1.5 * 0.997 / 1.003)


def test_stop_gap_fills_at_worse_open():
    m, f = tiny([[100, 100, 100, 100, 10], [100, 100, 100, 100, 10],
                 [100, 120, 90, 110, 10], [70, 80, 65, 75, 10]])
    r = simulate(m, f, "breakout", {"n": 1, "stop": 0.25}, "2021-01-01", "2021-01-02")
    assert r.equity.iloc[-1] == pytest.approx(0.7 * 0.997 / 1.003)
    assert r.cycles[0]["reason"] == "trailing_stop"


def test_entry_day_stop_and_no_same_day_high_lookahead():
    m, f = tiny([[100, 100, 100, 100, 10], [100, 100, 100, 100, 10],
                 [100, 200, 80, 150, 10], [150, 160, 100, 150, 10]])
    r = simulate(m, f, "breakout", {"n": 1, "stop": 0.25}, "2021-01-01", "2021-01-02")
    assert r.cycles[0]["exit"] == "2021-01-02"
    assert r.equity.iloc[-1] == pytest.approx(1.125 * 0.997 / 1.003)
    m.low.iloc[2, 0] = 70
    r = simulate(m, f, "breakout", {"n": 1, "stop": 0.25}, "2021-01-01", "2021-01-02")
    assert r.cycles[0]["exit"] == "2021-01-01"


def test_current_bar_signal_cannot_enter_at_current_open():
    m, f = tiny([[100, 100, 100, 100, 10]] * 5)
    f["surge"].iloc[:3] = 1
    r = simulate(m, f, "breakout", {"n": 1, "stop": 0.25}, "2021-01-01", "2021-01-03")
    assert r.cycles[0]["entry"] == "2021-01-03"


def test_no_empty_position_cycles_after_cash_is_exhausted():
    index = pd.date_range("2020-12-30", periods=6)
    frames = {}
    for symbol in ("BTC-USD", "ALT-USD", "THIRD-USD"):
        prices = np.full(6, 100.0)
        if symbol == "BTC-USD":
            prices[3:] = 1000
        frames[symbol] = pd.DataFrame({"open": prices, "high": prices * 1.1,
                                       "low": prices * 0.9, "close": prices, "volume": 100}, index=index)
    m = from_frames(frames)
    frame = lambda v: pd.DataFrame(v, index=index, columns=m.close.columns)
    eligible = frame(False)
    eligible["BTC-USD"] = True
    eligible.loc[index[2]:, "ALT-USD"] = True
    eligible.loc[index[3]:, "THIRD-USD"] = True
    f = {"eligible": eligible, "btc_regime": pd.Series(True, index=index),
         "return90": frame(1.0), "high_distance": frame(0.1), "surge": frame(2.0),
         "volatility60": frame(0.5), "above200": frame(True)}
    # The BTC gap raises the next slot budget above remaining cash. ALT exhausts it;
    # THIRD's later signal must not produce a fictitious zero-quantity position.
    r = simulate(m, f, "breakout", {"n": 3, "stop": 0.25}, "2021-01-01", "2021-01-04")
    assert len(r.cycles) == 2
    assert {x["symbol"] for x in r.cycles} == {"BTC-USD", "ALT-USD"}
    assert all(np.isfinite(x["return"]) for x in r.cycles)


def test_censored_horizons_and_missing_endpoints_are_not_losers():
    m = market(500)
    r = forward_returns(m.close, 180)
    assert r.iloc[-180:].isna().all().all()
    assert r.iloc[0].notna().all()
    m.close.iloc[180, 0] = np.nan
    assert pd.isna(forward_returns(m.close, 180).iloc[0, 0])


def test_missing_data_is_never_backfilled_into_eligibility():
    m = market()
    m.close.loc[m.close.index[:300], "BTC-USD"] = np.nan
    f = features(m)
    assert not f["eligible"]["BTC-USD"].iloc[:665].any()
    assert pd.isna(f["return180"]["BTC-USD"].iloc[300])
    assert protocol()["cost_per_side"] == 0.003
