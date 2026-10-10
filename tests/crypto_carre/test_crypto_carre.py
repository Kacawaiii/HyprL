import numpy as np
import pandas as pd
import pytest

from scripts.research.crypto_carre import engine, events, protocol
from scripts.research.crypto_carre.data import Blocked, Grant, coinbase_bars


def days(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def walk(n, seed, drift=0.001, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.03, n))), days(n, start))


def bars(close):
    c = close.astype(float)
    return pd.DataFrame({"low": c * 0.99, "high": c * 1.01, "open": c, "close": c, "volume": 10.0}, index=c.index)


# ---- cutoff
def test_cutoff_rejects_later_observation():
    idx = pd.DatetimeIndex(["2026-08-31", "2026-09-01"], tz="UTC")
    with pytest.raises(ValueError):
        protocol.assert_cutoff(idx)
    protocol.assert_cutoff(idx[:1])


def test_coinbase_payload_after_cutoff_is_refused_not_truncated():
    ts = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
    with pytest.raises(ValueError):
        coinbase_bars([[ts, 1, 2, 1, 2, 3]], protocol.FIRST_DAY, protocol.CUTOFF)


def test_engine_and_events_refuse_data_after_cutoff():
    late = walk(10, 1, start="2026-08-25")  # runs into September
    with pytest.raises(ValueError):
        engine.simulate(pd.DataFrame({"A": late}), {})
    with pytest.raises(ValueError):
        events.listing_events({"A-USD": bars(late)})


def test_network_needs_authorization_file(tmp_path):
    with pytest.raises(Exception):
        Grant(tmp_path / "grant.json", tmp_path / "cache")  # not in ~/authorizations, or inside no repo


# ---- costs and timing
def test_costs_charged_on_traded_notional_only():
    close = pd.DataFrame({"A": [100.0] * 40}, days(40))  # flat price: only costs move equity
    t = {close.index[0]: pd.Series({"A": 1.0})}
    r, held, turn = engine.simulate(close, t, side_cost=0.003)
    assert r.iloc[1] == pytest.approx(-0.003)
    assert r.iloc[2:].abs().sum() == 0 and turn.sum() == pytest.approx(1.0)
    r2, *_ = engine.simulate(close, t, side_cost=0.006)
    assert r2.iloc[1] == pytest.approx(-0.006)


def test_round_trip_cost_is_per_side_times_two():
    close = pd.DataFrame({"A": [100.0] * 40}, days(40))
    t = {close.index[0]: pd.Series({"A": 1.0}), close.index[10]: pd.Series(dtype=float)}
    r, *_ = engine.simulate(close, t, side_cost=0.003)
    assert (1 + r).prod() == pytest.approx((1 - 0.003) ** 2)


def test_signal_on_close_t_trades_next_close_no_lookahead():
    n = 30
    px = [100.0] * n
    px[10] = 200.0  # jump on the signal day itself
    px[11] = 300.0  # jump on the trade day
    px[12] = 330.0
    close = pd.DataFrame({"A": px}, days(n))
    r, held, _ = engine.simulate(close, {close.index[10]: pd.Series({"A": 1.0})}, side_cost=0.0)
    assert r.iloc[10] == 0 and r.iloc[11] == 0  # neither the signal-day nor trade-day move is captured
    assert r.iloc[12] == pytest.approx(0.10)


def test_targets_use_only_past_data():
    a, b = walk(400, 1), walk(400, 2)
    full = pd.DataFrame({"A": a, "B": b})
    cut = full.iloc[:300]
    t_full = engine.make_targets(full, ["A", "B"], "monthly", "inv_vol", ma=200)
    t_cut = engine.make_targets(cut, ["A", "B"], "monthly", "inv_vol", ma=200)
    for day, w in t_cut.items():
        pd.testing.assert_series_equal(w.sort_index(), t_full[day].sort_index())


def test_trend_filter_goes_to_cash_not_renormalised():
    up = pd.Series(np.linspace(100, 300, 300), days(300))
    down = pd.Series(np.linspace(300, 100, 300), days(300))
    t = engine.make_targets(pd.DataFrame({"U": up, "D": down}), ["U", "D"], "monthly", "equal", ma=200)
    last = list(t.values())[-1]
    assert list(last.index) == ["U"] and last["U"] == pytest.approx(0.5)  # other half stays in cash


def test_inverse_vol_gives_calmer_asset_more_weight():
    calm = pd.Series(100 + np.sin(np.arange(100)), days(100))
    wild = pd.Series(100 + 30 * np.sin(np.arange(100)), days(100))
    w = engine._inv_vol(pd.DataFrame({"C": calm, "W": wild}), days(100)[-1], ["C", "W"])
    assert w["C"] > w["W"] and w.sum() == pytest.approx(1)


# ---- universe rule
def test_universe_top4_by_past_dollar_volume_excludes_stables_and_young_assets():
    n = 300
    idx = days(n)
    names = ["AAA-USD", "BBB-USD", "CCC-USD", "DDD-USD", "EEE-USD", "USDC-USD", "NEW-USD"]
    close = pd.DataFrame(100.0, idx, names)
    vol = pd.DataFrame({k: 10.0 * (i + 1) for i, k in enumerate(names)}, idx)
    vol["USDC-USD"] = 1e9  # stablecoin: huge volume, must be excluded
    close.loc[idx[:150], "NEW-USD"] = np.nan  # only 150 days of history
    vol["NEW-USD"] = 1e8
    top = engine.universe_top(close, vol, idx[-1], n=4)
    assert "USDC-USD" not in top and "NEW-USD" not in top
    assert top == ["EEE-USD", "DDD-USD", "CCC-USD", "BBB-USD"]


def test_universe_ignores_future_volume():
    idx = days(300)
    close = pd.DataFrame(100.0, idx, ["A-USD", "B-USD"])
    vol = pd.DataFrame({"A-USD": 1.0, "B-USD": 2.0}, idx)
    vol.loc[idx[250:], "A-USD"] = 1e9  # future spike
    assert engine.universe_top(close, vol, idx[240], n=1) == ["B-USD"]


# ---- Part B
def test_event_cost_and_delisting_handling():
    life = bars(pd.Series(np.linspace(100, 110, 60), days(60, "2024-01-01")))  # dies after 60 days
    ev_last = events.listing_events({"X-USD": life}, entry_days=(1,), hold_days=(30, 90), dead_exit="last")
    ev_zero = events.listing_events({"X-USD": life}, entry_days=(1,), hold_days=(30, 90), dead_exit="zero")
    held = ev_last[ev_last.hold == 30].iloc[0]
    assert held.status == "held" and held.ret == pytest.approx(events.net(held.gross + 1))
    assert ev_last[ev_last.hold == 90].iloc[0].status == "delisted"
    assert ev_zero[ev_zero.hold == 90].iloc[0].ret == pytest.approx(events.net(0.0))
    assert events.net(1.0) == pytest.approx(-0.006, abs=1e-4)


def test_alive_product_right_censored_not_marked_to_cutoff():
    alive = bars(walk(40, 3, start="2026-07-23"))  # ends 2026-08-31
    assert events.listing_events({"Y-USD": alive}, entry_days=(1,), hold_days=(30, 365)).query("hold == 365").empty


def test_top_share_and_split_use_listing_date():
    r = pd.Series([1.0] * 3 + [-0.5] * 97)
    assert events.top_share(r, 0.01) == pytest.approx(1 / 3)
    ev = pd.DataFrame({"listed": pd.to_datetime(["2022-12-31", "2023-01-01"], utc=True)})
    tr, te = events.split(ev)
    assert len(tr) == 1 and len(te) == 1


def test_metrics_basic():
    r = pd.Series([0.01, -0.02, 0.015] * 200, days(600))
    m = engine.metrics(r)
    assert m["maxDD"] < 0 and np.isfinite(m["Sharpe"]) and m["worst_year"] < m["CAGR"] + 1


def test_runner_smoke_on_synthetic_bars(tmp_path):
    from scripts.research.crypto_carre import run
    names = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD", "XRP-USD", "USDC-USD", "ZZZ-USD"]
    data = {n: bars(walk(1400, i, start="2021-01-01" if i != 7 else "2022-03-01")) for i, n in enumerate(names)}
    run.part_a(data, tmp_path)
    run.part_b(data, tmp_path)
    assert (tmp_path / "part_a_metrics.csv").exists() and (tmp_path / "part_b_filters.csv").exists()
