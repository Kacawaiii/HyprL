import pytest

pl = pytest.importorskip("polars")
hy_sc = pytest.importorskip("hyprl_supercalc")


def make_toy_df():
    return pl.DataFrame(
        {
            "ts": [1, 2, 3, 4],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        }
    )


def test_run_batch_backtest_smoke():
    df = make_toy_df()
    # Exposition très simple : flat → long → long → flat
    signal = [0.0, 1.0, 1.0, 0.0]

    cfg = {
        "risk_pct": 0.1,
        "commission_pct": 0.0005,
        "slippage_pct": 0.0005,
        "max_leverage": 1.0,
        "params": [],
        "allow_short": False,
        "label": "test",
    }

    run_single = getattr(hy_sc, "run_backtest_py", None)
    if run_single is None:
        pytest.skip("run_backtest_py not available in installed hyprl_supercalc")
    rep = run_single(df, signal, cfg)
    assert isinstance(rep, dict)
    assert "metrics" in rep
    metrics = rep["metrics"]

    # Sanity checks
    assert "total_return" in metrics
    assert isinstance(metrics["total_return"], float)
    # equity curve doit être non-vide
    assert rep["equity_curve"]
    # Trades detail should be present
    assert "trades" in rep
    assert len(rep["trades"]) >= 1
