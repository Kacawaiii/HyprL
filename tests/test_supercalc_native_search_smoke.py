import pytest

import pytest

pl = pytest.importorskip("polars")
hy_sc = pytest.importorskip("hyprl_supercalc")

DAY_MS = 86_400_000


def _make_df() -> "pl.DataFrame":
    return pl.DataFrame(
        {
            "ts": [i * DAY_MS for i in range(9)],
            "open": [100.0 + i for i in range(9)],
            "high": [100.5 + i for i in range(9)],
            "low": [99.5 + i for i in range(9)],
            "close": [100.0 + i for i in range(9)],
            "volume": [1_000.0 + i * 50.0 for i in range(9)],
        }
    )


def _make_cfg(label: str, allow_short: bool, use_atr: bool) -> dict:
    return {
        "risk_pct": 0.02,
        "commission_pct": 0.0005,
        "slippage_pct": 0.0005,
        "max_leverage": 1.0,
        "params": [],
        "allow_short": allow_short,
        "label": label,
        "atr_window": 3,
        "atr_mult_stop": 2.0,
        "atr_mult_tp": 4.0,
        "use_atr_position_sizing": use_atr,
    }


def _score_tuple(report: dict) -> tuple[float, float, float, float]:
    metrics = report["metrics"]
    pf = float(metrics["profit_factor"])
    sharpe = float(metrics["sharpe"])
    ror = float(metrics["risk_of_ruin"])
    dd = max(abs(float(metrics["max_drawdown"])), float(metrics["maxdd_p95"]))
    expectancy = float(metrics["expectancy"])
    robustness = float(metrics.get("robustness_score", 0.0))
    return (
        -pf + ror * 2.0,
        -sharpe + ror,
        dd + ror * 100.0 - expectancy * 100.0,
        -robustness,
    )


def test_native_search_smoke() -> None:
    df = _make_df()
    signal = [0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0]
    configs = [
        _make_cfg("long_only", allow_short=False, use_atr=False),
        _make_cfg("long_short", allow_short=True, use_atr=False),
        _make_cfg("atr_short", allow_short=True, use_atr=True),
    ]
    lax_constraints = {
        "min_trades": 1,
        "min_profit_factor": 0.5,
        "min_sharpe": -5.0,
        "max_drawdown": 1.0,
        "max_risk_of_ruin": 1.0,
        "min_expectancy": -1.0,
        "min_robustness": 0.0,
        "max_maxdd_p95": 1.0,
        "min_pnl_p05": -1.0,
        "min_pnl_p50": -1.0,
        "min_pnl_p95": -1.0,
    }

    results = hy_sc.run_native_search_py(df, signal, configs, lax_constraints, 5)
    assert results, "Expected at least one candidate under lax constraints"
    score_list = [_score_tuple(rep) for rep in results]
    assert score_list == sorted(score_list), "Results must be sorted by score"

    strict_constraints = dict(lax_constraints)
    strict_constraints["min_trades"] = 3
    strict_constraints["min_profit_factor"] = 0.6
    filtered = hy_sc.run_native_search_py(df, signal, configs, strict_constraints, 5)
    assert filtered, "Strict constraints should still keep at least one config"
    assert all(rep["n_trades"] >= 3 for rep in filtered)
    assert all(rep["metrics"]["profit_factor"] >= 0.6 for rep in filtered)
    assert all(rep["config"]["allow_short"] for rep in filtered), "Long-only config should fail"