"""Gate evaluation (Gate1/Gate2) for strategy trade metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from hyprl.analysis.trade_aggregator import RollingMetrics, StrategyMetrics

# Overrides for known long-only v0 strategies (strategy_id -> thresholds)
LONG_ONLY_GATE1: Mapping[str, dict[str, float | int]] = {
    # NVDA 1h v0 long-only (legacy id)
    "06cf8b1aeb7d1b77": {"trades": 150, "pf": 1.7, "sharpe": 0.45, "maxdd": 0.10},
    # NVDA 1h v0 long-only (current id)
    "08bb23db6e7ec52c": {"trades": 150, "pf": 1.7, "sharpe": 0.45, "maxdd": 0.10},
}

@dataclass
class GateDecision:
    passed: bool
    reasons: Dict[str, bool]


def _rolling_ok(rolling: RollingMetrics, *, pf_floor: float, dd_ceiling: float) -> tuple[bool, bool]:
    pf_ok = True
    dd_ok = True
    if rolling.pf_min_100 is not None:
        pf_ok = rolling.pf_min_100 >= pf_floor
    if rolling.maxdd_max_100 is not None:
        dd_ok = rolling.maxdd_max_100 <= dd_ceiling
    return pf_ok, dd_ok


def _gate1_thresholds(strategy_id: str | None) -> dict[str, float | int]:
    if strategy_id and strategy_id in LONG_ONLY_GATE1:
        return LONG_ONLY_GATE1[strategy_id]  # type: ignore[return-value]
    return {"trades": 300, "pf": 1.3, "sharpe": 0.5, "maxdd": 0.20}


def check_gate1(metrics: StrategyMetrics, rolling: RollingMetrics, *, strategy_id: str | None = None) -> GateDecision:
    """Return Gate1 decision and criteria flags."""
    th = _gate1_thresholds(strategy_id)
    flags: Dict[str, bool] = {
        "trades_ok": metrics.n_trades >= th["trades"],
        "pf_ok": metrics.pf >= th["pf"],
        "sharpe_ok": metrics.sharpe >= th["sharpe"],
        "maxdd_ok": metrics.maxdd <= th["maxdd"],
    }
    pf_roll_ok, dd_roll_ok = _rolling_ok(rolling, pf_floor=1.0, dd_ceiling=0.30)
    flags["rolling_pf_ok"] = pf_roll_ok
    flags["rolling_dd_ok"] = dd_roll_ok
    passed = all(flags.values())
    return GateDecision(passed=passed, reasons=flags)


def check_gate2(metrics: StrategyMetrics, rolling: RollingMetrics) -> GateDecision:
    """Return Gate2 decision and criteria flags."""
    flags: Dict[str, bool] = {
        "trades_ok": metrics.n_trades >= 1000,
        "pf_ok": metrics.pf >= 1.5,
        "sharpe_ok": metrics.sharpe >= 0.8,
        "maxdd_ok": metrics.maxdd <= 0.20,
    }
    pf_roll_ok, dd_roll_ok = _rolling_ok(rolling, pf_floor=1.0, dd_ceiling=0.30)
    flags["rolling_pf_ok"] = pf_roll_ok
    flags["rolling_dd_ok"] = dd_roll_ok
    passed = all(flags.values())
    return GateDecision(passed=passed, reasons=flags)


__all__ = ["GateDecision", "check_gate1", "check_gate2"]
