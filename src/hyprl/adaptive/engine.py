from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, TYPE_CHECKING

import math
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from hyprl.backtest.runner import TradeRecord


@dataclass(slots=True)
class AdaptiveRegime:
    name: str
    min_equity_drawdown: float = 0.0
    max_equity_drawdown: float = 1.0
    min_profit_factor: float = 0.0
    min_sharpe: float = -math.inf
    min_expectancy: float = -math.inf
    risk_overrides: dict[str, float] = field(default_factory=dict)
    threshold_overrides: dict[str, float] = field(default_factory=dict)
    model_overrides: dict[str, str] = field(default_factory=dict)

    def matches(self, metrics: dict[str, float]) -> bool:
        drawdown = metrics.get("drawdown_pct")
        profit_factor = metrics.get("profit_factor")
        sharpe = metrics.get("sharpe")
        expectancy = metrics.get("expectancy")
        if drawdown is None or profit_factor is None or sharpe is None or expectancy is None:
            return False
        return (
            self.min_equity_drawdown <= drawdown <= self.max_equity_drawdown
            and profit_factor >= self.min_profit_factor
            and sharpe >= self.min_sharpe
            and expectancy >= self.min_expectancy
        )


@dataclass(slots=True)
class AdaptiveConfig:
    enable: bool = False
    lookback_trades: int = 20
    default_regime: str = "normal"
    regimes: dict[str, AdaptiveRegime] = field(default_factory=dict)

    def window(self) -> int:
        return max(1, int(self.lookback_trades))

    def get_regime(self, name: Optional[str]) -> Optional[AdaptiveRegime]:
        if not self.regimes:
            return None
        if name and name in self.regimes:
            return self.regimes[name]
        if self.default_regime in self.regimes:
            return self.regimes[self.default_regime]
        first = next(iter(self.regimes))
        return self.regimes[first]


@dataclass(slots=True)
class AdaptiveState:
    regime_name: str
    regime_counts: dict[str, int] = field(default_factory=dict)
    transitions: list[dict[str, int | str]] = field(default_factory=list)
    regime_changes: int = 0
    trades_seen: int = 0

    def record_trade(self) -> None:
        self.regime_counts[self.regime_name] = self.regime_counts.get(self.regime_name, 0) + 1
        self.trades_seen += 1


def _sharpe_ratio(returns: Sequence[float]) -> Optional[float]:
    valid = [float(r) for r in returns if math.isfinite(r)]
    if len(valid) < 2:
        return None
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1))
    if std == 0:
        return None
    return mean / std


def _drawdown_from_equity(equities: Sequence[float]) -> float:
    if not equities:
        return 0.0
    peak = float(equities[0])
    max_dd = 0.0
    for value in equities:
        v = float(value)
        peak = max(peak, v)
        if peak <= 0:
            continue
        drawdown = (peak - v) / peak
        max_dd = max(max_dd, drawdown)
    return max_dd


def evaluate_window(trades: Sequence["TradeRecord"]) -> dict[str, float]:
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "drawdown_pct": 0.0,
            "sharpe": 0.0,
        }
    pnls = np.array([float(trade.pnl) for trade in trades], dtype=float)
    returns = np.array([float(trade.return_pct) for trade in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    positive_sum = float(wins.sum()) if len(wins) else 0.0
    negative_sum = float(losses.sum()) if len(losses) else 0.0
    if negative_sum < 0:
        profit_factor = positive_sum / abs(negative_sum)
    elif positive_sum > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    expectancy = float(pnls.mean()) if len(pnls) else 0.0
    sharpe = _sharpe_ratio(returns) or 0.0
    equities = [float(trade.equity_after) for trade in trades]
    drawdown = _drawdown_from_equity(equities)
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "drawdown_pct": drawdown,
        "sharpe": sharpe,
    }


def _select_regime(config: AdaptiveConfig, metrics: dict[str, float], current: str) -> str:
    if not config.regimes:
        return current
    ordered = list(config.regimes.values())
    for regime in ordered:
        if regime.matches(metrics):
            return regime.name
    drawdown = metrics.get("drawdown_pct", 0.0)
    for regime in ordered:
        if regime.min_equity_drawdown <= drawdown <= regime.max_equity_drawdown:
            return regime.name
    if config.default_regime in config.regimes:
        return config.default_regime
    return current if current in config.regimes else ordered[0].name


def update_state(
    config: AdaptiveConfig,
    state: AdaptiveState,
    metrics: dict[str, float],
    trade_index: int,
) -> AdaptiveState:
    if not config.enable or not config.regimes:
        return state
    target = _select_regime(config, metrics, state.regime_name)
    if target != state.regime_name:
        state.regime_changes += 1
        state.transitions = [*state.transitions, {"trade": trade_index, "regime": target}]
        state.regime_name = target
    state.trades_seen = trade_index
    return state
