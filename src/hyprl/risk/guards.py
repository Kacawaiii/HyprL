from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(slots=True)
class RiskGuardConfig:
    max_drawdown_pct: float = 0.2
    min_pf: float = 1.0
    lookback_trades: int = 50
    max_consec_losses: int | None = None


@dataclass(slots=True)
class RiskGuardState:
    equity_start: float
    equity_current: float
    trade_pnls: List[float] = field(default_factory=list)
    consec_losses: int = 0
    is_triggered: bool = False
    reason: str | None = None


class RiskGuardMonitor:
    """Guard rails to halt trading when live metrics degrade."""

    def __init__(self, config: RiskGuardConfig, equity_start: float) -> None:
        self.config = config
        self.state = RiskGuardState(equity_start=float(equity_start), equity_current=float(equity_start))

    def _update_equity(self, pnl: float) -> None:
        self.state.equity_current += pnl
        self.state.trade_pnls.append(float(pnl))
        # Track consecutive losses
        if pnl < 0:
            self.state.consec_losses += 1
        else:
            self.state.consec_losses = 0
        # Trim history to lookback
        lookback = max(1, int(self.config.lookback_trades))
        if len(self.state.trade_pnls) > lookback:
            self.state.trade_pnls = self.state.trade_pnls[-lookback:]

    def _max_drawdown_pct(self) -> float:
        if not self.state.trade_pnls:
            return 0.0
        equity = self.state.equity_start
        peak = equity
        dd_max = 0.0
        for pnl in self.state.trade_pnls:
            equity += pnl
            peak = max(peak, equity)
            dd_max = max(dd_max, (peak - equity) / peak if peak > 0 else 0.0)
        return dd_max

    def _live_pf(self) -> float | None:
        if len(self.state.trade_pnls) < 2:
            return None
        gains = sum(p for p in self.state.trade_pnls if p > 0)
        losses = -sum(p for p in self.state.trade_pnls if p < 0)
        if losses <= 0:
            return float("inf") if gains > 0 else None
        return gains / losses

    def _check_drawdown(self) -> bool:
        dd = self._max_drawdown_pct()
        if dd > self.config.max_drawdown_pct:
            self.state.is_triggered = True
            self.state.reason = "max_drawdown_exceeded"
            return True
        return False

    def _check_pf(self) -> bool:
        live_pf = self._live_pf()
        if live_pf is None:
            return False
        if live_pf < self.config.min_pf:
            self.state.is_triggered = True
            self.state.reason = "min_pf_breached"
            return True
        return False

    def _check_consec_losses(self) -> bool:
        if self.config.max_consec_losses is None:
            return False
        if self.state.consec_losses >= self.config.max_consec_losses:
            self.state.is_triggered = True
            self.state.reason = "max_consec_losses"
            return True
        return False

    def on_new_trade(self, pnl: float) -> None:
        if self.state.is_triggered:
            return
        self._update_equity(pnl)
        self._check_drawdown()
        if self.state.is_triggered:
            return
        self._check_pf()
        if self.state.is_triggered:
            return
        self._check_consec_losses()

    def should_stop_trading(self) -> bool:
        return self.state.is_triggered

    def get_drawdown_pct(self) -> float:
        return self._max_drawdown_pct()

    def get_live_pf(self) -> float | None:
        return self._live_pf()
