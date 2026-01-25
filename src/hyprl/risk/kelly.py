from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(slots=True)
class KellyParams:
    lookback_trades: int
    base_risk_pct: float
    min_trades: int
    max_multiplier: float
    min_multiplier: float


@dataclass(slots=True)
class KellyStats:
    win_rate: float
    avg_win: float
    avg_loss: float
    kelly_full: float
    kelly_half: float


class KellySizer:
    """Rolling Kelly-based risk fraction with conservative caps."""

    def __init__(self, params: KellyParams) -> None:
        self.params = params
        self.lookback = max(1, int(params.lookback_trades))
        self.base_risk_pct = float(params.base_risk_pct)
        self._pnl_history: List[float] = []

    def update_from_pnls(self, pnls: Sequence[float]) -> None:
        self._pnl_history = [float(p) for p in pnls[-self.lookback :]]

    def update_with_trade(self, pnl: float) -> None:
        self._pnl_history.append(float(pnl))
        if len(self._pnl_history) > self.lookback:
            self._pnl_history.pop(0)

    def _compute_stats(self) -> KellyStats | None:
        if len(self._pnl_history) < max(1, self.params.min_trades):
            return None

        wins = [p for p in self._pnl_history if p > 0]
        losses = [p for p in self._pnl_history if p < 0]
        if not wins or not losses:
            return None

        win_rate = len(wins) / len(self._pnl_history)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p

        kelly_full = (p * b - q) / b if b > 0 else 0.0
        kelly_full = max(0.0, min(kelly_full, 1.0))

        return KellyStats(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            kelly_full=kelly_full,
            kelly_half=kelly_full / 2.0,
        )

    def get_risk_pct(self) -> float:
        stats = self._compute_stats()
        if stats is None:
            return self.base_risk_pct

        # Conservative: use half Kelly, clamp to configured multipliers.
        raw_multiplier = stats.kelly_half / max(self.base_risk_pct, 1e-9)
        multiplier = max(self.params.min_multiplier, min(self.params.max_multiplier, raw_multiplier))
        return self.base_risk_pct * multiplier


def compute_kelly_risk_pct(
    pnls: Sequence[float],
    params: KellyParams,
) -> float:
    sizer = KellySizer(params)
    sizer.update_from_pnls(pnls)
    return sizer.get_risk_pct()
