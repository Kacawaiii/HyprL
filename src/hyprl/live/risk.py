from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

from hyprl.live.broker import PaperBrokerImpl
from hyprl.live.types import TradeSignal
from hyprl.risk.sizing import clamp_position_size


@dataclass(slots=True)
class LiveRiskConfig:
    max_daily_loss_pct: float = 0.03
    max_position_notional_pct: float = 0.2
    max_notional_per_trade: float | None = None
    max_gross_exposure_pct: float = 1.0


@dataclass(slots=True)
class LiveRiskDecision:
    allowed: bool
    reason: str | None = None
    bypassed: bool = False


class LiveRiskManager:
    """Minimal live risk guard rails for paper/real execution."""

    def __init__(
        self,
        cfg: LiveRiskConfig,
        broker: PaperBrokerImpl,
        *,
        clock: Callable[[], date] | None = None,
        parity_mode: bool = False,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self._clock = clock or date.today
        self._current_day: date | None = None
        self._day_start_equity: float | None = None
        self._kill_switch_engaged = False
        self._kill_reason: str | None = None
        self._parity_mode = bool(parity_mode)

    def _maybe_reset_day(self) -> None:
        today = self._clock()
        if self._current_day != today:
            self._current_day = today
            self._day_start_equity = self.broker.get_balance()
            self._kill_switch_engaged = False
            self._kill_reason = None

    def allow_trade(self, signal: TradeSignal, price: float) -> LiveRiskDecision:
        self._maybe_reset_day()
        if self._kill_switch_engaged:
            return LiveRiskDecision(False, self._kill_reason or "kill_switch_engaged")

        equity_now = self.broker.get_balance()
        if self._day_start_equity is None:
            self._day_start_equity = equity_now

        start_equity = self._day_start_equity or equity_now
        if start_equity <= 0:
            return LiveRiskDecision(False, "insufficient_start_equity")

        drawdown = (equity_now - start_equity) / start_equity
        if self.cfg.max_daily_loss_pct > 0 and drawdown <= -self.cfg.max_daily_loss_pct:
            self._kill_switch_engaged = True
            self._kill_reason = "daily_loss_kill_switch"
            return LiveRiskDecision(False, "daily_loss_kill_switch")

        if equity_now <= 0:
            return LiveRiskDecision(False, "insufficient_equity")

        notional = abs(signal.size * float(price))
        cap_reason = None
        if not self._parity_mode:
            if self.cfg.max_notional_per_trade is not None and self.cfg.max_notional_per_trade > 0:
                if notional > self.cfg.max_notional_per_trade:
                    cap_reason = "position_notional_exceeded"

            if self.cfg.max_position_notional_pct > 0:
                limit = self.cfg.max_position_notional_pct * equity_now
                if notional > limit:
                    cap_reason = "position_notional_exceeded"

            if cap_reason:
                return LiveRiskDecision(False, cap_reason)

        if not self._parity_mode:
            gross = 0.0
            for pos in self.broker.get_positions():
                ref_price = float(price) if pos.symbol == signal.symbol else float(pos.avg_price)
                gross += abs(pos.size * ref_price)
            if self.cfg.max_gross_exposure_pct > 0:
                exposure_limit = self.cfg.max_gross_exposure_pct * equity_now
                if gross + notional > exposure_limit:
                    return LiveRiskDecision(False, "gross_exposure_exceeded")

        return LiveRiskDecision(True, None)
