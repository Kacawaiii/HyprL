from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from hyprl.live.broker import PaperBrokerImpl


@dataclass(slots=True)
class PortfolioRiskLimits:
    """Global portfolio risk caps expressed as fractions of equity."""

    max_total_risk_pct: float = 0.05
    max_ticker_risk_pct: float | None = 0.03
    max_group_risk_pct: float | None = 0.04
    max_positions: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.max_total_risk_pct <= 1.0):
            raise ValueError("max_total_risk_pct must be in (0, 1].")
        for name, value in (
            ("max_ticker_risk_pct", self.max_ticker_risk_pct),
            ("max_group_risk_pct", self.max_group_risk_pct),
        ):
            if value is None:
                continue
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1] when set.")
        if self.max_positions is not None and self.max_positions <= 0:
            raise ValueError("max_positions must be positive when set.")


@dataclass(slots=True)
class PortfolioRiskDecision:
    allowed: bool
    position_size: float
    risk_amount: float
    adjusted: bool = False
    reason: Optional[str] = None


class PortfolioRiskManager:
    """Enforces total/ticker/group risk caps across open positions."""

    def __init__(
        self,
        limits: PortfolioRiskLimits,
        group_map: Mapping[str, str] | None = None,
    ) -> None:
        self.limits = limits
        self.group_map: Dict[str, str] = {}
        if group_map:
            for symbol, group in group_map.items():
                self.register_symbol(symbol, group)

    def register_symbol(self, symbol: str, group: str | None = None) -> None:
        self.group_map[symbol.upper()] = (group or symbol).lower()

    def _group_for(self, symbol: str) -> str:
        symbol_norm = symbol.upper()
        return self.group_map.get(symbol_norm, symbol_norm)

    def _aggregate_group_risk(self, open_risk: Mapping[str, float]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for symbol, risk in open_risk.items():
            group = self._group_for(symbol)
            totals[group] = totals.get(group, 0.0) + float(risk)
        return totals

    def evaluate(
        self,
        symbol: str,
        *,
        proposed_position_size: float,
        risk_amount: float,
        equity: float,
        broker: PaperBrokerImpl,
        min_position_size: float = 0.0,
    ) -> PortfolioRiskDecision:
        symbol_norm = symbol.upper()
        if equity <= 0 or proposed_position_size <= 0 or risk_amount <= 0:
            return PortfolioRiskDecision(
                allowed=False,
                position_size=0.0,
                risk_amount=0.0,
                reason="invalid_inputs",
            )

        open_risk = broker.get_open_risk_amounts()
        group_risk = self._aggregate_group_risk(open_risk)
        total_open = sum(open_risk.values())
        total_cap = self.limits.max_total_risk_pct * equity
        available_total = total_cap - total_open
        if available_total <= 0:
            return PortfolioRiskDecision(
                allowed=False,
                position_size=0.0,
                risk_amount=0.0,
                reason="total_risk_cap",
            )

        if self.limits.max_positions is not None:
            if len(broker.get_positions()) >= self.limits.max_positions:
                return PortfolioRiskDecision(
                    allowed=False,
                    position_size=0.0,
                    risk_amount=0.0,
                    reason="max_positions",
                )

        ticker_cap = (
            self.limits.max_ticker_risk_pct * equity
            if self.limits.max_ticker_risk_pct is not None
            else float("inf")
        )
        group_cap = (
            self.limits.max_group_risk_pct * equity
            if self.limits.max_group_risk_pct is not None
            else float("inf")
        )
        ticker_open = open_risk.get(symbol_norm, 0.0)
        group_name = self._group_for(symbol_norm)
        group_open = group_risk.get(group_name, 0.0)

        available_ticker = ticker_cap - ticker_open
        available_group = group_cap - group_open

        capacity = min(available_total, available_ticker, available_group)
        if capacity <= 0:
            return PortfolioRiskDecision(
                allowed=False,
                position_size=0.0,
                risk_amount=0.0,
                reason="risk_cap_exceeded",
            )

        if risk_amount <= capacity:
            return PortfolioRiskDecision(
                allowed=True,
                position_size=proposed_position_size,
                risk_amount=risk_amount,
                adjusted=False,
                reason=None,
            )

        scale = capacity / risk_amount if risk_amount > 0 else 0.0
        scaled_size = proposed_position_size * scale
        scaled_risk = risk_amount * scale
        if scaled_size < max(min_position_size, 1e-9):
            return PortfolioRiskDecision(
                allowed=False,
                position_size=0.0,
                risk_amount=0.0,
                reason="scaled_below_min_size",
            )

        return PortfolioRiskDecision(
            allowed=True,
            position_size=scaled_size,
            risk_amount=scaled_risk,
            adjusted=True,
            reason="scaled_to_fit",
        )
