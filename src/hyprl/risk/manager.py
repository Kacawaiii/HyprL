from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

Direction = Literal["long", "short"]


@dataclass(slots=True)
class RiskConfig:
    balance: float = 100_000.0
    risk_pct: float = 0.01
    atr_multiplier: float = 2.0
    reward_multiple: float = 2.0
    min_position_size: int = 1


@dataclass(slots=True)
class RiskOutcome:
    direction: Direction
    entry_price: float
    position_size: int
    stop_price: float
    take_profit_price: float
    risk_amount: float
    rr_multiple: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "direction": self.direction,
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "risk_amount": self.risk_amount,
            "rr_multiple": self.rr_multiple,
        }


def _normalize_direction(direction: str) -> Direction:
    normalized = direction.lower()
    if normalized not in ("long", "short"):
        raise ValueError(f"Unsupported trade direction: {direction}")
    return normalized  # type: ignore[return-value]


def compute_stop_price(entry_price: float, atr: float, direction: Direction, atr_multiplier: float) -> float:
    if atr <= 0 or atr_multiplier <= 0 or entry_price <= 0:
        return entry_price
    distance = atr * atr_multiplier
    if direction == "long":
        return max(entry_price - distance, 0.0)
    return entry_price + distance


def compute_take_profit_price(
    entry_price: float,
    atr: float,
    direction: Direction,
    atr_multiplier: float,
    reward_multiple: float,
) -> float:
    if atr <= 0 or atr_multiplier <= 0 or reward_multiple <= 0 or entry_price <= 0:
        return entry_price
    distance = atr * atr_multiplier * reward_multiple
    if direction == "long":
        return entry_price + distance
    return max(entry_price - distance, 0.0)


def compute_position_size(
    balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    min_position_size: int = 1,
) -> int:
    if balance <= 0 or risk_pct <= 0 or entry_price <= 0:
        return 0
    per_unit_risk = abs(entry_price - stop_price)
    if per_unit_risk <= 0:
        return 0
    risk_budget = balance * risk_pct
    size = math.floor(risk_budget / per_unit_risk)
    if size < min_position_size:
        return 0
    return size


def plan_trade(
    entry_price: float,
    atr: float,
    direction: str,
    config: RiskConfig | None = None,
) -> RiskOutcome:
    cfg = config or RiskConfig()
    direction_normalized = _normalize_direction(direction)
    atr = float(atr)
    entry_price = float(entry_price)
    if atr <= 0 or entry_price <= 0 or not math.isfinite(atr) or not math.isfinite(entry_price):
        return RiskOutcome(
            direction=direction_normalized,
            entry_price=entry_price,
            position_size=0,
            stop_price=entry_price,
            take_profit_price=entry_price,
            risk_amount=0.0,
            rr_multiple=0.0,
        )

    stop_price = compute_stop_price(entry_price, atr, direction_normalized, cfg.atr_multiplier)
    take_profit_price = compute_take_profit_price(
        entry_price,
        atr,
        direction_normalized,
        cfg.atr_multiplier,
        cfg.reward_multiple,
    )
    position_size = compute_position_size(
        cfg.balance,
        cfg.risk_pct,
        entry_price,
        stop_price,
        cfg.min_position_size,
    )
    per_unit_risk = abs(entry_price - stop_price)
    risk_amount = position_size * per_unit_risk
    rr_multiple = cfg.reward_multiple if per_unit_risk > 0 else 0.0

    if position_size == 0:
        stop_price = entry_price
        take_profit_price = entry_price
        risk_amount = 0.0
        rr_multiple = 0.0

    return RiskOutcome(
        direction=direction_normalized,
        entry_price=entry_price,
        position_size=position_size,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        risk_amount=risk_amount,
        rr_multiple=rr_multiple,
    )
