from __future__ import annotations

from typing import Tuple


def clamp_position_size(
    entry_price: float,
    stop_price: float,
    position_size: float,
    *,
    equity: float | None = None,
    max_notional: float | None = None,
    max_notional_pct: float | None = None,
) -> Tuple[float, float]:
    """
    Apply absolute and percentage notional caps to a proposed position size.

    Returns (clamped_size, clamped_risk_amount).
    """
    size = float(position_size)
    price = float(entry_price)
    if size <= 0 or price <= 0:
        return 0.0, 0.0

    per_unit_risk = abs(float(entry_price) - float(stop_price))
    if max_notional is not None and max_notional > 0:
        size = min(size, max_notional / price)

    if max_notional_pct is not None and max_notional_pct > 0 and equity is not None and equity > 0:
        cap_value = equity * max_notional_pct
        size = min(size, cap_value / price)

    if size <= 0:
        return 0.0, 0.0

    risk_amount = size * per_unit_risk
    return size, risk_amount
