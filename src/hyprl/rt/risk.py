from __future__ import annotations

from typing import Optional, Tuple


def size_position(
    equity: float,
    risk_pct: float,
    atr: float,
    price: float,
    rr: float = 2.0,
    k_atr: float = 1.0,
    min_qty: int = 1,
    max_qty: Optional[int] = None,
) -> Tuple[int, float | None, float | None]:
    if atr is None or atr <= 0 or price <= 0:
        return 0, None, None
    budget = max(equity * max(risk_pct, 0.0), 0.0)
    raw_qty = int(budget / (atr * max(k_atr, 1e-6)))
    qty = max(min_qty, raw_qty)
    if max_qty is not None:
        qty = min(qty, max_qty)
    if qty <= 0:
        return 0, None, None
    stop_delta = k_atr * atr
    tp_delta = rr * k_atr * atr
    stop_price = price - stop_delta
    take_profit = price + tp_delta
    return qty, stop_price, take_profit
