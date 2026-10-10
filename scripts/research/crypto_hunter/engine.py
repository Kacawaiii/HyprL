"""Daily open execution, explicit friction, causal stop fills and cash accounting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Market


def rank_candidates(f: dict, i: int, family: str, config: dict) -> list[int]:
    """All inputs refer to a completed signal bar, including cross-sectional ranks."""
    if not bool(f["btc_regime"].iloc[i]):
        return []
    ok = f["eligible"].iloc[i].to_numpy().copy()
    strength = f["return90"].iloc[i].to_numpy()
    ok &= np.isfinite(strength) & (strength > 0)
    if family == "breakout":
        ok &= (f["high_distance"].iloc[i].to_numpy() > 0) & (f["surge"].iloc[i].to_numpy() >= 1.5)
    ids = np.flatnonzero(ok)
    # Stable deterministic symbol order resolves ties without future information.
    return sorted(ids, key=lambda j: (-strength[j], j))[:config["n"]]


def breadth_trigger(f: dict, i: int, threshold: float) -> bool:
    b = f["breadth7"]
    return (i >= 30 and bool(f["btc_regime"].iloc[i]) and
            b.iloc[i] >= threshold and b.iloc[i - 1] < threshold and
            b.iloc[i] - b.iloc[i - 30] >= 0.10)


@dataclass
class Run:
    equity: pd.Series
    held_at_open: np.ndarray
    cycles: list[dict]
    metrics: dict


def simulate(m: Market, f: dict, family: str, config: dict, start: str, end: str,
             cost: float = 0.003, delay: int = 1) -> Run:
    if delay < 1:
        raise ValueError("execution must follow the completed signal day")
    index, symbols = m.close.index, list(m.close.columns)
    dates = np.flatnonzero((index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end)))
    if not len(dates):
        raise ValueError("empty backtest period")
    o, h, lo, c = (getattr(m, x).to_numpy() for x in ("open", "high", "low", "close"))
    marks = m.close.ffill(limit=7).to_numpy()
    vol = f["volatility60"].to_numpy()
    regime, above = f["btc_regime"].to_numpy(), f["above200"].to_numpy()
    cash, nav, traded = 1.0, [], 0.0
    pos, cycles = {}, []
    held = np.zeros(c.shape, dtype=bool)

    def value_at(i: int, use_open: bool = False) -> float:
        total = cash
        for j, x in pos.items():
            price = o[i, j] if use_open and np.isfinite(o[i, j]) else marks[i, j]
            total += x["qty"] * (price if np.isfinite(price) else 0)
        return total

    def sell(j: int, qty: float, price: float, i: int, reason: str) -> None:
        nonlocal cash, traded
        x = pos[j]
        dollars = max(0.0, price) * qty
        before = value_at(i, True)
        traded += dollars / max(before, 1e-12)
        proceeds = dollars * (1 - cost)
        cash += proceeds
        x["received"] += proceeds
        x["qty"] -= qty
        if x["qty"] <= 1e-10:
            cycles.append({"symbol": symbols[j], "entry": str(index[x["entry"]].date()),
                           "exit": str(index[i].date()), "reason": reason,
                           "return": x["received"] / x["spent"] - 1})
            del pos[j]

    def buy(j: int, budget: float, i: int) -> None:
        nonlocal cash, traded
        if not np.isfinite(o[i, j]) or budget <= 1e-10:
            return
        spend = min(cash, budget)
        if spend <= 1e-10:
            return
        dollars = spend / (1 + cost)
        traded += dollars / max(value_at(i, True), 1e-12)
        qty = dollars / o[i, j]
        cash -= spend
        if j not in pos:
            pos[j] = {"qty": 0.0, "spent": 0.0, "received": 0.0,
                      "entry": i, "peak": o[i, j]}
        pos[j]["qty"] += qty
        pos[j]["spent"] += spend

    for i in dates:
        s = i - delay
        if s < 0:
            nav.append(value_at(i))
            continue
        exited = set()
        # Regime/time exits use previous completed day; no current-day prices in selection.
        for j, x in list(pos.items()):
            missing = not np.isfinite(marks[i, j])
            leave = missing
            if family == "breakout":
                leave |= not regime[s] or i - x["entry"] >= 365
            elif family == "breadth":
                leave |= (not regime[s] or f["breadth7"].iloc[s] < config["threshold"] - 0.10
                          or i - x["entry"] >= 90)
            if leave and (missing or np.isfinite(o[i, j])):
                sell(j, x["qty"], 0 if missing else o[i, j], i, "missing_writeoff" if missing else "rule_exit")
                exited.add(j)

        targets = None
        monthly = index[s].is_month_end
        if family == "btc" and i == dates[0]:
            targets = {symbols.index("BTC-USD"): 1.0}
        elif family == "carre" and (monthly or i == dates[0]):
            ids = [symbols.index(x) for x in ("BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD") if x in symbols]
            # Four fixed hindsight-selected names. Missing history remains in cash; no backfill.
            weights = {j: 1 / vol[s, j] for j in ids if above[s, j] and np.isfinite(vol[s, j]) and vol[s, j] > 0}
            denom = sum(1 / vol[s, j] for j in ids if np.isfinite(vol[s, j]) and vol[s, j] > 0)
            targets = {j: w / denom for j, w in weights.items()} if denom else {}
        elif family == "momentum" and (monthly or i == dates[0]):
            ids = rank_candidates(f, s, family, config)
            targets = {j: 1 / config["n"] for j in ids}
        elif family == "breadth" and not pos and breadth_trigger(f, s, config["threshold"]):
            ids = rank_candidates(f, s, family, config)
            targets = {j: 1 / config["n"] for j in ids}

        if targets is not None:
            base = value_at(i, True)
            # Retained positions resize; they are not artificially sold and rebought monthly.
            for j, x in list(pos.items()):
                if not np.isfinite(o[i, j]):
                    continue
                desired = base * targets.get(j, 0)
                excess = x["qty"] * o[i, j] - desired
                if excess > 1e-10:
                    sell(j, min(x["qty"], excess / o[i, j]), o[i, j], i, "rebalance")
            for j, weight in targets.items():
                if np.isfinite(o[i, j]):
                    current = pos.get(j, {}).get("qty", 0) * o[i, j]
                    buy(j, max(0.0, base * weight - current), i)
        elif family == "breakout":
            budget = value_at(i, True) / config["n"]
            for j in rank_candidates(f, s, family, config):
                if len(pos) >= config["n"]:
                    break
                if j not in pos and j not in exited:
                    buy(j, budget, i)

        for j in pos:
            held[i, j] = True
        if family == "breakout":
            for j, x in list(pos.items()):
                stop = x["peak"] * (1 - config["stop"])
                if np.isfinite(lo[i, j]) and lo[i, j] <= stop:
                    sell(j, x["qty"], min(o[i, j], stop), i, "trailing_stop")
                elif np.isfinite(c[i, j]):
                    x["peak"] = max(x["peak"], c[i, j])
        if i == dates[-1]:
            for j, x in list(pos.items()):
                sell(j, x["qty"], marks[i, j] if np.isfinite(marks[i, j]) else 0, i, "terminal_liquidation")
        nav.append(value_at(i))

    equity = pd.Series(nav, index=index[dates])
    years = len(equity) / 365.25
    path = np.r_[1.0, equity.to_numpy()]
    dd = float((path / np.maximum.accumulate(path) - 1).min())
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    r = np.array([x["return"] for x in cycles])
    annual, previous = {}, 1.0
    for year, x in equity.groupby(equity.index.year):
        annual[str(year)] = float(x.iloc[-1] / previous - 1)
        previous = x.iloc[-1]
    metrics = {"cagr": cagr, "max_drawdown": dd, "calmar": cagr / abs(dd) if dd else 0,
               "cycles": len(r), "hit_rate": float((r > 0).mean()) if len(r) else None,
               "average_win": float(r[r > 0].mean()) if (r > 0).any() else None,
               "average_loss": float(r[r <= 0].mean()) if (r <= 0).any() else None,
               "realized_200_cycles": int((r >= 2).sum()), "turnover_annual": traded / years,
               "annual_returns": annual,
               "writeoffs": sum(x["reason"] == "missing_writeoff" for x in cycles)}
    return Run(equity, held, cycles, metrics)
