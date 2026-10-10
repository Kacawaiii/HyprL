"""Part B: early-listing event study (equal tiny weights, one position per product/entry/hold)."""

import numpy as np
import pandas as pd

from .protocol import LAST_DAY, OOS_START, assert_cutoff

ROUND_TRIP = 0.006
ENTRY_DAYS = (1, 7, 30)
HOLD_DAYS = (30, 90, 365)
ALIVE_GRACE_DAYS = 3  # last candle within this many days of the cutoff = still listed


def net(gross_multiple, round_trip=ROUND_TRIP):
    half = round_trip / 2
    return gross_multiple * (1 - half) / (1 + half) - 1


def listing_events(bars_by_product, entry_days=ENTRY_DAYS, hold_days=HOLD_DAYS, dead_exit="last"):
    """One row per (product, entry day, hold). Day 0 is the first candle.

    Gaps inside a product's life are carried forward. If the product stops
    trading before the exit day: dead_exit='last' exits at the last close,
    'zero' books -100%. If it is alive at the cutoff and the exit day is past
    it, the event is right-censored and dropped (never marked to the cutoff).
    """
    rows = []
    for product, bars in bars_by_product.items():
        assert_cutoff(bars.index)
        if bars.empty:
            continue
        first, last = bars.index[0], bars.index[-1]
        alive = (LAST_DAY - last).days <= ALIVE_GRACE_DAYS
        close = bars["close"].reindex(pd.date_range(first, last, freq="D")).ffill()
        vol30 = (bars["close"] * bars["volume"]).loc[:first + pd.Timedelta(days=29)].sum()
        for e in entry_days:
            entry_day = first + pd.Timedelta(days=e)
            if entry_day > last:
                continue  # no entry possible before it stopped trading
            entry = close.loc[entry_day]
            for h in hold_days:
                exit_day = entry_day + pd.Timedelta(days=h)
                if exit_day <= last:
                    multiple, status = close.loc[exit_day] / entry, "held"
                elif alive:
                    continue
                else:
                    multiple, status = (close.iloc[-1] / entry if dead_exit == "last" else 0.0), "delisted"
                rows.append({
                    "product": product, "listed": first, "entry": e, "hold": h, "status": status,
                    "ret": net(multiple), "gross": multiple - 1,
                    "dollar_vol_30d": vol30,
                    "trend_at_entry": entry / close.iloc[0] - 1,
                })
    return pd.DataFrame(rows)


def top_share(returns, frac):
    """Share of total positive P&L contributed by the best `frac` of events."""
    gains = np.sort(returns[returns > 0].to_numpy())[::-1]
    if not len(gains):
        return np.nan
    k = max(1, int(np.ceil(frac * len(returns))))
    return gains[:k].sum() / gains.sum()


def distribution(ret):
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return {"n": 0}
    return {
        "n": len(ret), "share_lose_50": (ret <= -0.5).mean(), "share_lose_90": (ret <= -0.9).mean(),
        "median": ret.median(), "mean": ret.mean(),
        "top1_gain_share": top_share(ret, 0.01), "top5_gain_share": top_share(ret, 0.05),
    }


def summary(events):
    rows = {}
    for (e, h), g in events.groupby(["entry", "hold"]):
        rows[f"day{e}/hold{h}"] = distribution(g["ret"])
    return pd.DataFrame(rows).T


def split(events):
    """Discovery = listed before 2023; validation = listed 2023 onward. No reuse across the split."""
    return events[events.listed < OOS_START], events[events.listed >= OOS_START]


def fit_filters(train):
    """Thresholds come from the discovery sample only (median of each feature)."""
    base = train.drop_duplicates("product")
    return {"dollar_vol_30d": base.dollar_vol_30d.median()}


def apply_filter(events, name, params):
    if name == "high_volume":
        return events[events.dollar_vol_30d >= params["dollar_vol_30d"]]
    if name == "uptrend":
        return events[events.trend_at_entry > 0]
    if name == "high_volume_and_uptrend":
        return events[(events.dollar_vol_30d >= params["dollar_vol_30d"]) & (events.trend_at_entry > 0)]
    raise ValueError(name)


FILTERS = ("high_volume", "uptrend", "high_volume_and_uptrend")


def filter_report(events, entry=30, hold=90):
    train, test = split(events[(events.entry == entry) & (events.hold == hold)])
    params = fit_filters(train)
    rows = {"no filter": {**{"sample": "train"}, **distribution(train.ret)}}
    rows["no filter (validation)"] = {**{"sample": "validation"}, **distribution(test.ret)}
    for f in FILTERS:
        rows[f + " (train)"] = {"sample": "train", **distribution(apply_filter(train, f, params).ret)}
        rows[f + " (validation)"] = {"sample": "validation", **distribution(apply_filter(test, f, params).ret)}
    return pd.DataFrame(rows).T
