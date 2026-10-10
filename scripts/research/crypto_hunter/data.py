"""Read cached Coinbase six-field candles. This module never uses the network."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def protocol() -> dict:
    return json.loads(Path(__file__).with_name("protocol.json").read_text())


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Market:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    evidence: dict


def from_frames(frames: dict[str, pd.DataFrame], evidence: dict | None = None) -> Market:
    if not frames:
        raise ValueError("no valid candles")
    start = min(f.index.min() for f in frames.values())
    end = max(f.index.max() for f in frames.values())
    index = pd.date_range(start, end, freq="D")
    panels = {k: pd.DataFrame({s: f[k] for s, f in sorted(frames.items())}, index=index)
              for k in ("open", "high", "low", "close", "volume")}
    return Market(**panels, evidence=evidence or {})


def load_cache(cache: Path, cutoff: str) -> Market:
    """Dates later than cutoff are discarded before indicators or outcomes exist."""
    frames, inventory, rejected = {}, [], 0
    end = pd.Timestamp(cutoff)
    excluded = set(protocol()["exclude_bases"])
    for path in sorted((cache / "coinbase").glob("*-USD/complete.json")):
        symbol = path.parent.name
        raw = path.read_bytes()
        blob = json.loads(raw)
        inventory.append([symbol, hashlib.sha256(raw).hexdigest()])
        if symbol[:-4] in excluded or not blob.get("rows"):
            continue
        a = np.asarray(blob["rows"], dtype=float)
        if a.ndim != 2 or a.shape[1] != 6:
            raise ValueError(f"invalid six-field candle shape: {symbol}")
        dates = pd.to_datetime(a[:, 0], unit="s")
        f = pd.DataFrame(a[:, 1:], index=dates, columns=["low", "high", "open", "close", "volume"])
        f = f.loc[f.index <= end].sort_index()
        if f.empty:
            continue
        if f.index.has_duplicates or not (f.index == f.index.normalize()).all():
            raise ValueError(f"duplicate or non-daily timestamp: {symbol}")
        valid = (np.isfinite(f).all(axis=1) & (f[["open", "close", "low", "high"]] > 0).all(axis=1)
                 & (f.volume >= 0) & (f.low <= f[["open", "close"]].min(axis=1))
                 & (f.high >= f[["open", "close"]].max(axis=1)))
        rejected += int((~valid).sum())
        f.loc[~valid, :] = np.nan
        frames[symbol] = f
    return from_frames(frames, {"cache_digest": canonical_hash({"files": inventory}),
                                "files": len(inventory), "coins": len(frames),
                                "rejected_candles": rejected, "cutoff": cutoff})


def features(m: Market, p: dict | None = None) -> dict[str, pd.DataFrame | pd.Series]:
    p = p or protocol()
    c = m.close
    if "BTC-USD" not in c:
        raise ValueError("BTC-USD required for regime and relative strength")
    dv = c * m.volume
    first = pd.Series({s: c[s].first_valid_index() for s in c})
    age = pd.DataFrame({s: (c.index - first[s]).days.astype(float) for s in c}, index=c.index)
    age = age.where(c.notna())
    ma = c.rolling(200, min_periods=200).mean()
    vol = np.log(c / c.shift(1))
    ret = {n: c / c.shift(n) - 1 for n in (30, 90, 180)}
    eligible = ((dv.rolling(30, min_periods=30).mean() >= p["liquidity_usd_daily"])
                & (age >= p["minimum_age_days"])
                & (c.notna().rolling(365).sum() >= p["minimum_observations_365"])
                & ma.notna() & ret[90].notna() & m.open.notna())
    breadth = ((c > ma) & eligible).sum(axis=1) / eligible.sum(axis=1).replace(0, np.nan)
    return {"dollar_volume30": dv.rolling(30, min_periods=30).mean(), "age": age,
            **{f"return{n}": x for n, x in ret.items()},
            "relative90": ret[90].sub(ret[90]["BTC-USD"], axis=0),
            "high_distance": c / c.shift(1).rolling(365, min_periods=350).max() - 1,
            "ath_drawdown": c / c.cummax() - 1,
            "contraction": vol.rolling(14, min_periods=14).std() / vol.rolling(60, min_periods=60).std(),
            "surge": dv / dv.shift(1).rolling(30, min_periods=30).mean(),
            "volatility60": vol.rolling(60, min_periods=60).std() * np.sqrt(365),
            "above200": c > ma, "btc_regime": (c["BTC-USD"] > ma["BTC-USD"]),
            "eligible": eligible, "breadth": breadth, "breadth7": breadth.rolling(7).mean()}


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    observed = close.notna().rolling(horizon + 1, min_periods=horizon + 1).sum().shift(-horizon)
    return (close.shift(-horizon) / close - 1).where(observed >= 0.95 * (horizon + 1))
