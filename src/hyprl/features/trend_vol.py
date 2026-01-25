from __future__ import annotations

import numpy as np
import pandas as pd


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    window_high = high.rolling(n, min_periods=n).max()
    window_low = low.rolling(n, min_periods=n).min()
    mid = (window_high + window_low) / 2.0
    return pd.DataFrame({"donch_high": window_high, "donch_low": window_low, "donch_mid": mid})


def hv_percentile(ret: pd.Series, w: int = 252) -> pd.Series:
    vol = ret.rolling(w, min_periods=w).std()
    ranks = vol.rolling(w, min_periods=w).rank(pct=True)
    return ranks.clip(0.0, 1.0)


def kama_slope(close: pd.Series, fast: int = 2, slow: int = 30, n: int = 10) -> pd.Series:
    er_num = close.diff(n).abs()
    er_den = close.diff().abs().rolling(n, min_periods=n).sum()
    er = er_num / er_den.replace(0.0, np.nan)

    sc_fast = 2.0 / (fast + 1.0)
    sc_slow = 2.0 / (slow + 1.0)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2

    kama = pd.Series(index=close.index, dtype=float)
    prev = close.dropna().iloc[0] if not close.dropna().empty else np.nan
    for idx, price in close.items():
        if pd.notna(price) and pd.notna(sc.loc[idx]):
            prev = prev + sc.loc[idx] * (price - prev) if pd.notna(prev) else price
            kama.loc[idx] = prev
        else:
            kama.loc[idx] = np.nan
    return kama.diff()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean().replace(0.0, np.nan)

    plus_di = 100.0 * plus_dm.rolling(n, min_periods=n).sum() / atr
    minus_di = 100.0 * minus_dm.rolling(n, min_periods=n).sum() / atr
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = (abs(plus_di - minus_di) / denom) * 100.0
    return dx.rolling(n, min_periods=n).mean()
