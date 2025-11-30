from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _zscore(series: pd.Series, win: int) -> pd.Series:
    rolling = series.rolling(win, min_periods=win)
    mean = rolling.mean()
    std = rolling.std().replace(0.0, np.nan)
    return (series - mean) / std


def _rolling_corr(lhs: pd.Series, rhs: pd.Series, win: int) -> pd.Series:
    rolling = lhs.rolling(win, min_periods=win)
    return rolling.corr(rhs)


def enrich_with_crossasset(
    df: pd.DataFrame,
    ctx: Dict[str, pd.Series],
    win: int = 50,
    shift: int = 1,
) -> pd.DataFrame:
    """
    Add cross-asset statistics (shifted to avoid leakage) to the target OHLCV frame.
    """
    if df.empty or not ctx:
        return df.copy()
    out = df.copy()
    target_ret = out["close"].pct_change()
    for name, series in ctx.items():
        aligned = series.reindex(out.index).ffill()
        shifted = aligned.shift(shift)
        ret = shifted.pct_change()
        out[f"{name}_ret"] = ret
        out[f"{name}_z"] = _zscore(shifted, win=win)
        out[f"corr_{name}_win{win}"] = _rolling_corr(target_ret, ret, win=win)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"ratio_close_{name}"] = out["close"] / shifted.replace(0.0, np.nan)
    return out
