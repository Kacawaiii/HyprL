from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.features.crossasset import enrich_with_crossasset
from hyprl.features.trend_vol import donchian, hv_percentile


def test_crossasset_enrichment_shapes() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="H")
    df = pd.DataFrame({"close": np.linspace(100, 101, len(idx))}, index=idx)
    ctx = {"SPY": pd.Series(np.linspace(400, 401, len(idx)), index=idx)}
    enriched = enrich_with_crossasset(df, ctx, win=3, shift=1)

    expected = {"SPY_ret", "SPY_z", "corr_SPY_win3", "ratio_close_SPY"}
    assert expected.issubset(set(enriched.columns))
    assert not enriched.isna().all().all()


def test_trend_vol_helpers() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="H")
    high = pd.Series(100 + np.sin(np.linspace(0, 3, len(idx))), index=idx)
    low = high - 0.5
    close = (high + low) / 2.0
    don = donchian(high, low, n=5)
    assert set(["donch_high", "donch_low", "donch_mid"]).issubset(don.columns)
    ret = close.pct_change()
    hv = hv_percentile(ret, w=20).dropna()
    assert ((hv >= 0.0) & (hv <= 1.0)).all()
