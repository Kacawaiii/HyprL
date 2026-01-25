import pandas as pd

from hyprl.features.equity_v2 import compute_equity_v2_features


def _synthetic_prices(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    import numpy as np

    rng = np.random.default_rng(42)
    base = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, rows)), index=idx, dtype=float)
    vol = pd.Series(
        1_000_000 + 50_000 * np.sin(np.linspace(0, 6.28, rows)) + 10_000 * rng.standard_normal(rows),
        index=idx,
        dtype=float,
    )
    return pd.DataFrame(
        {
            "open": base * 0.999,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": vol,
        }
    )


def test_equity_v2_features_non_empty_and_columns_present() -> None:
    prices = _synthetic_prices()
    feats = compute_equity_v2_features(prices)
    required = [
        "ret_1h",
        "atr_14",
        "atr_72",
        "rsi_7",
        "vol_ratio_10_30",
        "volume_zscore_24",
        "range_pct",
        "true_range",
        "ret_skew_20",
        "ret_kurt_20",
    ]
    assert not feats.empty
    for col in required:
        assert col in feats.columns, f"missing column {col}"
