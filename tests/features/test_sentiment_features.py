from __future__ import annotations

import pandas as pd

from hyprl.features.sentiment import SentimentFeatureConfig, enrich_sentiment_features


def test_enrich_sentiment_features_basic() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="h")
    frame = pd.DataFrame(
        {
            "sentiment_score": [0.0, 0.1, 0.2, 0.4, 0.6, 0.9, -0.5, -0.8, -0.2, 0.0],
            "price": list(range(10)),
        },
        index=index,
    )
    config = SentimentFeatureConfig(window=3, fear_z=-0.5, greed_z=0.5)
    enriched = enrich_sentiment_features(frame, config)
    assert "sentiment_zscore" in enriched.columns
    assert "sentiment_volume" in enriched.columns
    assert "extreme_fear_flag" in enriched.columns
    assert "extreme_greed_flag" in enriched.columns
    assert enriched["sentiment_volume"].iloc[0] == 1.0
    assert enriched["extreme_greed_flag"].iloc[5] == 1
    assert enriched["extreme_fear_flag"].iloc[7] == 1
