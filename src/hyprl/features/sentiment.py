from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SentimentFeatureConfig:
    window: int = 50
    min_periods: int = 5
    fear_z: float = -1.5
    greed_z: float = 1.5
    score_column: str = "sentiment_score"
    volume_column: str = "sentiment_volume"


def enrich_sentiment_features(
    frame: pd.DataFrame,
    config: SentimentFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Add derived sentiment indicators (z-score, volume proxy, extreme flags).

    Parameters
    ----------
    frame:
        Feature frame containing at least a `sentiment_score` column.
    config:
        Optional overrides for rolling window / thresholds / column names.

    Returns
    -------
    pd.DataFrame
        Copy of the frame with `sentiment_zscore`, `sentiment_volume`,
        `extreme_fear_flag`, and `extreme_greed_flag` columns appended.
    """

    if config is None:
        config = SentimentFeatureConfig()

    df = frame.copy()
    score_col = config.score_column
    if score_col not in df:
        df[score_col] = 0.0

    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    window = max(config.window, 5)
    min_periods = max(config.min_periods, 5)
    rolling = scores.rolling(window=window, min_periods=min_periods)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=0).replace(0.0, np.nan)
    zscores = ((scores - rolling_mean) / rolling_std).fillna(0.0)
    df["sentiment_zscore"] = zscores

    volume_col = config.volume_column
    if volume_col in df:
        volume = pd.to_numeric(df[volume_col], errors="coerce").fillna(0.0)
    else:
        volume = pd.Series(1.0, index=df.index)
    df["sentiment_volume"] = volume

    df["extreme_fear_flag"] = (df["sentiment_zscore"] <= config.fear_z).astype(int)
    df["extreme_greed_flag"] = (df["sentiment_zscore"] >= config.greed_z).astype(int)
    return df
