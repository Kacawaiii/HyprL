from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from hyprl.data.market import MarketDataFetcher
from hyprl.data.news import NewsFetcher
from hyprl.indicators.technical import compute_feature_frame
from hyprl.model.probability import ProbabilityModel
from hyprl.sentiment.analyzer import SentimentScorer
from hyprl.risk.manager import RiskConfig, RiskOutcome, plan_trade


@dataclass(slots=True)
class AnalysisConfig:
    ticker: str = "AAPL"
    interval: str = "5m"
    period: str = "5d"
    sma_short_window: int = 5  # 5 * 5m = 25 minutes
    sma_long_window: int = 36  # 36 * 5m = 3 hours
    rsi_window: int = 14
    atr_window: int = 14
    threshold: float = 0.5
    risk: RiskConfig = field(default_factory=RiskConfig)


@dataclass(slots=True)
class AnalysisResult:
    latest_probability_up: float
    latest_probability_down: float
    predicted_direction: str
    latest_row: pd.Series
    metrics: Dict[str, float]
    news_sentiment: float
    risk: RiskOutcome | None


class AnalysisPipeline:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.market_fetcher = MarketDataFetcher(config.ticker)
        self.news_fetcher = NewsFetcher(config.ticker)
        self.sentiment_scorer = SentimentScorer.default()
        self.model = ProbabilityModel.create()

    def _augment_with_sentiment(
        self, feature_df: pd.DataFrame, sentiment_score: float
    ) -> pd.DataFrame:
        feature_df = feature_df.copy()
        feature_df["sentiment_score"] = sentiment_score
        feature_df = feature_df.dropna()
        return feature_df

    def _build_training_matrices(
        self, feature_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        returns_forward = feature_df["returns_next"]
        target = (returns_forward > 0).astype(int)
        design = feature_df[["trend_ratio", "rsi_normalized", "volatility", "sentiment_score"]]
        finite_mask = np.isfinite(design).all(axis=1)
        design = design[finite_mask]
        target = target.loc[design.index]
        return design, target

    def run(self) -> AnalysisResult:
        config = self.config
        prices = self.market_fetcher.get_prices(
            interval=config.interval,
            period=config.period,
        )

        features = compute_feature_frame(
            prices,
            sma_short_window=config.sma_short_window,
            sma_long_window=config.sma_long_window,
            rsi_window=config.rsi_window,
            atr_window=config.atr_window,
        )

        headlines = self.news_fetcher.fetch_latest(limit=12)
        annotated = self.sentiment_scorer.annotate_articles(headlines)
        sentiment_scores = [article.sentiment or 0.0 for article in annotated]
        sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0

        feature_with_sentiment = self._augment_with_sentiment(features, sentiment)
        design, target = self._build_training_matrices(feature_with_sentiment)
        if design.empty or target.nunique() < 2:
            raise RuntimeError("Insufficient data variance for training the probability model.")

        _, target_vector = self.model.fit(design, target)
        probabilities = self.model.predict_proba(design)
        preds_binary = (probabilities >= self.config.threshold).astype(int)
        accuracy = float((preds_binary == target_vector.astype(int)).mean())
        positive_mask = target_vector.astype(int) == 1
        if positive_mask.any():
            shortfall = np.maximum(0.0, self.config.threshold - probabilities[positive_mask])
            expected_loss = float(shortfall.mean())
        else:
            expected_loss = 0.0
        latest_prob_up, latest_prob_down, signal = self.model.latest_prediction(design)
        predicted_direction = "UP" if signal == 1 else "DOWN"
        latest_row = feature_with_sentiment.loc[design.index[-1]]

        metrics = {
            "training_rows": float(len(design)),
            "accuracy": accuracy,
            "expected_shortfall_vs_threshold": expected_loss,
        }

        entry_price = float(latest_row["price"])
        atr_column = f"atr_{self.config.atr_window}"
        atr_value = float(latest_row.get(atr_column, 0.0))
        if atr_value <= 0.0:
            volatility = float(latest_row.get("volatility", 0.0))
            atr_value = abs(entry_price * volatility)
        trade_direction = "long" if predicted_direction == "UP" else "short"
        risk_plan = plan_trade(
            entry_price=entry_price,
            atr=atr_value,
            direction=trade_direction,
            config=self.config.risk,
        )

        return AnalysisResult(
            latest_probability_up=latest_prob_up or 0.0,
            latest_probability_down=latest_prob_down or 0.0,
            predicted_direction=predicted_direction,
            latest_row=latest_row,
            metrics=metrics,
            news_sentiment=sentiment,
            risk=risk_plan,
        )
