"""Enhanced Signal Pipeline - All Filters Combined.

Integrates:
1. ML Probability Model (XGBoost)
2. Smart Filter (momentum/RSI rules)
3. Sentiment Filter (news-based)
4. Signal Quality Filter
5. Multi-Timeframe Fusion
6. Options Income Scanner

Usage:
    from hyprl.strategy.enhanced_pipeline import EnhancedPipeline

    pipeline = EnhancedPipeline()
    result = pipeline.analyze("NVDA")
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import json

import numpy as np
import pandas as pd


class SignalStrength(Enum):
    STRONG_LONG = "strong_long"
    LONG = "long"
    WEAK_LONG = "weak_long"
    NEUTRAL = "neutral"
    WEAK_SHORT = "weak_short"
    SHORT = "short"
    STRONG_SHORT = "strong_short"
    BLOCKED = "blocked"


@dataclass
class FilterResult:
    """Result from a single filter."""
    name: str
    passed: bool
    score: float  # 0-1
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class OptionsOpportunity:
    """Options income opportunity."""
    strategy: str
    strike: float
    premium: float
    annualized_return: float
    probability_profit: float
    days_to_expiry: int
    recommendation: str


@dataclass
class PipelineResult:
    """Complete pipeline analysis result."""
    symbol: str
    timestamp: datetime

    # ML Signal
    ml_probability: float
    ml_direction: str
    ml_passed: bool

    # Filters
    filters: list[FilterResult]
    all_filters_passed: bool

    # Final Decision
    final_direction: str
    final_strength: SignalStrength
    confidence: float  # 0-1

    # Sizing
    recommended_size_pct: float
    max_risk_pct: float

    # Multi-Timeframe
    mtf_alignment: float  # -1 to 1
    timeframe_signals: dict  # {"1h": "long", "15m": "long", "4h": "neutral"}

    # Sentiment
    sentiment_score: float
    sentiment_level: str
    news_count: int

    # Options
    options_opportunities: list[OptionsOpportunity]

    # Metadata
    price: float
    atr: float
    rsi: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        emoji = {
            SignalStrength.STRONG_LONG: "🟢🟢",
            SignalStrength.LONG: "🟢",
            SignalStrength.WEAK_LONG: "🟡",
            SignalStrength.NEUTRAL: "⚪",
            SignalStrength.WEAK_SHORT: "🟡",
            SignalStrength.SHORT: "🔴",
            SignalStrength.STRONG_SHORT: "🔴🔴",
            SignalStrength.BLOCKED: "🚫",
        }

        lines = [
            f"\n{'='*60}",
            f"  {emoji.get(self.final_strength, '?')} {self.symbol} - {self.final_strength.value.upper()}",
            f"{'='*60}",
            f"",
            f"📊 ML Signal:",
            f"   Direction: {self.ml_direction} | Probability: {self.ml_probability:.1%}",
            f"   Passed threshold: {'✅' if self.ml_passed else '❌'}",
            f"",
            f"🔍 Filters ({sum(1 for f in self.filters if f.passed)}/{len(self.filters)} passed):",
        ]

        for f in self.filters:
            status = "✅" if f.passed else "❌"
            lines.append(f"   {status} {f.name}: {f.reason}")

        lines.extend([
            f"",
            f"📈 Multi-Timeframe:",
            f"   Alignment: {self.mtf_alignment:+.2f}",
        ])
        for tf, sig in self.timeframe_signals.items():
            lines.append(f"   {tf}: {sig}")

        lines.extend([
            f"",
            f"📰 Sentiment:",
            f"   Level: {self.sentiment_level} | Score: {self.sentiment_score:.2f}",
            f"   News count: {self.news_count}",
            f"",
            f"💰 Position Sizing:",
            f"   Recommended: {self.recommended_size_pct:.1%} of portfolio",
            f"   Max Risk: {self.max_risk_pct:.1%}",
            f"   Confidence: {self.confidence:.0%}",
        ])

        if self.options_opportunities:
            lines.extend([
                f"",
                f"📋 Options Opportunities:",
            ])
            for opp in self.options_opportunities[:2]:
                lines.append(
                    f"   {opp.strategy}: ${opp.strike} strike, "
                    f"{opp.annualized_return:.1f}% ann., "
                    f"{opp.probability_profit:.0%} prob"
                )

        lines.extend([
            f"",
            f"📉 Technicals:",
            f"   Price: ${self.price:.2f} | ATR: ${self.atr:.2f} | RSI: {self.rsi:.0f}",
            f"{'='*60}",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "ml_probability": self.ml_probability,
            "ml_direction": self.ml_direction,
            "final_direction": self.final_direction,
            "final_strength": self.final_strength.value,
            "confidence": self.confidence,
            "recommended_size_pct": self.recommended_size_pct,
            "mtf_alignment": self.mtf_alignment,
            "sentiment_score": self.sentiment_score,
            "sentiment_level": self.sentiment_level,
            "all_filters_passed": self.all_filters_passed,
            "filters": [{"name": f.name, "passed": f.passed, "reason": f.reason} for f in self.filters],
            "price": self.price,
            "atr": self.atr,
            "rsi": self.rsi,
        }


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # ML Thresholds
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    min_probability: float = 0.52

    # Smart Filter
    enable_smart_filter: bool = True
    falling_knife_mom5: float = -6.0
    falling_knife_mom10: float = -10.0

    # Sentiment
    enable_sentiment: bool = True
    sentiment_lookback_hours: int = 24
    block_very_bearish: bool = True

    # Multi-Timeframe
    enable_mtf: bool = True
    mtf_frames: list[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    mtf_weights: dict[str, float] = field(default_factory=lambda: {"15m": 0.2, "1h": 0.6, "4h": 0.2})

    # Signal Quality
    min_quality_score: float = 0.5

    # Sizing
    base_size_pct: float = 0.02  # 2% base
    max_size_pct: float = 0.05   # 5% max
    conviction_multiplier: float = 1.5

    # Options
    enable_options_scan: bool = True
    min_shares_for_options: int = 100
    min_annualized_return: float = 12.0


class EnhancedPipeline:
    """Enhanced signal pipeline with all filters."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._models = {}
        self._sentiment_filter = None
        self._options_analyzer = None

    def _load_model(self, symbol: str):
        """Load ML model for symbol."""
        if symbol in self._models:
            return self._models[symbol]

        import joblib

        model_paths = [
            f"models/{symbol.lower()}_1h_xgb_v3.joblib",
            f"models/{symbol.lower().replace('/', '_')}_xgb.joblib",
            f"models/crypto/{symbol.lower().replace('/', '_')}_xgb.joblib",
        ]

        for path in model_paths:
            if Path(path).exists():
                model = joblib.load(path)
                self._models[symbol] = model
                return model

        return None

    def _get_sentiment_filter(self):
        """Get or create sentiment filter."""
        if self._sentiment_filter is None:
            try:
                from hyprl.sentiment.trading_filter import TradingSentimentFilter, SentimentConfig
                self._sentiment_filter = TradingSentimentFilter(SentimentConfig(
                    news_lookback_hours=self.config.sentiment_lookback_hours,
                    block_on_very_bearish=self.config.block_very_bearish,
                ))
            except ImportError:
                pass
        return self._sentiment_filter

    def _get_options_analyzer(self):
        """Get or create options analyzer."""
        if self._options_analyzer is None:
            try:
                from hyprl.options.income import OptionsIncomeAnalyzer, IncomeConfig
                self._options_analyzer = OptionsIncomeAnalyzer(IncomeConfig(
                    min_annualized_return=self.config.min_annualized_return,
                ))
            except ImportError:
                pass
        return self._options_analyzer

    def _fetch_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Fetch OHLCV data."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol.replace("/", "-"))
            df = ticker.history(period=f"{days}d", interval="1h")
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            return pd.DataFrame()

    def _compute_features(self, df: pd.DataFrame) -> dict:
        """Compute technical features."""
        if df.empty or len(df) < 50:
            return {}

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Returns
        ret_1h = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
        ret_24h = (close[-1] - close[-25]) / close[-25] if len(close) > 24 else 0

        # Momentum
        mom_5 = (close[-1] - close[-6]) / close[-6] * 100 if len(close) > 5 else 0
        mom_10 = (close[-1] - close[-11]) / close[-11] * 100 if len(close) > 10 else 0
        mom_20 = (close[-1] - close[-21]) / close[-21] * 100 if len(close) > 20 else 0

        # ATR (simplified)
        tr_hl = high[-14:] - low[-14:]
        atr_14 = np.mean(tr_hl) if len(tr_hl) > 0 else 0

        # RSI
        deltas = np.diff(close[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) + 1e-10
        avg_loss = np.mean(losses) + 1e-10
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))

        # Volume
        vol_avg = np.mean(volume[-20:]) if len(volume) > 20 else volume[-1]
        vol_ratio = volume[-1] / vol_avg if vol_avg > 0 else 1

        # Consecutive bars
        recent = close[-6:]
        down_bars = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        up_bars = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])

        return {
            "price": close[-1],
            "ret_1h": ret_1h,
            "ret_24h": ret_24h,
            "mom_5": mom_5,
            "mom_10": mom_10,
            "mom_20": mom_20,
            "atr_14": atr_14,
            "atr_14_norm": atr_14 / close[-1] if close[-1] > 0 else 0,
            "rsi_14": rsi_14,
            "vol_ratio": vol_ratio,
            "down_bars": down_bars,
            "up_bars": up_bars,
            "closes": close,
        }

    def _apply_smart_filter(self, direction: str, prob: float, features: dict) -> FilterResult:
        """Apply smart filter rules."""
        if not self.config.enable_smart_filter:
            return FilterResult("smart_filter", True, 1.0, "disabled")

        mom_5 = features.get("mom_5", 0)
        mom_10 = features.get("mom_10", 0)
        rsi = features.get("rsi_14", 50)
        down_bars = features.get("down_bars", 0)
        up_bars = features.get("up_bars", 0)

        # Low probability
        if prob < self.config.min_probability:
            return FilterResult("smart_filter", False, 0.0,
                              f"low_prob({prob:.2f} < {self.config.min_probability})")

        if direction == "long":
            # Falling knife
            if mom_5 < self.config.falling_knife_mom5 and mom_10 < self.config.falling_knife_mom10:
                return FilterResult("smart_filter", False, 0.2,
                                  f"falling_knife(mom5={mom_5:.1f}%, mom10={mom_10:.1f}%)")
            # Capitulation
            if down_bars >= 5:
                return FilterResult("smart_filter", False, 0.3,
                                  f"capitulation({down_bars}_down_bars)")
            # Extreme panic
            if rsi < 15:
                return FilterResult("smart_filter", False, 0.2,
                                  f"extreme_panic(rsi={rsi:.0f})")

        elif direction == "short":
            # Pump in progress
            if mom_5 > 6 and mom_10 > 10:
                return FilterResult("smart_filter", False, 0.2,
                                  f"pump_in_progress(mom5={mom_5:.1f}%, mom10={mom_10:.1f}%)")
            # Melt-up
            if up_bars >= 5:
                return FilterResult("smart_filter", False, 0.3,
                                  f"meltup({up_bars}_up_bars)")
            # Extreme FOMO
            if rsi > 85:
                return FilterResult("smart_filter", False, 0.2,
                                  f"extreme_fomo(rsi={rsi:.0f})")

        return FilterResult("smart_filter", True, 1.0,
                          f"passed(mom5={mom_5:.1f}%, rsi={rsi:.0f})")

    def _apply_sentiment_filter(self, symbol: str, direction: str) -> FilterResult:
        """Apply sentiment filter."""
        if not self.config.enable_sentiment:
            return FilterResult("sentiment", True, 1.0, "disabled", {"level": "neutral", "score": 0})

        sentiment_filter = self._get_sentiment_filter()
        if sentiment_filter is None:
            return FilterResult("sentiment", True, 1.0, "unavailable", {"level": "neutral", "score": 0})

        try:
            result = sentiment_filter.analyze(symbol)

            # Conflict detection
            if direction == "long" and result.level.value in ("very_bearish", "bearish"):
                if result.level.value == "very_bearish" and self.config.block_very_bearish:
                    return FilterResult("sentiment", False, 0.0,
                                      f"blocked({result.level.value})",
                                      {"level": result.level.value, "score": result.score, "news": result.news_count})
                return FilterResult("sentiment", True, 0.5,
                                  f"conflict(long vs {result.level.value})",
                                  {"level": result.level.value, "score": result.score, "news": result.news_count})

            if direction == "short" and result.level.value in ("very_bullish", "bullish"):
                return FilterResult("sentiment", True, 0.5,
                                  f"conflict(short vs {result.level.value})",
                                  {"level": result.level.value, "score": result.score, "news": result.news_count})

            return FilterResult("sentiment", True, result.signal_modifier,
                              f"{result.level.value}({result.score:.2f})",
                              {"level": result.level.value, "score": result.score, "news": result.news_count})
        except Exception as e:
            return FilterResult("sentiment", True, 1.0, f"error({e})", {"level": "neutral", "score": 0})

    def _apply_quality_filter(self, features: dict) -> FilterResult:
        """Apply signal quality filter."""
        vol_ratio = features.get("vol_ratio", 1)
        atr_norm = features.get("atr_14_norm", 0.02)
        rsi = features.get("rsi_14", 50)

        score = 0.5  # Base

        # Volume bonus
        if vol_ratio > 1.5:
            score += 0.15
        elif vol_ratio > 1.2:
            score += 0.1
        elif vol_ratio < 0.5:
            score -= 0.15

        # Volatility (prefer moderate)
        if 0.01 < atr_norm < 0.03:
            score += 0.1
        elif atr_norm > 0.05:
            score -= 0.1

        # RSI extremes (less reliable)
        if rsi < 20 or rsi > 80:
            score -= 0.1

        score = max(0, min(1, score))

        if score < self.config.min_quality_score:
            return FilterResult("quality", False, score,
                              f"low_quality({score:.2f} < {self.config.min_quality_score})")

        return FilterResult("quality", True, score, f"quality_ok({score:.2f})")

    def _compute_mtf_alignment(self, symbol: str) -> tuple[float, dict]:
        """Compute multi-timeframe alignment."""
        if not self.config.enable_mtf:
            return 0.0, {}

        signals = {}
        alignment = 0.0

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol.replace("/", "-"))

            for tf in self.config.mtf_frames:
                interval = tf
                if tf == "4h":
                    interval = "1h"  # yfinance doesn't have 4h, we'll aggregate

                df = ticker.history(period="30d", interval=interval)
                if df.empty:
                    signals[tf] = "neutral"
                    continue

                close = df['Close'].values

                # Simple trend detection
                sma_10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
                sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]

                current = close[-1]

                if current > sma_10 > sma_20:
                    signals[tf] = "long"
                    alignment += self.config.mtf_weights.get(tf, 0.33)
                elif current < sma_10 < sma_20:
                    signals[tf] = "short"
                    alignment -= self.config.mtf_weights.get(tf, 0.33)
                else:
                    signals[tf] = "neutral"

        except Exception:
            pass

        return alignment, signals

    def _scan_options(self, symbol: str, price: float, shares: int = 0,
                     volatility: float = 0.30) -> list[OptionsOpportunity]:
        """Scan for options opportunities."""
        if not self.config.enable_options_scan:
            return []

        if shares < self.config.min_shares_for_options:
            return []

        analyzer = self._get_options_analyzer()
        if analyzer is None:
            return []

        opportunities = []

        try:
            # Covered call
            cc_result = analyzer.analyze_covered_call(
                symbol=symbol,
                stock_price=price,
                shares=shares,
                volatility=volatility,
                days_to_expiry=30,
            )

            if cc_result and cc_result.recommendation != "avoid":
                opportunities.append(OptionsOpportunity(
                    strategy="covered_call",
                    strike=cc_result.legs[0].strike,
                    premium=cc_result.total_premium,
                    annualized_return=cc_result.annualized_return,
                    probability_profit=cc_result.probability_profit,
                    days_to_expiry=cc_result.days_to_expiry,
                    recommendation=cc_result.recommendation,
                ))
        except Exception:
            pass

        return opportunities

    def _determine_final_signal(self, ml_dir: str, ml_prob: float,
                               filters: list[FilterResult],
                               mtf_alignment: float) -> tuple[str, SignalStrength, float]:
        """Determine final signal and strength."""

        # Check if blocked
        if not all(f.passed for f in filters):
            return "flat", SignalStrength.BLOCKED, 0.0

        # Base confidence from ML
        if ml_dir == "long":
            base_conf = (ml_prob - 0.5) * 2  # Scale 0.5-1.0 to 0-1
        elif ml_dir == "short":
            base_conf = (0.5 - ml_prob) * 2  # Scale 0-0.5 to 0-1
        else:
            return "flat", SignalStrength.NEUTRAL, 0.0

        # Adjust by filter scores
        avg_filter_score = np.mean([f.score for f in filters])
        confidence = base_conf * avg_filter_score

        # Adjust by MTF alignment
        if ml_dir == "long" and mtf_alignment > 0:
            confidence *= (1 + mtf_alignment * 0.2)
        elif ml_dir == "short" and mtf_alignment < 0:
            confidence *= (1 + abs(mtf_alignment) * 0.2)
        elif (ml_dir == "long" and mtf_alignment < -0.3) or (ml_dir == "short" and mtf_alignment > 0.3):
            confidence *= 0.7  # Penalize MTF conflict

        confidence = min(1.0, confidence)

        # Determine strength
        if ml_dir == "long":
            if confidence >= 0.7:
                strength = SignalStrength.STRONG_LONG
            elif confidence >= 0.5:
                strength = SignalStrength.LONG
            elif confidence >= 0.3:
                strength = SignalStrength.WEAK_LONG
            else:
                strength = SignalStrength.NEUTRAL
        else:
            if confidence >= 0.7:
                strength = SignalStrength.STRONG_SHORT
            elif confidence >= 0.5:
                strength = SignalStrength.SHORT
            elif confidence >= 0.3:
                strength = SignalStrength.WEAK_SHORT
            else:
                strength = SignalStrength.NEUTRAL

        return ml_dir, strength, confidence

    def _compute_sizing(self, confidence: float, atr_norm: float) -> tuple[float, float]:
        """Compute position sizing."""
        # Base size adjusted by confidence
        size_pct = self.config.base_size_pct * (1 + confidence * self.config.conviction_multiplier)

        # Reduce if high volatility
        if atr_norm > 0.03:
            size_pct *= 0.7
        elif atr_norm > 0.04:
            size_pct *= 0.5

        size_pct = min(size_pct, self.config.max_size_pct)

        # Risk based on ATR
        risk_pct = atr_norm * 1.5 * size_pct  # 1.5 ATR stop

        return size_pct, risk_pct

    def analyze(self, symbol: str, shares_held: int = 0) -> PipelineResult:
        """Run complete pipeline analysis."""
        timestamp = datetime.now(timezone.utc)

        # Fetch data
        df = self._fetch_data(symbol)
        if df.empty:
            return self._empty_result(symbol, timestamp, "no_data")

        # Compute features
        features = self._compute_features(df)
        if not features:
            return self._empty_result(symbol, timestamp, "insufficient_data")

        # Load model and get prediction
        model = self._load_model(symbol)
        ml_prob = 0.5
        ml_dir = "flat"

        if model is not None:
            try:
                # Prepare features for model
                feature_cols = [
                    'ret_1h', 'ret_24h', 'atr_14', 'atr_14_norm',
                    'rsi_14', 'vol_ratio', 'mom_5', 'mom_10', 'mom_20'
                ]
                feat_vector = np.array([[
                    features.get('ret_1h', 0),
                    features.get('ret_24h', 0),
                    features.get('atr_14', 0),
                    features.get('atr_14_norm', 0),
                    features.get('rsi_14', 50),
                    features.get('vol_ratio', 1),
                    features.get('mom_5', 0) / 100,
                    features.get('mom_10', 0) / 100,
                    features.get('mom_20', 0) / 100,
                ]])

                proba = model.predict_proba(feat_vector)
                if hasattr(proba, '__len__') and len(proba) > 0:
                    ml_prob = float(proba[0] if proba.ndim == 1 else proba[0, 1])
            except Exception:
                pass

        # Determine ML direction
        if ml_prob > self.config.long_threshold:
            ml_dir = "long"
        elif ml_prob < self.config.short_threshold:
            ml_dir = "short"
        else:
            ml_dir = "flat"

        ml_passed = ml_dir != "flat"

        # Apply filters
        filters = []

        if ml_dir != "flat":
            filters.append(self._apply_smart_filter(ml_dir, ml_prob, features))
            filters.append(self._apply_sentiment_filter(symbol, ml_dir))
            filters.append(self._apply_quality_filter(features))

        all_passed = all(f.passed for f in filters) if filters else False

        # Multi-timeframe
        mtf_alignment, timeframe_signals = self._compute_mtf_alignment(symbol)

        # Final decision
        final_dir, strength, confidence = self._determine_final_signal(
            ml_dir, ml_prob, filters, mtf_alignment
        )

        # Sizing
        size_pct, risk_pct = self._compute_sizing(
            confidence, features.get('atr_14_norm', 0.02)
        )

        # Sentiment details
        sentiment_filter = next((f for f in filters if f.name == "sentiment"), None)
        sentiment_score = sentiment_filter.details.get("score", 0) if sentiment_filter else 0
        sentiment_level = sentiment_filter.details.get("level", "neutral") if sentiment_filter else "neutral"
        news_count = sentiment_filter.details.get("news", 0) if sentiment_filter else 0

        # Options scan
        options = self._scan_options(
            symbol, features.get("price", 100),
            shares_held, features.get("atr_14_norm", 0.02) * 20  # Rough IV estimate
        )

        return PipelineResult(
            symbol=symbol,
            timestamp=timestamp,
            ml_probability=ml_prob,
            ml_direction=ml_dir,
            ml_passed=ml_passed,
            filters=filters,
            all_filters_passed=all_passed,
            final_direction=final_dir,
            final_strength=strength,
            confidence=confidence,
            recommended_size_pct=size_pct if all_passed else 0,
            max_risk_pct=risk_pct if all_passed else 0,
            mtf_alignment=mtf_alignment,
            timeframe_signals=timeframe_signals,
            sentiment_score=sentiment_score,
            sentiment_level=sentiment_level,
            news_count=news_count,
            options_opportunities=options,
            price=features.get("price", 0),
            atr=features.get("atr_14", 0),
            rsi=features.get("rsi_14", 50),
        )

    def _empty_result(self, symbol: str, timestamp: datetime, reason: str) -> PipelineResult:
        """Create empty result for error cases."""
        return PipelineResult(
            symbol=symbol,
            timestamp=timestamp,
            ml_probability=0.5,
            ml_direction="flat",
            ml_passed=False,
            filters=[FilterResult("data", False, 0, reason)],
            all_filters_passed=False,
            final_direction="flat",
            final_strength=SignalStrength.BLOCKED,
            confidence=0,
            recommended_size_pct=0,
            max_risk_pct=0,
            mtf_alignment=0,
            timeframe_signals={},
            sentiment_score=0,
            sentiment_level="neutral",
            news_count=0,
            options_opportunities=[],
            price=0,
            atr=0,
            rsi=50,
        )

    def scan_portfolio(self, symbols: list[str],
                       positions: Optional[dict[str, int]] = None) -> list[PipelineResult]:
        """Scan multiple symbols."""
        positions = positions or {}
        results = []

        for symbol in symbols:
            shares = positions.get(symbol, 0)
            result = self.analyze(symbol, shares)
            results.append(result)

        # Sort by confidence (strongest signals first)
        return sorted(results, key=lambda x: x.confidence, reverse=True)
