#!/usr/bin/env python3
"""Full Analysis Script - Real Models + All Filters.

Runs complete analysis using:
1. Real XGBoost models (v3)
2. Smart Filter
3. Sentiment Filter (optional)
4. Options Scanner
5. Multi-timeframe alignment

Usage:
    python scripts/run_full_analysis.py
    python scripts/run_full_analysis.py --live  # Fetch fresh data
    python scripts/run_full_analysis.py --symbol NVDA --shares 200
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.strategy.smart_filter import smart_filter, SmartFilterResult
from hyprl.options.income import OptionsIncomeAnalyzer, IncomeConfig, format_opportunity

# Try importing sentiment
try:
    from hyprl.sentiment.trading_filter import TradingSentimentFilter, format_sentiment
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False


FEATURE_COLS = [
    'ret_1h', 'ret_3h', 'ret_6h', 'ret_24h',
    'atr_14', 'atr_72', 'atr_14_norm', 'atr_72_norm',
    'rsi_7', 'rsi_14', 'rsi_21',
    'vol_ratio_10_30', 'vol_regime_high', 'volume_zscore_24', 'volume_surge',
    'range_pct', 'true_range', 'ret_skew_20', 'ret_kurt_20'
]

CONFIGS = {
    "NVDA": {"model": "models/nvda_1h_xgb_v3.joblib", "long": 0.53, "short": 0.45, "min_prob": 0.53},
    "MSFT": {"model": "models/msft_1h_xgb_v3.joblib", "long": 0.63, "short": 0.55, "min_prob": 0.55},
    "QQQ": {"model": "models/qqq_1h_xgb_v3.joblib", "long": 0.60, "short": 0.53, "min_prob": 0.53},
}


@dataclass
class AnalysisResult:
    symbol: str
    timestamp: datetime
    # ML
    ml_probability: float
    ml_direction: str
    ml_passed: bool
    # Smart Filter
    smart_filter_passed: bool
    smart_filter_reason: str
    # Sentiment
    sentiment_score: float
    sentiment_level: str
    sentiment_should_trade: bool
    # Final
    final_direction: str
    final_confidence: float
    recommended_size_pct: float
    # Context
    price: float
    rsi: float
    atr_norm: float
    # Options
    options_opportunity: Optional[str]

    def summary(self) -> str:
        emoji = {"long": "🟢", "short": "🔴", "flat": "⚪", "blocked": "🚫"}
        lines = [
            f"\n{'='*60}",
            f"  {emoji.get(self.final_direction, '?')} {self.symbol} - {self.final_direction.upper()}",
            f"{'='*60}",
            f"",
            f"📊 ML Model:",
            f"   Probability: {self.ml_probability:.1%}",
            f"   Direction: {self.ml_direction}",
            f"   Passed threshold: {'✅' if self.ml_passed else '❌'}",
            f"",
            f"🔍 Smart Filter:",
            f"   Passed: {'✅' if self.smart_filter_passed else '❌'}",
            f"   Reason: {self.smart_filter_reason}",
            f"",
            f"📰 Sentiment:",
            f"   Score: {self.sentiment_score:.2f}",
            f"   Level: {self.sentiment_level}",
            f"   Should trade: {'✅' if self.sentiment_should_trade else '❌'}",
            f"",
            f"🎯 Final Decision:",
            f"   Direction: {self.final_direction}",
            f"   Confidence: {self.final_confidence:.0%}",
            f"   Size: {self.recommended_size_pct:.1%}",
            f"",
            f"📈 Technicals:",
            f"   Price: ${self.price:.2f}",
            f"   RSI: {self.rsi:.0f}",
            f"   ATR%: {self.atr_norm*100:.2f}%",
        ]

        if self.options_opportunity:
            lines.extend([
                f"",
                f"💰 Options:",
                f"   {self.options_opportunity}",
            ])

        lines.append(f"{'='*60}")
        return "\n".join(lines)


def load_cached_features(symbol: str) -> Optional[pd.DataFrame]:
    """Load cached features from parquet."""
    paths = [
        f"data/cache/{symbol.lower()}_1h_features_v3.parquet",
        f"data/cache/ohlcv_{symbol}_1h_730d.parquet",
    ]
    for path in paths:
        if Path(path).exists():
            return pd.read_parquet(path)
    return None


def fetch_live_features(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch live data and compute features."""
    try:
        import yfinance as yf
        from hyprl.features.enhanced_v3 import compute_features_v3

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", interval="1h")
        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        features = compute_features_v3(df)
        return features
    except Exception as e:
        print(f"  Warning: Could not fetch live data for {symbol}: {e}")
        return None


def analyze_symbol(
    symbol: str,
    live: bool = False,
    shares: int = 0,
    enable_sentiment: bool = True,
    enable_options: bool = True,
) -> Optional[AnalysisResult]:
    """Analyze a single symbol."""

    cfg = CONFIGS.get(symbol)
    if not cfg:
        print(f"  ⚠️ No config for {symbol}")
        return None

    # Load model
    model_path = cfg["model"]
    if not Path(model_path).exists():
        print(f"  ⚠️ Model not found: {model_path}")
        return None

    model = joblib.load(model_path)

    # Load features
    if live:
        df = fetch_live_features(symbol)
    else:
        df = load_cached_features(symbol)

    if df is None or df.empty:
        print(f"  ⚠️ No data for {symbol}")
        return None

    # Get latest row
    latest = df.dropna().iloc[-1]
    feat = latest[FEATURE_COLS].values.reshape(1, -1)

    # ML prediction
    ml_prob = float(model.predict_proba(feat)[0])

    if ml_prob > cfg["long"]:
        ml_dir = "long"
    elif ml_prob < cfg["short"]:
        ml_dir = "short"
    else:
        ml_dir = "flat"

    ml_passed = ml_dir != "flat"

    # Smart filter
    closes = df['close'].values[-200:] if 'close' in df.columns else np.array([latest.get('close', 100)])

    if ml_dir != "flat":
        sf_result = smart_filter(
            ml_direction=ml_dir,
            ml_probability=ml_prob,
            closes=closes,
            min_prob=cfg["min_prob"],
        )
        sf_passed = sf_result.allowed
        sf_reason = sf_result.reason
    else:
        sf_passed = False
        sf_reason = "no_signal"

    # Sentiment
    sent_score = 0.0
    sent_level = "neutral"
    sent_should_trade = True

    if enable_sentiment and SENTIMENT_AVAILABLE and ml_dir != "flat":
        try:
            sentiment_filter = TradingSentimentFilter()
            sent_result = sentiment_filter.analyze(symbol)
            sent_score = sent_result.score
            sent_level = sent_result.level.value
            sent_should_trade = sent_result.should_trade
        except Exception:
            pass

    # Final decision
    if ml_dir == "flat":
        final_dir = "flat"
        confidence = 0
    elif not sf_passed:
        final_dir = "blocked"
        confidence = 0
    elif not sent_should_trade:
        final_dir = "blocked"
        confidence = 0
    else:
        final_dir = ml_dir
        # Confidence based on ML prob distance from threshold
        if ml_dir == "long":
            confidence = min(1.0, (ml_prob - cfg["long"]) / 0.2 + 0.5)
        else:
            confidence = min(1.0, (cfg["short"] - ml_prob) / 0.2 + 0.5)

    # Size (Kelly-inspired)
    if final_dir in ("long", "short"):
        base_size = 0.02  # 2% base
        size = base_size * confidence
        size = min(0.05, size)  # Cap at 5%
    else:
        size = 0

    # Options
    options_str = None
    if enable_options and shares >= 100 and final_dir != "blocked":
        try:
            analyzer = OptionsIncomeAnalyzer()
            price = float(latest.get('close', 100))
            iv = float(latest.get('atr_14_norm', 0.02)) * 15  # Rough IV estimate

            opp = analyzer.analyze_covered_call(
                symbol=symbol,
                stock_price=price,
                shares=shares,
                volatility=iv,
                days_to_expiry=30,
            )
            if opp and opp.recommendation != "avoid":
                options_str = f"CC ${opp.legs[0].strike:.0f} @ {opp.annualized_return:.1f}% ann, {opp.probability_profit:.0%} prob"
        except Exception:
            pass

    return AnalysisResult(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        ml_probability=ml_prob,
        ml_direction=ml_dir,
        ml_passed=ml_passed,
        smart_filter_passed=sf_passed,
        smart_filter_reason=sf_reason,
        sentiment_score=sent_score,
        sentiment_level=sent_level,
        sentiment_should_trade=sent_should_trade,
        final_direction=final_dir,
        final_confidence=confidence,
        recommended_size_pct=size,
        price=float(latest.get('close', 0)),
        rsi=float(latest.get('rsi_14', 50)),
        atr_norm=float(latest.get('atr_14_norm', 0.02)),
        options_opportunity=options_str,
    )


def main():
    parser = argparse.ArgumentParser(description="Full Analysis with Real Models")
    parser.add_argument("--symbols", default="NVDA,MSFT,QQQ",
                       help="Comma-separated symbols")
    parser.add_argument("--symbol", default=None,
                       help="Single symbol")
    parser.add_argument("--shares", type=int, default=0,
                       help="Shares held for options")
    parser.add_argument("--live", action="store_true",
                       help="Fetch live data")
    parser.add_argument("--no-sentiment", action="store_true",
                       help="Disable sentiment")
    parser.add_argument("--no-options", action="store_true",
                       help="Disable options")
    parser.add_argument("--output", default=None,
                       help="Output JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("  🚀 HyprL Full Analysis (Real Models)")
    print("=" * 70)
    print()

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print(f"📊 Analyzing: {', '.join(symbols)}")
    print(f"   Live data: {'Yes' if args.live else 'No (cached)'}")
    print(f"   Sentiment: {'Yes' if not args.no_sentiment else 'No'}")
    print(f"   Options: {'Yes' if not args.no_options else 'No'}")
    print()

    results = []
    for symbol in symbols:
        print(f"  Analyzing {symbol}...")
        result = analyze_symbol(
            symbol,
            live=args.live,
            shares=args.shares,
            enable_sentiment=not args.no_sentiment,
            enable_options=not args.no_options,
        )
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Symbol':<8} {'ML Prob':>10} {'Direction':>10} {'Smart':>8} {'Sent':>8} {'Final':>10} {'Size':>8}")
    print("-" * 80)

    for r in results:
        sf = "✅" if r.smart_filter_passed else "❌"
        se = "✅" if r.sentiment_should_trade else "❌"
        print(f"{r.symbol:<8} {r.ml_probability:>10.1%} {r.ml_direction:>10} {sf:>8} {se:>8} "
              f"{r.final_direction:>10} {r.recommended_size_pct:>8.1%}")

    print("-" * 80)

    # Detailed output
    for r in results:
        print(r.summary())

    # Recommendations
    longs = [r for r in results if r.final_direction == "long"]
    shorts = [r for r in results if r.final_direction == "short"]

    print("\n" + "=" * 60)
    print("  🎯 Trading Recommendations")
    print("=" * 60)

    if longs:
        print("\n  📈 LONG:")
        for r in longs:
            print(f"    {r.symbol}: {r.recommended_size_pct:.1%} position, {r.final_confidence:.0%} confidence")
            print(f"       ML: {r.ml_probability:.1%} | RSI: {r.rsi:.0f} | Price: ${r.price:.2f}")

    if shorts:
        print("\n  📉 SHORT:")
        for r in shorts:
            print(f"    {r.symbol}: {r.recommended_size_pct:.1%} position, {r.final_confidence:.0%} confidence")
            print(f"       ML: {r.ml_probability:.1%} | RSI: {r.rsi:.0f} | Price: ${r.price:.2f}")

    if not longs and not shorts:
        print("\n  ⏸️  No actionable signals right now.")
        blocked = [r for r in results if r.final_direction == "blocked"]
        if blocked:
            print(f"\n  Blocked signals ({len(blocked)}):")
            for r in blocked:
                print(f"    {r.symbol}: {r.smart_filter_reason}")

    # Save
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "symbol": r.symbol,
                    "ml_probability": r.ml_probability,
                    "ml_direction": r.ml_direction,
                    "final_direction": r.final_direction,
                    "confidence": r.final_confidence,
                    "size_pct": r.recommended_size_pct,
                    "smart_filter": r.smart_filter_reason,
                    "sentiment": r.sentiment_level,
                    "price": r.price,
                }
                for r in results
            ]
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\n💾 Saved to {args.output}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
