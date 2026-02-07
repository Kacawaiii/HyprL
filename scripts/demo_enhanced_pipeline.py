#!/usr/bin/env python3
"""Demo: Enhanced Signal Pipeline with All Filters.

Run locally without VPS - shows:
1. ML predictions
2. Smart Filter rules
3. Sentiment from news
4. Multi-timeframe alignment
5. Options income opportunities
6. Final recommendation

Usage:
    python scripts/demo_enhanced_pipeline.py
    python scripts/demo_enhanced_pipeline.py --symbols NVDA,MSFT,QQQ
    python scripts/demo_enhanced_pipeline.py --symbol NVDA --shares 200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Enhanced Pipeline Demo")
    parser.add_argument("--symbols", default="NVDA,MSFT,QQQ",
                       help="Comma-separated symbols to analyze")
    parser.add_argument("--symbol", default=None,
                       help="Single symbol for detailed analysis")
    parser.add_argument("--shares", type=int, default=0,
                       help="Shares held (for options scanning)")
    parser.add_argument("--output", default=None,
                       help="Output JSON file")
    parser.add_argument("--no-sentiment", action="store_true",
                       help="Disable sentiment filter")
    parser.add_argument("--no-mtf", action="store_true",
                       help="Disable multi-timeframe")
    parser.add_argument("--no-options", action="store_true",
                       help="Disable options scanning")
    args = parser.parse_args()

    print("=" * 70)
    print("  🚀 HyprL Enhanced Pipeline Demo")
    print("=" * 70)
    print()

    # Import pipeline
    try:
        from hyprl.strategy.enhanced_pipeline import EnhancedPipeline, PipelineConfig
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the project root and venv is activated")
        return 1

    # Configure
    config = PipelineConfig(
        enable_sentiment=not args.no_sentiment,
        enable_mtf=not args.no_mtf,
        enable_options_scan=not args.no_options,
        min_shares_for_options=100,
    )

    pipeline = EnhancedPipeline(config)

    # Single symbol mode
    if args.symbol:
        print(f"📊 Analyzing {args.symbol}...")
        print()

        result = pipeline.analyze(args.symbol, shares_held=args.shares)
        print(result.summary())

        if args.output:
            Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))
            print(f"\n💾 Saved to {args.output}")

        return 0

    # Portfolio scan mode
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print(f"📊 Scanning {len(symbols)} symbols: {', '.join(symbols)}")
    print()

    results = pipeline.scan_portfolio(symbols)

    # Summary table
    print("=" * 80)
    print(f"{'Symbol':<8} {'Signal':<15} {'Confidence':>10} {'ML Prob':>10} {'Sentiment':>12} {'MTF':>8}")
    print("-" * 80)

    for r in results:
        emoji = {
            "strong_long": "🟢🟢",
            "long": "🟢",
            "weak_long": "🟡",
            "neutral": "⚪",
            "weak_short": "🟡",
            "short": "🔴",
            "strong_short": "🔴🔴",
            "blocked": "🚫",
        }

        sig_emoji = emoji.get(r.final_strength.value, "?")
        print(f"{r.symbol:<8} {sig_emoji} {r.final_strength.value:<12} {r.confidence:>10.0%} "
              f"{r.ml_probability:>10.1%} {r.sentiment_level:>12} {r.mtf_alignment:>+8.2f}")

    print("-" * 80)
    print()

    # Detailed results for each
    for r in results:
        if r.final_strength.value not in ("neutral", "blocked"):
            print(r.summary())

    # Options opportunities
    all_options = []
    for r in results:
        all_options.extend([(r.symbol, o) for o in r.options_opportunities])

    if all_options:
        print("\n" + "=" * 60)
        print("  💰 Options Income Opportunities")
        print("=" * 60)
        for symbol, opp in all_options:
            print(f"\n  {symbol} - {opp.strategy.upper()}")
            print(f"    Strike: ${opp.strike:.0f} ({opp.days_to_expiry} DTE)")
            print(f"    Premium: ${opp.premium:.2f}")
            print(f"    Annualized: {opp.annualized_return:.1f}%")
            print(f"    Prob Profit: {opp.probability_profit:.0%}")
            print(f"    Recommendation: {opp.recommendation}")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "results": [r.to_dict() for r in results],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\n💾 Results saved to {args.output}")

    # Trading recommendations
    longs = [r for r in results if r.final_strength.value in ("strong_long", "long")]
    shorts = [r for r in results if r.final_strength.value in ("strong_short", "short")]

    print("\n" + "=" * 60)
    print("  🎯 Trading Recommendations")
    print("=" * 60)

    if longs:
        print("\n  LONG:")
        for r in longs:
            print(f"    {r.symbol}: {r.recommended_size_pct:.1%} size, {r.confidence:.0%} confidence")

    if shorts:
        print("\n  SHORT:")
        for r in shorts:
            print(f"    {r.symbol}: {r.recommended_size_pct:.1%} size, {r.confidence:.0%} confidence")

    if not longs and not shorts:
        print("\n  No strong signals at this time. Consider:")
        print("    - Waiting for better setups")
        print("    - Selling premium if positions available")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
