#!/usr/bin/env python3
"""Multi-Timeframe Signal Generator.

Combines 1h and 15min models for better entry timing.
Runs every 15 minutes during market hours.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from hyprl.strategy.multi_timeframe import evaluate_mtf_signal, MTFConfig, SignalStrength


def is_market_hours() -> bool:
    """Check if within US market hours (9:30-16:00 ET)."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return True

    et = ZoneInfo("America/New_York")
    now_et = datetime.now(et)

    # Weekend check
    if now_et.weekday() > 4:
        return False

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_et <= market_close


def fetch_latest_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe_minutes: int,
    bars_needed: int = 100,
) -> pd.DataFrame:
    """Fetch latest bars from Alpaca."""
    if timeframe_minutes == 60:
        tf = TimeFrame(1, TimeFrameUnit.Hour)
    else:
        tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)

    end = datetime.now(timezone.utc)
    # Get extra bars to compute features
    start = end - timedelta(days=30)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df = df.sort_index()
    return df.tail(bars_needed)


def compute_features_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for 15min model (same as training)."""
    df = df.copy()

    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    for window in [4, 8, 16, 32]:
        df[f"volatility_{window}"] = df["returns"].rolling(window).std()

    for window in [4, 8, 16, 32, 64]:
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"close_vs_sma_{window}"] = (df["close"] - df[f"sma_{window}"]) / df[f"sma_{window}"]

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["volume_sma_16"] = df["volume"].rolling(16).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_16"].replace(0, np.nan)

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    for period in [4, 8, 16]:
        df[f"momentum_{period}"] = df["close"].pct_change(period)

    df["range_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["time_of_day"] = df["hour"] + df["minute"] / 60
    df["day_of_week"] = df.index.dayofweek

    return df


def compute_features_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for 1h model (simplified version)."""
    df = df.copy()

    df["returns"] = df["close"].pct_change()

    for window in [5, 10, 20]:
        df[f"volatility_{window}"] = df["returns"].rolling(window).std()
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"close_vs_sma_{window}"] = (df["close"] - df[f"sma_{window}"]) / df[f"sma_{window}"]

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift(1))
    low_close = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    return df


def predict_probability(model_or_bundle, df: pd.DataFrame, is_15m: bool = True) -> float:
    """Get probability from model.

    Handles both:
    - ProbabilityModel (1h models) with predict_proba method
    - Dict bundle (15m models) with model and feature_cols keys
    """
    if is_15m:
        df = compute_features_15m(df)
    else:
        df = compute_features_1h(df)

    # Check if it's a dict bundle (15m) or ProbabilityModel (1h)
    if isinstance(model_or_bundle, dict):
        # 15m model - dict bundle
        model = model_or_bundle["model"]
        feature_cols = model_or_bundle["feature_cols"]

        # Get last row
        last_row = df.iloc[-1:][feature_cols].dropna(axis=1)

        # Handle missing columns
        missing = set(feature_cols) - set(last_row.columns)
        for col in missing:
            last_row[col] = 0

        last_row = last_row[feature_cols]

        if last_row.isna().any().any():
            return 0.5

        proba = model.predict_proba(last_row)[0]
        return float(proba[1])

    else:
        # 1h model - ProbabilityModel object
        # It expects features to be computed already
        # Drop non-feature columns
        feature_df = df.drop(columns=["open", "high", "low", "close", "volume", "vwap", "trade_count"],
                             errors="ignore")
        feature_df = feature_df.dropna(axis=1, how="all").dropna()

        if feature_df.empty:
            return 0.5

        try:
            proba = model_or_bundle.predict_proba(feature_df.tail(1))
            return float(proba[0]) if len(proba) > 0 else 0.5
        except Exception as e:
            print(f"[MTF] Warning: 1h prediction error: {e}")
            return 0.5


def calculate_exits(
    entry_price: float,
    is_long: bool,
    atr: float,
    risk_mult: float = 2.0,
    reward_mult: float = 3.0,
) -> dict:
    """Calculate stop and take profit prices."""
    risk = atr * risk_mult

    if is_long:
        stop = entry_price - risk
        take_profit = entry_price + (risk * reward_mult)
    else:
        stop = entry_price + risk
        take_profit = entry_price - (risk * reward_mult)

    return {
        "stop_price": round(stop, 2),
        "take_profit_price": round(take_profit, 2),
        "atr": round(atr, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe signal generator")
    parser.add_argument("--symbol", required=True, help="Symbol to generate signals for")
    parser.add_argument("--model-1h", required=True, help="Path to 1h model")
    parser.add_argument("--model-15m", required=True, help="Path to 15min model")
    parser.add_argument("--output", required=True, help="Signal output JSONL")
    parser.add_argument("--force", action="store_true", help="Run even if market closed")
    parser.add_argument("--equity", type=float, default=100000, help="Account equity")
    args = parser.parse_args()

    # Check market hours
    if not args.force and not is_market_hours():
        print(f"[MTF] Market closed, skipping {args.symbol}")
        return

    # Load API keys
    api_key = os.environ.get("APCA_API_KEY_ID")
    secret_key = os.environ.get("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        print("Error: Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        sys.exit(1)

    # Load models
    print(f"[MTF] Loading models for {args.symbol}...")
    model_1h = joblib.load(args.model_1h)
    model_15m = joblib.load(args.model_15m)

    # Create Alpaca client
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    # Fetch data
    print(f"[MTF] Fetching data...")
    df_1h = fetch_latest_bars(client, args.symbol, 60, bars_needed=100)
    df_15m = fetch_latest_bars(client, args.symbol, 15, bars_needed=100)

    if len(df_1h) < 50 or len(df_15m) < 50:
        print(f"[MTF] Not enough data: 1h={len(df_1h)}, 15m={len(df_15m)}")
        return

    # Get predictions
    prob_1h = predict_probability(model_1h, df_1h, is_15m=False)
    prob_15m = predict_probability(model_15m, df_15m, is_15m=True)

    print(f"[MTF] {args.symbol} prob_1h={prob_1h:.3f}, prob_15m={prob_15m:.3f}")

    # Evaluate multi-timeframe signal
    config = MTFConfig(
        threshold_1h_long=0.60,
        threshold_1h_short=0.40,
        threshold_15m_long=0.55,
        threshold_15m_short=0.45,
        allow_15m_only=True,
        skip_conflicting=False,
    )

    mtf_signal = evaluate_mtf_signal(prob_1h, prob_15m, config)

    # Get current price and ATR
    current_price = float(df_15m["close"].iloc[-1])
    df_15m_feat = compute_features_15m(df_15m)
    current_atr = float(df_15m_feat["atr_14"].iloc[-1]) if "atr_14" in df_15m_feat else current_price * 0.02

    # Calculate exits
    is_long = mtf_signal.direction == "long"
    exits = calculate_exits(current_price, is_long, current_atr)

    # Calculate position size
    base_size = (args.equity * 0.02) / current_atr  # 2% risk per trade
    adjusted_size = base_size * mtf_signal.size_multiplier

    # Create signal record
    now = datetime.now(timezone.utc)
    signal = {
        "timestamp": now.isoformat(),
        "signal_id": f"{args.symbol}:{now.isoformat()}",
        "timeframe": "mtf",
        "bar_end_time": df_15m.index[-1].isoformat(),
        "data_source": "alpaca:iex:mtf",
        "symbol": args.symbol,
        "decision": mtf_signal.direction,
        "probability_1h": round(prob_1h, 4),
        "probability_15m": round(prob_15m, 4),
        "decision_1h": mtf_signal.decision_1h,
        "decision_15m": mtf_signal.decision_15m,
        "strength": mtf_signal.strength.value,
        "size_multiplier": mtf_signal.size_multiplier,
        "size": round(adjusted_size, 2),
        "entry_price": current_price,
        "stop_price": exits["stop_price"],
        "take_profit_price": exits["take_profit_price"],
        "atr": exits["atr"],
        "reason": mtf_signal.reason,
        "equity": args.equity,
    }

    # Write to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(signal) + "\n")

    # Log result
    strength_emoji = {
        SignalStrength.STRONG: "💪",
        SignalStrength.NORMAL: "→",
        SignalStrength.WEAK: "⚠️",
        SignalStrength.OPPORTUNISTIC: "🎯",
        SignalStrength.NONE: "—",
    }

    emoji = strength_emoji.get(mtf_signal.strength, "")
    print(f"[MTF] {args.symbol} {emoji} {mtf_signal.direction.upper()} "
          f"(1h:{mtf_signal.decision_1h}, 15m:{mtf_signal.decision_15m}) "
          f"size_mult={mtf_signal.size_multiplier}x reason={mtf_signal.reason}")


if __name__ == "__main__":
    main()
