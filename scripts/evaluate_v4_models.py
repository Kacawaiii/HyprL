#!/usr/bin/env python3
"""
Evaluate V4 ensemble models with proper backtest metrics.
Outputs: Sharpe, Max Drawdown, Profit Factor, Win Rate, Total Return
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# === CONFIG ===
TICKERS = ["NVDA", "MSFT", "QQQ"]
CRYPTO = ["BTC-USD", "ETH-USD"]
PERIOD = "1y"
INTERVAL = "1h"
INITIAL_CAPITAL = 10000
RISK_PCT = 0.01
ATR_MULT = 1.5
REWARD_MULT = 1.9
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45

# === FEATURE COMPUTATION ===
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all V4 features (V2 + V3 enhanced)."""
    df = df.copy()

    # Basic returns
    df['ret_1h'] = df['close'].pct_change(1)
    df['ret_3h'] = df['close'].pct_change(3)
    df['ret_6h'] = df['close'].pct_change(6)
    df['ret_24h'] = df['close'].pct_change(24)

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['true_range'] = tr
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_72'] = tr.rolling(72).mean()
    df['atr_14_norm'] = df['atr_14'] / df['close']
    df['atr_72_norm'] = df['atr_72'] / df['close']
    df['range_pct'] = (df['high'] - df['low']) / df['close']

    # RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # Volume features
    vol_ma_10 = df['volume'].rolling(10).mean()
    vol_ma_30 = df['volume'].rolling(30).mean()
    df['vol_ratio_10_30'] = vol_ma_10 / vol_ma_30.replace(0, 1e-10)
    df['vol_regime_high'] = (df['vol_ratio_10_30'] > 1.2).astype(float)
    vol_std = df['volume'].rolling(24).std()
    vol_mean = df['volume'].rolling(24).mean()
    df['volume_zscore_24'] = (df['volume'] - vol_mean) / vol_std.replace(0, 1e-10)
    df['volume_surge'] = (df['volume'] > vol_mean * 2).astype(float)

    # Higher moments
    df['ret_skew_20'] = df['ret_1h'].rolling(20).skew()
    df['ret_kurt_20'] = df['ret_1h'].rolling(20).kurt()

    # V3 Enhanced features
    # VWAP distance
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].rolling(24).sum()
    cum_vp = (typical_price * df['volume']).rolling(24).sum()
    vwap_24 = cum_vp / cum_vol.replace(0, 1e-10)
    df['vwap_dist_24'] = (df['close'] - vwap_24) / vwap_24.replace(0, 1e-10)

    cum_vol_8 = df['volume'].rolling(8).sum()
    cum_vp_8 = (typical_price * df['volume']).rolling(8).sum()
    vwap_8 = cum_vp_8 / cum_vol_8.replace(0, 1e-10)
    df['vwap_dist_8'] = (df['close'] - vwap_8) / vwap_8.replace(0, 1e-10)

    # Overnight gap (approximation)
    df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1

    # Flow imbalance (proxy using price movement and volume)
    price_dir = np.sign(df['close'] - df['open'])
    df['flow_imbalance_10'] = (price_dir * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum().replace(0, 1e-10)
    df['flow_imbalance_5'] = (price_dir * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum().replace(0, 1e-10)

    # Intraday position
    day_high = df['high'].rolling(24).max()
    day_low = df['low'].rolling(24).min()
    df['intraday_position'] = (df['close'] - day_low) / (day_high - day_low).replace(0, 1e-10)

    # Close location
    df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-10)

    # Range expansion
    avg_range = df['range_pct'].rolling(20).mean()
    df['range_expansion'] = df['range_pct'] / avg_range.replace(0, 1e-10)

    # Momentum divergence
    price_slope = df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    rsi_slope = df['rsi_14'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    df['momentum_div'] = np.sign(price_slope) - np.sign(rsi_slope)

    return df

# === BACKTEST ENGINE ===
def run_backtest(ticker: str, model_path: str, is_crypto: bool = False) -> dict:
    """Run backtest and return metrics."""

    # Load data
    data = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    if data.empty:
        return None

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
    else:
        data.columns = [c.lower() for c in data.columns]

    # Compute features
    df = compute_features(data)
    df = df.dropna()

    if len(df) < 100:
        return None

    # Load model
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"  [WARN] Model not found: {model_path}")
        return None

    bundle = joblib.load(model_file)
    model = bundle['ensemble']
    scaler = bundle['scaler']
    calibrator = bundle.get('calibrator')
    feature_cols = bundle['feature_columns']

    # Check features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing[:5]}...")
        return None

    # Prepare data
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    # Get predictions
    raw_probs = model.predict_proba(X_scaled)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        probs = raw_probs

    # Backtest simulation
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    position = None

    for i in range(100, len(df)):
        row = df.iloc[i]
        prob = probs[i]
        price = row['close']
        atr = row['atr_14'] if not np.isnan(row['atr_14']) else price * 0.02

        # Exit logic
        if position is not None:
            exit_price = None
            exit_reason = None

            if position['direction'] == 'long':
                if price <= position['stop']:
                    exit_price = position['stop']
                    exit_reason = 'stop'
                elif price >= position['target']:
                    exit_price = position['target']
                    exit_reason = 'target'
            else:  # short
                if price >= position['stop']:
                    exit_price = position['stop']
                    exit_reason = 'stop'
                elif price <= position['target']:
                    exit_price = position['target']
                    exit_reason = 'target'

            if exit_price:
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - exit_price) * position['size']

                capital += pnl
                trades.append({
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pnl': pnl,
                    'direction': position['direction'],
                    'reason': exit_reason
                })
                position = None

        # Entry logic
        if position is None:
            direction = None
            if prob > LONG_THRESHOLD:
                direction = 'long'
            elif prob < SHORT_THRESHOLD:
                direction = 'short'

            if direction:
                risk_amount = capital * RISK_PCT
                stop_dist = atr * ATR_MULT
                position_size = risk_amount / stop_dist if stop_dist > 0 else 0

                if position_size > 0:
                    if direction == 'long':
                        stop = price - stop_dist
                        target = price + stop_dist * REWARD_MULT
                    else:
                        stop = price + stop_dist
                        target = price - stop_dist * REWARD_MULT

                    position = {
                        'direction': direction,
                        'entry': price,
                        'stop': stop,
                        'target': target,
                        'size': position_size
                    }

        equity_curve.append(capital)

    # Calculate metrics
    if len(trades) == 0:
        return {
            'ticker': ticker,
            'trades': 0,
            'return_pct': 0,
            'sharpe': 0,
            'max_dd': 0,
            'profit_factor': 0,
            'win_rate': 0
        }

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]

    # Sharpe (annualized for hourly)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6.5)  # ~6.5 trading hours
    else:
        sharpe = 0

    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) * 100

    # Profit Factor
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = sum(losses) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win Rate
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    # Total Return
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        'ticker': ticker,
        'trades': len(trades),
        'return_pct': round(total_return, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 2),
        'profit_factor': round(profit_factor, 2),
        'win_rate': round(win_rate, 1)
    }

# === MAIN ===
def main():
    print("=" * 70)
    print("  HYPRL V4 ENSEMBLE MODEL EVALUATION")
    print("=" * 70)
    print(f"  Period: {PERIOD} | Interval: {INTERVAL}")
    print(f"  Capital: ${INITIAL_CAPITAL:,} | Risk: {RISK_PCT*100}%")
    print(f"  Thresholds: Long>{LONG_THRESHOLD}, Short<{SHORT_THRESHOLD}")
    print("=" * 70)

    results = []

    # Equities
    print("\n[EQUITIES]")
    for ticker in TICKERS:
        model_path = f"models/{ticker.lower()}_1h_ensemble_v4.joblib"
        print(f"  Testing {ticker}...", end=" ")
        result = run_backtest(ticker, model_path)
        if result:
            results.append(result)
            print(f"✓ {result['trades']} trades, {result['return_pct']:+.1f}%")
        else:
            print("✗ Failed")

    # Crypto (use NVDA model as proxy for now)
    print("\n[CRYPTO]")
    for ticker in CRYPTO:
        model_path = "models/nvda_1h_ensemble_v4.joblib"  # Use as proxy
        print(f"  Testing {ticker}...", end=" ")
        result = run_backtest(ticker, model_path, is_crypto=True)
        if result:
            results.append(result)
            print(f"✓ {result['trades']} trades, {result['return_pct']:+.1f}%")
        else:
            print("✗ Failed")

    # Summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Ticker':<10} {'Trades':>8} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'PF':>8} {'WinRate%':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['ticker']:<10} {r['trades']:>8} {r['return_pct']:>+10.1f} {r['sharpe']:>8.2f} {r['max_dd']:>8.1f} {r['profit_factor']:>8.2f} {r['win_rate']:>10.1f}")

    print("-" * 70)

    # Aggregate stats
    if results:
        avg_return = np.mean([r['return_pct'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_dd = np.mean([r['max_dd'] for r in results])
        avg_pf = np.mean([r['profit_factor'] for r in results if r['profit_factor'] < 100])
        avg_wr = np.mean([r['win_rate'] for r in results])
        total_trades = sum([r['trades'] for r in results])

        print(f"{'AVERAGE':<10} {total_trades:>8} {avg_return:>+10.1f} {avg_sharpe:>8.2f} {avg_dd:>8.1f} {avg_pf:>8.2f} {avg_wr:>10.1f}")

    print("=" * 70)

    return results

if __name__ == "__main__":
    results = main()
