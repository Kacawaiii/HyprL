#!/usr/bin/env python3
"""
Full V4 evaluation with crypto models and comparison.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from pathlib import Path
from datetime import datetime

# === CONFIG ===
INITIAL_CAPITAL = 10000
RISK_PCT = 0.01
ATR_MULT = 1.5
REWARD_MULT = 1.9

# === FEATURE COMPUTATION FOR EQUITIES (V4) ===
def compute_v4_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret_1h'] = df['close'].pct_change(1)
    df['ret_3h'] = df['close'].pct_change(3)
    df['ret_6h'] = df['close'].pct_change(6)
    df['ret_24h'] = df['close'].pct_change(24)

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

    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    vol_ma_10 = df['volume'].rolling(10).mean()
    vol_ma_30 = df['volume'].rolling(30).mean()
    df['vol_ratio_10_30'] = vol_ma_10 / vol_ma_30.replace(0, 1e-10)
    df['vol_regime_high'] = (df['vol_ratio_10_30'] > 1.2).astype(float)
    vol_std = df['volume'].rolling(24).std()
    vol_mean = df['volume'].rolling(24).mean()
    df['volume_zscore_24'] = (df['volume'] - vol_mean) / vol_std.replace(0, 1e-10)
    df['volume_surge'] = (df['volume'] > vol_mean * 2).astype(float)

    df['ret_skew_20'] = df['ret_1h'].rolling(20).skew()
    df['ret_kurt_20'] = df['ret_1h'].rolling(20).kurt()

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].rolling(24).sum()
    cum_vp = (typical_price * df['volume']).rolling(24).sum()
    vwap_24 = cum_vp / cum_vol.replace(0, 1e-10)
    df['vwap_dist_24'] = (df['close'] - vwap_24) / vwap_24.replace(0, 1e-10)

    cum_vol_8 = df['volume'].rolling(8).sum()
    cum_vp_8 = (typical_price * df['volume']).rolling(8).sum()
    vwap_8 = cum_vp_8 / cum_vol_8.replace(0, 1e-10)
    df['vwap_dist_8'] = (df['close'] - vwap_8) / vwap_8.replace(0, 1e-10)

    df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1

    price_dir = np.sign(df['close'] - df['open'])
    df['flow_imbalance_10'] = (price_dir * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum().replace(0, 1e-10)
    df['flow_imbalance_5'] = (price_dir * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum().replace(0, 1e-10)

    day_high = df['high'].rolling(24).max()
    day_low = df['low'].rolling(24).min()
    df['intraday_position'] = (df['close'] - day_low) / (day_high - day_low).replace(0, 1e-10)
    df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1e-10)

    avg_range = df['range_pct'].rolling(20).mean()
    df['range_expansion'] = df['range_pct'] / avg_range.replace(0, 1e-10)

    price_slope = df['close'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    rsi_slope = df['rsi_14'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    df['momentum_div'] = np.sign(price_slope) - np.sign(rsi_slope)

    return df

# === FEATURE COMPUTATION FOR CRYPTO ===
def compute_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df['ret_1h'] = df['close'].pct_change(1)
    df['ret_4h'] = df['close'].pct_change(4)
    df['ret_12h'] = df['close'].pct_change(12)
    df['ret_24h'] = df['close'].pct_change(24)
    df['ret_72h'] = df['close'].pct_change(72)

    # Volatility
    df['volatility_12h'] = df['ret_1h'].rolling(12).std()
    df['volatility_24h'] = df['ret_1h'].rolling(24).std()
    df['volatility_72h'] = df['ret_1h'].rolling(72).std()

    # RSI
    for period in [6, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # SMA ratios
    df['sma_ratio_12'] = df['close'] / df['close'].rolling(12).mean()
    df['sma_ratio_24'] = df['close'] / df['close'].rolling(24).mean()
    df['sma_ratio_72'] = df['close'] / df['close'].rolling(72).mean()

    # Volume
    df['volume_ratio_12'] = df['volume'] / df['volume'].rolling(12).mean().replace(0, 1e-10)
    df['volume_ratio_24'] = df['volume'] / df['volume'].rolling(24).mean().replace(0, 1e-10)
    vol_ma = df['volume'].rolling(24).mean()
    vol_std = df['volume'].rolling(24).std()
    df['volume_zscore'] = (df['volume'] - vol_ma) / vol_std.replace(0, 1e-10)

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_14_norm'] = df['atr_14'] / df['close']
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['true_range'] = tr / df['close']

    # Time features
    df['hour_of_day'] = df.index.hour if hasattr(df.index, 'hour') else 0
    df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Session indicators (UTC)
    hour = df.index.hour if hasattr(df.index, 'hour') else pd.Series([0] * len(df))
    df['is_asia_session'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['is_europe_session'] = ((hour >= 7) & (hour < 16)).astype(int)
    df['is_us_session'] = ((hour >= 13) & (hour < 22)).astype(int)

    return df

# === BACKTEST ENGINE ===
def run_backtest(ticker: str, model_path: str, feature_func, long_thresh: float, short_thresh: float) -> dict:
    # Download data
    data = yf.download(ticker, period="1y", interval="1h", progress=False)
    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
    else:
        data.columns = [c.lower() for c in data.columns]

    df = feature_func(data)
    df = df.dropna()

    if len(df) < 100:
        return None

    # Load model
    model_file = Path(model_path)
    if not model_file.exists():
        return None

    bundle = joblib.load(model_file)

    # Handle different model formats
    if 'ensemble' in bundle:
        model = bundle['ensemble']
        scaler = bundle['scaler']
        calibrator = bundle.get('calibrator')
        feature_cols = bundle['feature_columns']
    elif 'model' in bundle:
        model = bundle['model']
        scaler = bundle['scaler']
        calibrator = None
        feature_cols = bundle['feature_columns']
    else:
        return None

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return None

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    raw_probs = model.predict_proba(X_scaled)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        probs = raw_probs

    # Backtest
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    trades = []
    position = None

    for i in range(100, len(df)):
        row = df.iloc[i]
        prob = probs[i]
        price = row['close']
        atr = row['atr_14'] if 'atr_14' in row and not np.isnan(row['atr_14']) else price * 0.02

        if position is not None:
            exit_price = None
            if position['direction'] == 'long':
                if price <= position['stop']:
                    exit_price = position['stop']
                elif price >= position['target']:
                    exit_price = position['target']
            else:
                if price >= position['stop']:
                    exit_price = position['stop']
                elif price <= position['target']:
                    exit_price = position['target']

            if exit_price:
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - exit_price) * position['size']
                capital += pnl
                trades.append({'pnl': pnl})
                position = None

        if position is None:
            direction = None
            if prob > long_thresh:
                direction = 'long'
            elif prob < short_thresh:
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

    if len(trades) == 0:
        return {'ticker': ticker, 'trades': 0, 'return_pct': 0, 'sharpe': 0, 'max_dd': 0, 'profit_factor': 0, 'win_rate': 0}

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6.5) if len(returns) > 1 and np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) * 100

    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = sum(losses) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    win_rate = len(wins) / len(trades) * 100 if trades else 0
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

def main():
    print("=" * 80)
    print("  HYPRL V4 ENSEMBLE - FULL EVALUATION REPORT")
    print("=" * 80)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Capital: ${INITIAL_CAPITAL:,} | Risk: {RISK_PCT*100}% per trade")
    print("=" * 80)

    all_results = []

    # === EQUITIES V4 ===
    print("\n[1] EQUITIES - V4 Ensemble (XGB+LGB+CatBoost)")
    print("-" * 60)
    equities = [
        ("NVDA", "models/nvda_1h_ensemble_v4.joblib", 0.55, 0.45),
        ("MSFT", "models/msft_1h_ensemble_v4.joblib", 0.55, 0.45),
        ("QQQ", "models/qqq_1h_ensemble_v4.joblib", 0.55, 0.45),
    ]

    equity_results = []
    for ticker, model, lt, st in equities:
        print(f"  {ticker}...", end=" ", flush=True)
        r = run_backtest(ticker, model, compute_v4_features, lt, st)
        if r:
            equity_results.append(r)
            all_results.append(r)
            print(f"✓ {r['trades']} trades | {r['return_pct']:+.1f}% | Sharpe {r['sharpe']:.2f}")
        else:
            print("✗")

    # === CRYPTO ===
    print("\n[2] CRYPTO - XGBoost Models")
    print("-" * 60)
    cryptos = [
        ("BTC-USD", "models/crypto/btc_usd_xgb.joblib", 0.55, 0.45),
        ("ETH-USD", "models/crypto/eth_usd_xgb.joblib", 0.55, 0.45),
    ]

    crypto_results = []
    for ticker, model, lt, st in cryptos:
        print(f"  {ticker}...", end=" ", flush=True)
        r = run_backtest(ticker, model, compute_crypto_features, lt, st)
        if r:
            crypto_results.append(r)
            all_results.append(r)
            print(f"✓ {r['trades']} trades | {r['return_pct']:+.1f}% | Sharpe {r['sharpe']:.2f}")
        else:
            print("✗")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 80)
    print("  PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Asset':<12} {'Type':<10} {'Trades':>8} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'PF':>8} {'WR%':>8}")
    print("-" * 80)

    for r in equity_results:
        print(f"{r['ticker']:<12} {'Equity':<10} {r['trades']:>8} {r['return_pct']:>+10.1f} {r['sharpe']:>8.2f} {r['max_dd']:>8.1f} {r['profit_factor']:>8.2f} {r['win_rate']:>8.1f}")

    for r in crypto_results:
        print(f"{r['ticker']:<12} {'Crypto':<10} {r['trades']:>8} {r['return_pct']:>+10.1f} {r['sharpe']:>8.2f} {r['max_dd']:>8.1f} {r['profit_factor']:>8.2f} {r['win_rate']:>8.1f}")

    print("-" * 80)

    # Aggregates
    if all_results:
        eq_return = np.mean([r['return_pct'] for r in equity_results]) if equity_results else 0
        cr_return = np.mean([r['return_pct'] for r in crypto_results]) if crypto_results else 0
        total_return = np.mean([r['return_pct'] for r in all_results])
        total_sharpe = np.mean([r['sharpe'] for r in all_results])
        total_dd = np.mean([r['max_dd'] for r in all_results])
        total_pf = np.mean([r['profit_factor'] for r in all_results if r['profit_factor'] < 100])
        total_wr = np.mean([r['win_rate'] for r in all_results])
        total_trades = sum([r['trades'] for r in all_results])

        print(f"{'EQUITIES':<12} {'Avg':<10} {sum(r['trades'] for r in equity_results):>8} {eq_return:>+10.1f}")
        print(f"{'CRYPTO':<12} {'Avg':<10} {sum(r['trades'] for r in crypto_results):>8} {cr_return:>+10.1f}")
        print("-" * 80)
        print(f"{'TOTAL':<12} {'Avg':<10} {total_trades:>8} {total_return:>+10.1f} {total_sharpe:>8.2f} {total_dd:>8.1f} {total_pf:>8.2f} {total_wr:>8.1f}")

    print("=" * 80)

    return all_results

if __name__ == "__main__":
    main()
