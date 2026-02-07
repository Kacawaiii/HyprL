# Trained Models (Private)

This directory contains trained XGBoost v4 ensemble models for price direction prediction.

## Model Architecture

- **Algorithm**: XGBoost VotingClassifier ensemble
- **Features**: 28 technical indicators (equity_v2 + enhanced_v3)
- **Target**: Next-hour price direction (up/down)
- **Calibration**: Platt scaling for probability estimates
- **Preprocessing**: StandardScaler normalization

## Available Models

| File | Symbol | Type | Features |
|------|--------|------|----------|
| `nvda_1h_prob_v4.joblib` | NVDA | ProbabilityModel | 28 |
| `msft_1h_prob_v4.joblib` | MSFT | ProbabilityModel | 28 |
| `drift_baseline.npz` | ALL | Drift baseline | 12 RT features |

## Feature Set (28)

**equity_v2** (19): ret_1h, ret_3h, ret_6h, ret_24h, atr_14, atr_72, atr_14_norm, rsi_14, macd, macd_hist, macd_signal, bb_width, bb_pct, volume_ratio, volume_zscore, stoch_k, stoch_d, adx_14, obv_slope

**enhanced_v3** (9): vwap_dist, flow_imbalance, momentum_div, wick_ratio, gap_pct, session_return, intraday_range, volume_price_trend, relative_volume

## Backtest Performance (v4)

| Symbol | Sharpe | Win Rate | Profit Factor | Return |
|--------|--------|----------|---------------|--------|
| NVDA | 1.55 | 55.3% | 1.51 | +41.1% |
| MSFT | 1.03 | 52.9% | 1.39 | +35.9% |

## Why Private?

Trained model weights contain learned patterns that generate alpha. Releasing them would eliminate the trading edge.

**Want signals?** See [hyprlcore.com](https://hyprlcore.com) for subscriptions.

## Training Your Own

```bash
# Train v4 ensemble models
python scripts/train_model_v4_ensemble.py --symbol NVDA --timeframe 1h

# Generate drift baseline
python scripts/generate_drift_baseline.py --symbols NVDA,MSFT --period 1y
```
