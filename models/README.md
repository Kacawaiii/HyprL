# Trained Models (Private)

This directory contains trained XGBoost models for price direction prediction.

## Model Architecture

- **Algorithm**: XGBoost Classifier
- **Features**: 19 technical indicators (momentum, volatility, trend)
- **Target**: Next-hour price direction (up/down)
- **Calibration**: Platt scaling for probability estimates

## Available Models

| Symbol | Timeframe | Version | Sharpe | Win Rate |
|--------|-----------|---------|--------|----------|
| NVDA   | 1h        | v4      | 1.42   | 54%      |
| MSFT   | 1h        | v4      | 1.15   | 52%      |
| QQQ    | 1h        | v4      | 1.28   | 53%      |
| SPY    | 1h        | v4      | 1.08   | 51%      |

## Why Private?

Trained model weights contain the learned patterns that generate alpha. Releasing them would eliminate the trading edge.

**Want signals?** See [hyprlcore.com](https://hyprlcore.com) for subscriptions.

## Training Your Own

See `scripts/train_model.py` for the training pipeline structure. You'll need:
1. Historical OHLCV data
2. Feature engineering (see `src/hyprl/features/`)
3. Walk-forward validation setup
