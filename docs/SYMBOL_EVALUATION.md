# Symbol Evaluation Methodology

## Overview

This document describes the rigorous process used to evaluate and add new symbols to the HyprL trading system.

## Evaluation Pipeline

### Phase 1: Screening

**Criteria:**
| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Daily Volume | > 5M shares | Sufficient liquidity |
| Market Cap | > $50B | Stability, reliable data |
| Average Spread | < 0.1% | Low execution costs |
| Historical Data | > 2 years | Training data availability |

### Phase 2: Model Training

- XGBoost V4 ensemble (XGBoost + LightGBM + CatBoost)
- 28 features (momentum, volatility, microstructure)
- Walk-forward validation with 5-day embargo

**Minimum Requirements:**
| Metric | Threshold |
|--------|-----------|
| Validation Accuracy | > 50% |
| Brier Score | < 0.30 |
| AUC-ROC | > 0.52 |

### Phase 3: Backtest

Walk-forward out-of-sample testing with realistic costs.

**Go/No-Go Criteria:**
| Metric | Minimum | Ideal |
|--------|---------|-------|
| Sharpe Ratio | > 0.8 | > 1.2 |
| Profit Factor | > 1.3 | > 1.5 |
| Max Drawdown | < 25% | < 15% |
| Win Rate | > 50% | > 55% |

### Phase 4: Correlation Analysis

New symbols must not be too correlated with existing portfolio.

| Correlation | Action |
|-------------|--------|
| < 0.5 | Add (excellent diversification) |
| 0.5 - 0.7 | Add with reduced allocation |
| > 0.7 | Reject (too correlated) |

## Evaluation Results (January 2026)

| Symbol | Screening | Val Acc | Backtest | Correlation | Decision |
|--------|:---------:|:-------:|:--------:|:-----------:|:--------:|
| NVDA | PASS | 49.7% | PASS | - | IN PORTFOLIO |
| MSFT | PASS | 51.4% | PASS | 0.45 | IN PORTFOLIO |
| META | PASS | **57.2%** | PASS | 0.32 | **ADDED** |
| AAPL | PASS | 46.6% | - | - | REJECTED |
| GOOGL | PASS | 46.9% | - | - | REJECTED |
| AMD | PASS | 49.7% | - | - | REJECTED |

## Key Insight

**Validation accuracy < 50% means the model is worse than random.**

Symbols with val_acc below 50% are automatically rejected regardless of other metrics. This prevents adding symbols where the ML model has no predictive power.

## Current Portfolio Allocation

```
NVDA:  45% (semiconductor leader)
MSFT:  35% (software/cloud)
META:  20% (social/advertising - best model)
```

## Profit Sources

Analysis revealed the actual profit drivers:

| Component | Contribution |
|-----------|-------------|
| ATR Trailing Stops | ~60% |
| Smart Filter (RSI + Momentum) | ~25% |
| Guards V2 (VIX, earnings protection) | ~15% |
| ML Prediction | ~0% (acts as filter only) |

The ML model acts primarily as a signal filter, not a predictor. Profit comes from disciplined risk management via ATR-based trailing stops.
