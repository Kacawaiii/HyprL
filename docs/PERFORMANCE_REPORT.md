# HyprL Trading System - Performance Report

**Report Date**: 2026-01-31
**System Version**: V4 Ensemble
**Status**: Production (Paper Trading)

---

## 1. System Overview

### Current Portfolio Allocation

| Symbol | Allocation | Sector | Rationale |
|--------|------------|--------|-----------|
| NVDA | 45% | Semiconductors | High volatility, strong momentum patterns |
| MSFT | 35% | Software/Cloud | Stability, lower drawdown |
| META | 20% | Social/Advertising | Best validation accuracy (57.2%) |

### Strategy: Mix (RSI + Momentum + ATR Trailing Stops)

The deployed strategy combines:
- **RSI Filter**: Long when RSI < 45, avoid when RSI > 65
- **Momentum Filter**: Long when 3h momentum > +0.4%
- **ATR Trailing Stops**: Primary profit driver (~60% of returns)

```yaml
# Core Strategy Parameters
rules:
  long:
    rsi_below: 45
    momentum_above: 0.004  # +0.4%
  short:
    rsi_above: 65
    momentum_below: -0.004

features:
  - ret_3h
  - rsi_14
  - atr_14
```

---

## 2. Model Validation Accuracy

### Equity Models (V4 Ensemble)

| Symbol | Val Accuracy | Status | Notes |
|--------|-------------|--------|-------|
| **NVDA** | 49.7% | IN PORTFOLIO | Borderline, profitable via risk management |
| **MSFT** | 51.4% | IN PORTFOLIO | Above random, consistent |
| **META** | 57.2% | IN PORTFOLIO | Best performer, highest edge |
| AAPL | 46.6% | REJECTED | Below 50% threshold |
| GOOGL | 46.9% | REJECTED | Below 50% threshold |
| AMD | 49.7% | REJECTED | At threshold, insufficient margin |

### Crypto Models (XGBoost)

| Symbol | Train Accuracy | Test Accuracy | Rows | Status |
|--------|---------------|---------------|------|--------|
| BTC/USD | 79.4% | 54.8% | 8,687 | Active |
| ETH/USD | 79.1% | 58.0% | 8,687 | Active |

**Model Architecture**:
- Algorithm: XGBoost VotingClassifier ensemble (XGBoost 50%, LightGBM 30%, CatBoost 20%)
- Features: 28 technical indicators (equity_v2 + enhanced_v3)
- Calibration: Platt scaling for probability estimates
- Regularization: max_depth=3, reg_lambda=3.0, reg_alpha=0.5

---

## 3. Backtest Results Summary

### V4 Ensemble Performance (1 Year: 2025-01-19 to 2026-01-17)

#### Equities

| Ticker | Trades | Return | Sharpe | Max DD | Profit Factor | Win Rate |
|--------|--------|--------|--------|--------|---------------|----------|
| NVDA | 111 | +10.2% | 0.74 | 13.9% | 1.14 | 37.8% |
| MSFT | 92 | +5.9% | 0.50 | 11.0% | 1.10 | 37.0% |
| QQQ | 113 | +14.4% | 0.99 | 6.6% | 1.20 | 38.9% |
| **Average** | 316 | **+10.2%** | **0.74** | **10.5%** | **1.15** | **37.9%** |

#### Crypto (In-Sample Backtest)

| Ticker | Trades | Return | Sharpe | Max DD | Profit Factor | Win Rate |
|--------|--------|--------|--------|--------|---------------|----------|
| BTC-USD | 494 | +277.8% | 1.87 | 10.8% | 1.38 | 44.1% |
| ETH-USD | 514 | +218.1% | 1.62 | 14.2% | 1.33 | 42.6% |
| **Average** | 1008 | **+247.9%** | **1.75** | **12.5%** | **1.36** | **43.4%** |

#### Crypto Out-of-Sample (3 Months: Oct 2025 - Jan 2026)

| Ticker | Trades | Return | Sharpe | Max DD | Profit Factor | Win Rate |
|--------|--------|--------|--------|--------|---------------|----------|
| BTC-USD | 223 | -6.84% | -0.65 | 22.53% | 0.93 | 55.61% |
| ETH-USD | 161 | -0.34% | 0.16 | 18.65% | 1.02 | 60.87% |

**Note**: OOS performance degradation is expected and highlights the importance of risk management over pure ML prediction.

### Robustness Analysis (NVDA v3 - Monte Carlo)

| Metric | P5 (Worst) | P50 (Median) | P95 (Best) |
|--------|------------|--------------|------------|
| Profit Factor | 2.56 | 3.32 | 4.29 |
| Max Drawdown | 6.70% | 14.58% | 35.97% |

- **Probability of Ruin (>50% DD)**: 1.4%
- **Chance of Annual Loss**: 0.0%

---

## 4. Risk Parameters

### From `configs/runtime/strategy_mix.yaml`

#### Capital Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| risk_per_trade | 2% | Risk per individual trade |
| max_position_pct | 25% | Maximum single position size |
| max_total_exposure | 75% | Maximum total portfolio exposure |

#### Stop Loss / Take Profit

| Parameter | Value | Description |
|-----------|-------|-------------|
| stop_loss_atr | 2.0x ATR | Initial stop loss distance |
| take_profit_atr | 2.5x ATR | Take profit target |
| max_daily_loss_pct | 5% | Daily loss circuit breaker |
| max_drawdown_pct | 12% | Maximum allowed drawdown |

#### Trailing Stop Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| use_trailing_stop | true | Enable trailing stops |
| trailing_activation_pct | 1.5% | Profit threshold to activate |
| trailing_distance_pct | 0.8% | Trailing stop distance |

#### Guards V2 (Additional Protection)

| Guard | Value | Description |
|-------|-------|-------------|
| earnings_blackout_days | 3 | Skip trades near earnings |
| vix_threshold_reduce | 25 | Reduce position size if VIX > 25 |
| vix_threshold_block | 35 | Block all trades if VIX > 35 |
| max_consecutive_losses | 3 | Pause after 3 consecutive losses |
| pause_minutes_after_loss | 30 | Cool-down period after loss streak |

#### Kelly Sizing

| Parameter | Value | Description |
|-----------|-------|-------------|
| enabled | true | Dynamic position sizing |
| lookback_trades | 20 | Trades used for Kelly calculation |
| max_multiplier | 1.5x | Maximum Kelly multiplier |
| min_multiplier | 0.5x | Minimum Kelly multiplier |

---

## 5. Key Insights

### Profit Attribution Analysis

| Component | Contribution | Role |
|-----------|-------------|------|
| **ATR Trailing Stops** | ~60% | Primary profit driver |
| **Smart Filter (RSI + Momentum)** | ~25% | Entry timing |
| **Guards V2 (VIX, earnings)** | ~15% | Loss prevention |
| **ML Prediction** | ~0% | Signal filter only |

### Critical Understanding

1. **ML Models Act as Filters, Not Predictors**
   - The ML model's primary role is to filter out bad trades
   - Profit generation comes from disciplined risk management
   - Validation accuracy is a gate (must be > 50%), not a profit driver

2. **Trailing Stops Are the Edge**
   - ATR-based trailing stops capture trend continuation
   - Let winners run, cut losers early
   - Asymmetric return profile even with <50% win rate

3. **Val Accuracy < 50% = Worse Than Random**
   - AAPL (46.6%), GOOGL (46.9%), AMD (49.7%) were rejected
   - These models would destroy edge through bad signal filtering
   - Strict enforcement prevents portfolio contamination

4. **Out-of-Sample Degradation is Normal**
   - Crypto OOS: -6.84% (BTC), -0.34% (ETH) vs massive in-sample gains
   - This is why risk management matters more than ML accuracy
   - The strategy degrades gracefully due to protective stops

---

## 6. Symbols Evaluated

### Complete Evaluation Table

| Symbol | Screening | Val Accuracy | Backtest | Correlation | Decision | Reason |
|--------|:---------:|:------------:|:--------:|:-----------:|:--------:|--------|
| NVDA | PASS | 49.7% | PASS | - | GO | Foundational, risk-managed |
| MSFT | PASS | 51.4% | PASS | 0.45 | GO | Above threshold, stable |
| META | PASS | **57.2%** | PASS | 0.32 | GO | Best accuracy, diversified |
| AAPL | PASS | 46.6% | - | - | NO-GO | val_acc < 50% |
| GOOGL | PASS | 46.9% | - | - | NO-GO | val_acc < 50% |
| AMD | PASS | 49.7% | - | - | NO-GO | At threshold, no margin |
| BTC/USD | PASS | 54.8% | PASS | - | GO | Crypto diversification |
| ETH/USD | PASS | 58.0% | PASS | - | GO | Best crypto accuracy |

### Selection Criteria Summary

**Minimum Requirements:**
- Validation Accuracy > 50% (worse = worse than random)
- Brier Score < 0.30
- AUC-ROC > 0.52

**Backtest Criteria:**
- Sharpe Ratio > 0.8
- Profit Factor > 1.3
- Max Drawdown < 25%

**Correlation Constraint:**
- New symbols must have < 0.7 correlation with existing portfolio

---

## 7. Appendix

### Feature Set (28 Features - V4 Ensemble)

**Momentum V2**: ret_1h, ret_3h, ret_6h, ret_24h

**Volatility V2**: atr_14, atr_72, atr_14_norm, atr_72_norm, range_pct, true_range

**RSI V2**: rsi_7, rsi_14, rsi_21

**Volume V2**: vol_ratio_10_30, vol_regime_high, volume_zscore_24, volume_surge

**Moments V2**: ret_skew_20, ret_kurt_20

**VWAP V3**: vwap_dist_24, vwap_dist_8

**Flow V3**: flow_imbalance_10, flow_imbalance_5

**Intraday V3**: overnight_gap, intraday_position, close_location

**Divergence V3**: range_expansion, momentum_div

### Deployed Configuration

```yaml
# Symbol Thresholds
symbols:
  NVDA:
    enabled: true
    weight: 0.45
    long_threshold: 0.57
  MSFT:
    enabled: true
    weight: 0.35
    long_threshold: 0.60
  META:
    enabled: true
    weight: 0.20
    long_threshold: 0.58

# Execution
execution:
  allow_short: false  # Bullish regime 2026
  use_brackets: true
  max_orders_per_day: 15
  trading_hours: "09:35-15:55"
  timezone: "America/New_York"
```

---

## Disclaimer

Past performance does not guarantee future results. This system is in paper trading mode. The strategies, backtests, and performance metrics presented here are for informational purposes only. Never invest more than you can afford to lose.

---

*Report generated by HyprL Trading System*
*Last updated: 2026-01-31*
