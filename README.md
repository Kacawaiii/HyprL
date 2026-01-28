# HyprL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: Paper Trading](https://img.shields.io/badge/status-paper%20trading-blue)]()

End-to-end algorithmic trading system for US equities. Covers the full pipeline from data ingestion and feature engineering to ML prediction, risk management, execution, and real-time monitoring.

Built as a solo project to explore quantitative finance, machine learning in production, and systems engineering.

---

## Overview

HyprL is a complete trading infrastructure that handles:

- **Feature engineering** — 28 technical indicators across multiple timeframes
- **ML prediction** — XGBoost v4 ensemble with Platt calibration
- **Risk management** — Kelly sizing, ATR-based stops, circuit breakers, regime detection
- **Execution** — Bracket orders via Alpaca API (paper + live)
- **Monitoring** — Drift detection, kill switch, Discord alerting
- **Backtesting** — Realistic simulation with slippage, commissions, walk-forward validation

```
Data (Alpaca)                                                  Monitoring
     │                                                              ▲
     ▼                                                              │
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐   │
│  OHLCV   │───▶│  28 Feature│───▶│  XGBoost   │───▶│  Filters │───┤
│  Stream   │    │  Pipeline  │    │  Ensemble  │    │  & Gates │   │
└──────────┘    └────────────┘    └────────────┘    └────┬─────┘   │
                                                         │         │
                                                         ▼         │
                                                    ┌──────────┐   │
                                                    │ Position │───┘
                                                    │ Sizing + │
                                                    │ Execution│
                                                    └──────────┘
```

---

## Backtest Results

Full stack backtest — 1 year, 1H bars, 5 bps slippage, long-only, all filters active.

| Metric | NVDA | MSFT | Portfolio |
|--------|------|------|-----------|
| Return | +41.1% | +35.9% | +17.9% |
| Sharpe | 1.55 | 1.03 | 1.29 |
| Win Rate | 55.3% | 52.9% | 54.1% |
| Profit Factor | 1.51 | 1.39 | 1.45 |
| Trades | 47 | 51 | 98 |
| Max Drawdown | -9% | -15% | -22.6% |

Live track record dashboard: [app.hyprlcore.com](https://app.hyprlcore.com)

---

## Technical Architecture

### Feature Engineering

Two-layer feature pipeline producing 28 indicators per bar:

**equity_v2** (19 features): multi-horizon returns (1h, 3h, 6h, 24h), ATR (14, 72, normalized), RSI, MACD + histogram + signal, Bollinger width + %B, volume ratio + z-score, Stochastic K/D, ADX, OBV slope.

**enhanced_v3** (9 features): VWAP distance, order flow imbalance, momentum divergence, wick ratio, gap percentage, session return, intraday range, volume-price trend, relative volume.

Features are computed incrementally on each new bar and stored in a rolling window of 600 bars.

### ML Models

- **Algorithm**: `VotingClassifier` (XGBoost ensemble, 3 estimators)
- **Preprocessing**: `StandardScaler` normalization
- **Calibration**: Platt scaling via `CalibratedClassifierCV` for reliable probability estimates
- **Output**: P(up) for the next bar, used as signal strength
- **Training**: Walk-forward validation with expanding window splits (`src/hyprl/search/wfo.py`)

Models are serialized as `ProbabilityModel` objects wrapping scaler + classifier + calibrator.

### Risk Management

**Position sizing**: Half-Kelly criterion constrained by ATR-based risk budgets. Each trade risks a fixed percentage of equity, with position size derived from the ATR stop distance.

**Stop/take-profit**: Bracket orders (OCO) with ATR-multiplied levels. Trailing stops activate after reaching a configurable profit threshold.

**Portfolio-level controls**:

| Control | Implementation |
|---------|---------------|
| Circuit breakers | 4 levels (-2% / -5% / -10% / -15%), progressive response |
| Kill switch | File-based emergency halt at 30% drawdown |
| Regime detection | ADX + volatility percentile → block trades in extreme vol (>95th pctile) |
| Drift detection | PSI + Kolmogorov-Smirnov test on feature distributions vs baseline |
| Correlation tracking | Limits notional exposure to correlated positions |
| Earnings blackout | No trading 3 days before/after earnings |
| VIX guard | Reduce size at VIX > 25, block at VIX > 35 |

### Signal Filtering

Signals pass through multiple gates before execution:

1. **Smart Filter** — blocks falling knives (momentum < -6%), panic entries, FOMO trades
2. **Quality Filter** — minimum signal quality score threshold
3. **Sentiment Filter** — news sentiment via Alpaca News API, blocks on very bearish
4. **Regime Gate** — reduces/blocks in volatile markets
5. **Drift Gate** — reduces/blocks when feature distributions shift

### Real-Time Engine

Async engine (`src/hyprl/rt/engine.py`) processes market events in real-time:

- `_BarBuilder` aggregates tick data into OHLCV bars (1m/5m/15m/1h)
- Features computed on each completed bar
- Online model fitting with rolling feature history
- Bracket order simulation with TP/SL tracking
- Equity curve and drawdown monitoring per bar
- Session resume capability (picks up from last processed bar)

### Execution

Three independent strategy modes run in parallel, each with its own Alpaca account:

| | Normal | Aggressive | Mix |
|--|--------|-----------|-----|
| Risk/trade | 2% | 3% | 2% |
| Max exposure | 60% | 100% | 75% |
| Stop loss | 1.5 ATR | 1.5 ATR | 2.0 ATR |
| Take profit | 3.0 ATR | 2.5 ATR | 2.5 ATR |
| Max drawdown | -10% | -15% | -12% |
| EOD close | No | Yes | No |

Managed via systemd services with per-strategy configs (`configs/runtime/strategy_*.yaml`).

### Monitoring & Alerting

- **Discord**: Webhooks per strategy per channel (trades, alerts, summaries)
- **Drift detector**: Compares live feature distributions against baseline (PSI + KS test), saved as `models/drift_baseline.npz`
- **Paper vs backtest validation**: Script that compares live metrics against backtest expectations with tolerance bands
- **Health**: Startup/crash/stop notifications, heartbeat

---

## Project Structure

```
src/hyprl/
├── backtest/          Backtesting engine with realistic costs
├── broker/            Alpaca integration (paper + live), abstract base
├── calendar/          Earnings, FOMC, OpEx blackout calendar
├── crypto/            BTC/ETH extension (24/7, dedicated models)
├── features/          Feature engineering (equity_v2, enhanced_v3)
├── indicators/        Technical indicators (RSI, MACD, Stochastic, ADX, VWAP)
├── meta/              Model registry, versioning, rollback
├── model/             ProbabilityModel (XGBoost + scaler + calibrator)
├── monitoring/        Drift detection, regime detector, Discord notifier
├── native/            Rust FFI bindings (PyO3)
├── options/           Options overlay (covered calls, puts, collars)
├── risk/              Circuit breakers, kill switch, correlation tracker
├── rt/                Real-time async engine, bar builder, position sizing
├── search/            Walk-forward optimization, grid search
├── sentiment/         News sentiment analysis + trade filter
└── strategy/          Core strategy logic, smart filter, quality filter

scripts/
├── execution/         Strategy bridge, live runner
├── ops/               Deployment, health checks
└── analysis/          Backtest scripts, model evaluation

configs/runtime/       Per-strategy YAML configs + environment files
models/                Trained XGBoost v4 ensembles + drift baseline
deploy/                Docker, systemd services, Caddy reverse proxy
apps/
├── landing/           Marketing site (hyprlcore.com)
└── track_record/      Streamlit performance dashboard
native/                Rust accelerator module (hyprl_supercalc)
tests/                 Unit & integration tests
```

---

## Rust Accelerator

Optional native module for compute-intensive operations. Built with PyO3 and maturin.

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| ATR 14 (10k bars) | 45ms | 1.2ms | 37x |
| Single backtest | 450ms | 12ms | 37x |
| Grid search (1000 configs) | 7.5min | 8s | 56x |

Covers: SMA, EMA, RSI, MACD, ATR, Bollinger Bands, vectorized backtesting, parallel grid search (Rayon).

```bash
cd native/hyprl_supercalc && maturin develop --release
```

---

## Tech Stack

| | |
|--|--|
| **Language** | Python 3.10+, Rust (optional accelerator) |
| **ML** | XGBoost, scikit-learn, joblib |
| **Data** | pandas, numpy, polars, yfinance |
| **Broker** | alpaca-trade-api |
| **Async** | asyncio, aiohttp |
| **Monitoring** | discord.py (webhooks) |
| **Dashboard** | Streamlit |
| **Deployment** | Docker, systemd, Caddy |
| **Testing** | pytest, mypy, ruff |
| **Rust FFI** | PyO3, maturin |

---

## Getting Started

```bash
git clone https://github.com/Kacawaiii/HyprL.git
cd HyprL
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure Alpaca paper trading keys
cp configs/runtime/.env.example configs/runtime/.env.normal

# Run backtest
python scripts/backtest_full_stack.py

# Start paper trading
python scripts/execution/run_strategy_bridge.py --strategy normal --paper
```

---

## Disclaimer

This project is for educational and research purposes. Trading involves risk. Past backtest performance does not guarantee future results.

---

## License

MIT
