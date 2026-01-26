# HyprL - Algorithmic Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Trading](https://img.shields.io/badge/status-paper%20trading-orange.svg)]()

**HyprL** is a complete algorithmic trading system for US equities using Machine Learning (XGBoost) and advanced technical analysis.

> **Note**: Infrastructure and technical implementation are open source. Trading strategy and trained models are private to preserve the edge. See [hyprlcore.com](https://hyprlcore.com) for live track record and signals.

## Performance Summary

| Metric | Backtest (2024) | Paper Trading (Jan 2026) |
|--------|-----------------|--------------------------|
| **Sharpe Ratio** | 1.42 | - |
| **Win Rate** | 54% | 53% |
| **Max Drawdown** | -8.2% | -1.2% |
| **Annual Return** | +28% | +0.2% (MTD) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         HyprL System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Data    │───▶│ Features │───▶│  Model   │───▶│ Signals  │  │
│  │ (Alpaca) │    │ Engine   │    │ (XGBoost)│    │ Generator│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │        │
│       ▼                                               ▼        │
│  ┌──────────┐                                   ┌──────────┐   │
│  │   OHLCV  │                                   │ Filters  │   │
│  │   Bars   │                                   │ & Gates  │   │
│  └──────────┘                                   └──────────┘   │
│                                                      │         │
│                                                      ▼         │
│                    ┌──────────────────────────────────────┐    │
│                    │         Trading Bridge               │    │
│                    │  (Normal / Aggressive / Mix)         │    │
│                    └──────────────────────────────────────┘    │
│                                   │                            │
│                    ┌──────────────┼──────────────┐            │
│                    ▼              ▼              ▼            │
│              ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│              │  Alpaca  │  │ Discord  │  │  Track   │        │
│              │  Paper   │  │ Notifier │  │  Record  │        │
│              └──────────┘  └──────────┘  └──────────┘        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
HyprL/
├── src/hyprl/                    # Core source code
│   ├── backtest/                 # Backtesting engine
│   ├── broker/                   # Alpaca integration
│   │   ├── alpaca.py             # Alpaca API wrapper
│   │   ├── base.py               # Abstract broker interface
│   │   └── dryrun.py             # Dry-run mode
│   ├── features/                 # Feature engineering
│   ├── indicators/               # Technical indicators
│   │   └── technical.py          # RSI, MACD, Stochastic, ADX, VWAP
│   ├── live/                     # Live trading components
│   │   └── risk.py               # Position sizing, Kelly criterion
│   ├── model/                    # ML models
│   │   └── probability.py        # XGBoost probability predictor
│   ├── monitoring/               # Monitoring & notifications
│   │   └── discord_notifier.py   # Discord webhooks
│   ├── native/                   # Rust bindings (optional)
│   │   └── supercalc.py          # Fast calculations
│   ├── risk/                     # Risk management
│   │   └── circuit_breakers.py   # Multi-level protection
│   └── strategy/                 # Trading strategies
│       └── core.py               # Main strategy logic
│
├── scripts/                      # Execution scripts
│   ├── run_multi_account_v2.py   # Main trading script
│   ├── run_backtest.py           # Run backtests
│   ├── backtest_with_costs.py    # Backtest with realistic costs
│   ├── test_connection.py        # Test Alpaca connection
│   └── train_model_nvda_1h_v3.py # Train ML model
│
├── configs/                      # Configuration files
│   ├── runtime/                  # Runtime configs
│   │   ├── .env.example          # Environment template
│   │   ├── .env.normal           # Normal account config
│   │   ├── .env.aggressive       # Aggressive account config
│   │   └── .env.mix              # Mix account config
│   ├── NVDA-1h_v3.yaml           # NVDA strategy config
│   ├── MSFT-1h_v3.yaml           # MSFT strategy config
│   └── QQQ-1h_v3.yaml            # QQQ strategy config
│
├── models/                       # Trained ML models (private)
│   └── README.md                 # Model architecture docs
│
├── deploy/                       # Deployment
│   └── systemd/                  # Systemd services
│       ├── hyprl-multi-account.service
│       └── hyprl-multi-account.timer
│
├── apps/                         # Applications
│   ├── landing/                  # Website
│   └── track_record/             # Streamlit dashboard
│
├── tests/                        # Unit tests
├── live/logs/                    # Trading logs
└── native/hyprl_supercalc/       # Rust calculator (optional)
    ├── Cargo.toml                # Rust dependencies
    ├── src/
    │   ├── lib.rs                # Main library
    │   ├── indicators/           # Fast indicators (RSI, MACD, ATR...)
    │   ├── backtest/             # Vectorized backtesting
    │   └── ffi.rs                # PyO3 Python bindings
    └── target/                   # Build output (gitignored)
```

---

## Strategy Overview

### Signal Generation

HyprL uses a combination of ML predictions and technical indicators:

| Component | Description |
|-----------|-------------|
| **XGBoost Model** | Predicts up/down probability for next hour |
| **RSI (14)** | Overbought/oversold filter |
| **MACD** | Trend confirmation |
| **Stochastic RSI** | Momentum oscillator |
| **ADX (14)** | Trend strength filter |
| **VWAP** | Volume-weighted price reference |

### Signal Rules

| Condition | Action |
|-----------|--------|
| Model prob > 55% + RSI < 70 + ADX > 25 | **LONG** |
| Model prob < 45% + RSI > 30 + ADX > 25 | **SHORT** |
| RSI > 80 or < 20 | **BLOCKED** (extreme) |
| ADX < 20 | **BLOCKED** (no trend) |

### Regime Detection

The system detects market regimes and adapts:

| Regime | ADX | Volatility | Action |
|--------|-----|------------|--------|
| **Trending** | > 30 | Normal | Full position size |
| **Ranging** | < 20 | Low | Reduce size 50% |
| **Volatile** | Any | High (>2x avg) | Skip trading |

---

## Portfolio & Risk Management

### Three Account Strategies

| Account | Risk/Trade | Max Positions | Daily Limit | Stop (ATR) | TP (ATR) |
|---------|------------|---------------|-------------|------------|----------|
| **Normal** | 1.5% | 5 | -4% | 1.5x | 2.5x |
| **Aggressive** | 2.5% | 8 | -6% | 1.5x | 2.5x |
| **Mix** | 1.8% | 6 | -5% | 1.5x | 2.5x |

### Position Sizing (Kelly Criterion)

```python
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
position_size = min(kelly_fraction * 0.5, max_risk_per_trade)  # Half-Kelly
```

### Circuit Breakers

| Level | Trigger | Action |
|-------|---------|--------|
| **L1** | -2% daily | Reduce positions 50% |
| **L2** | -5% daily | Stop new positions |
| **L3** | -10% daily | Close all, halt day |
| **L4** | -15% total | Halt 1 week |

---

## Symbols Traded

### Core Universe (16 symbols)

| Category | Symbols |
|----------|---------|
| **Tech** | NVDA, MSFT, AAPL, GOOGL, META, AMZN |
| **Semiconductors** | AMD, AVGO, TSM |
| **ETFs** | QQQ, SPY, IWM |
| **Finance** | JPM, V |
| **Growth** | TSLA, NFLX |

---

## Backtest Results (2024)

### Individual Symbols

| Symbol | Sharpe | Win Rate | Profit Factor | Max DD | Return |
|--------|--------|----------|---------------|--------|--------|
| NVDA | 1.42 | 54% | 1.85 | -8.2% | +32.4% |
| MSFT | 1.15 | 52% | 1.62 | -6.8% | +24.1% |
| QQQ | 1.28 | 53% | 1.74 | -7.1% | +28.7% |

### Portfolio Combined

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.38 |
| Win Rate | 53% |
| Max Drawdown | -9.5% |
| Annual Return | +28.4% |
| Calmar Ratio | 2.99 |

### v1 vs v2 Comparison

| Metric | v1 (Basic) | v2 (Optimized) | Improvement |
|--------|------------|----------------|-------------|
| Profit Factor | 0.84 | 0.93 | +10.7% |
| Win Rate | 49% | 53% | +4% |
| Avg Win | 1.2% | 1.4% | +16.7% |
| Avg Loss | -1.5% | -1.3% | +13.3% |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Kacawaiii/HyprL.git
cd hyprl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Alpaca API

Get your **paper trading** keys at [Alpaca Markets](https://alpaca.markets).

```bash
cp configs/runtime/.env.example configs/runtime/.env.normal
```

Edit `.env.normal`:
```env
APCA_API_KEY_ID=your_api_key_here
APCA_API_SECRET_KEY=your_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
HYPRL_STRATEGY=normal
```

### 3. Test Connection

```bash
python scripts/test_connection.py
```

Expected output:
```
Testing Alpaca connection...
  API Key: XXXXXXXX...
  Base URL: https://paper-api.alpaca.markets

✅ Connection successful!
   Account: XXXXXXXXX
   Equity: $100,000.00
   Cash: $100,000.00
   Status: ACTIVE
```

### 4. Run Backtest

```bash
python scripts/backtest_with_costs.py
```

### 5. Start Paper Trading

```bash
python scripts/run_multi_account_v2.py
```

---

## VPS Deployment (Optional)

### Using systemd

```bash
# Deploy to VPS
./scripts/deploy_to_vps.sh user@your-vps-ip

# On VPS - check status
systemctl status hyprl-multi-account.timer

# View logs
journalctl -u hyprl-multi-account -f
```

The system runs every 15 minutes during market hours (Mon-Fri 9:30-16:00 ET).

---

## Discord Notifications (Optional)

HyprL can send real-time notifications to Discord:

1. Create a Discord bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Run setup:
   ```bash
   python scripts/setup_discord_channels.py
   ```
3. Notifications include:
   - Trade entries/exits
   - Daily summaries
   - Error alerts
   - Position updates

---

## Rust Engine (Optional)

HyprL includes a native Rust accelerator for **37-56x faster** backtests and grid searches.

### Performance Benchmarks

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| ATR 14 (10k bars) | 45ms | 1.2ms | **37x** |
| Single backtest | 450ms | 12ms | **37x** |
| Grid search (1000) | 7.5min | 8sec | **56x** |

### Build Instructions

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build the extension
cd native/hyprl_supercalc
maturin develop --release

# Verify installation
python -c "from hyprl_supercalc import run_backtest; print('Rust engine OK')"
```

### Features

- **Indicators**: SMA, EMA, RSI, MACD, ATR, Bollinger Bands
- **Backtest**: Full vectorized backtesting with costs
- **Grid Search**: Parallel parameter optimization with Rayon
- **PyO3 Bindings**: Seamless Python integration

See [docs/RUST_ENGINE.md](docs/RUST_ENGINE.md) for full API reference.

---

## ML Model Details

### Features Used

```python
FEATURES = [
    'ret_1h', 'ret_3h', 'ret_6h', 'ret_24h',  # Momentum
    'atr_14', 'atr_72', 'atr_14_norm',        # Volatility
    'rsi_14', 'macd', 'macd_hist',            # Technical
    'bb_width', 'volume_ratio'                 # Volume
]
```

### XGBoost Parameters

```python
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

### Feature Importance (NVDA)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | ret_1h | 0.234 |
| 2 | ret_24h | 0.182 |
| 3 | atr_14_norm | 0.148 |
| 4 | rsi_14 | 0.121 |
| 5 | macd_hist | 0.098 |

---

## API Reference

### Core Classes

```python
# Strategy
from src.hyprl.strategy.core import CoreStrategy
strategy = CoreStrategy(config)
signal = strategy.generate_signal(symbol, features)

# Broker
from src.hyprl.broker.alpaca import AlpacaBroker
broker = AlpacaBroker()
broker.submit_order(symbol, qty, side, order_type)

# Risk
from src.hyprl.live.risk import calculate_position_size
size = calculate_position_size(capital, risk_pct, stop_distance)
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/broker/test_alpaca_broker_mock.py -v

# Test with coverage
pytest tests/ --cov=src/hyprl
```

---

## Disclaimer

**THIS SOFTWARE IS FOR EDUCATIONAL PURPOSES ONLY.**

- Paper trading is strongly recommended before live trading
- Past performance does not guarantee future results
- Trading involves significant risk of loss
- You are solely responsible for your trading decisions
- The authors are not responsible for any financial losses

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Kacawaiii/HyprL.git)
- **Discord**: [Join our server](#) (coming soon)

---

*Built with Python, XGBoost, and Alpaca API*
