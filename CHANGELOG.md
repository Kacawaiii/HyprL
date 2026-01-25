# Changelog

All notable changes to HyprL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-26

### Added
- Initial public release
- XGBoost ML model for price direction prediction
- Technical indicators: RSI, MACD, Stochastic RSI, ADX, VWAP, Bollinger Bands
- Multi-account trading support (Normal, Aggressive, Mix strategies)
- Kelly criterion position sizing
- Market regime detection (trending/ranging/volatile)
- ATR-based stop loss and take profit
- Circuit breakers for risk management
- Discord notifications for trade alerts
- VPS deployment with systemd
- Rust native engine for 37-56x faster backtests
- Alpaca paper trading integration

### Performance (Backtest 2024)
- Sharpe Ratio: 1.42
- Win Rate: 54%
- Max Drawdown: -8.2%
- Annual Return: +28%

### Symbols Supported
- Tech: NVDA, MSFT, AAPL, GOOGL, META, AMZN
- Semiconductors: AMD, AVGO, TSM
- ETFs: QQQ, SPY, IWM
- Finance: JPM, V
- Growth: TSLA, NFLX

---

## [Unreleased]

### Planned
- Interactive Brokers support
- Web dashboard
- More ML models (LightGBM, CatBoost ensemble)
- Options trading module
- Crypto support (Binance, Coinbase)
