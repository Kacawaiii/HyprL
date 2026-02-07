# HyprL Research Starter Kit

This is a research-only template for building a crypto ML pipeline:
- Data -> features -> XGBoost models
- Signal generation
- Paper execution via Alpaca

No data or API keys are included. This package is for research and paper trading only.

## Contents
- Core Python package: `src/hyprl/`
- Training: `scripts/train_crypto_xgb.py`
- Signals: `scripts/run_crypto_signals.py`
- Bridge (paper): `scripts/execution/run_alpaca_bridge.py`
- Backtest + analysis: `scripts/run_backtest.py`, `scripts/analyze_trades.py`

## Quickstart
1) Create env and install deps
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2) Set env vars
```
cp .env.example .env.bridge
# Edit .env.bridge with Alpaca paper keys
```

3) Train models (1y, 1h bars)
```
python scripts/train_crypto_xgb.py \
  --symbols BTC/USD ETH/USD \
  --days 365 \
  --timeframe 1Hour \
  --output-dir models/crypto
```

4) Generate signals
```
python scripts/run_crypto_signals.py \
  --symbols BTC/USD ETH/USD \
  --timeframe 5Min \
  --threshold-long 0.52 \
  --threshold-short 0.20 \
  --max-position-pct 0.40 \
  --base-dir .
```

5) Run paper bridge (reads signals from `live/logs/crypto_signals.jsonl`)
```
python scripts/ops/run_crypto_signals.sh
python scripts/execution/run_alpaca_bridge.py \
  --enable-crypto \
  --crypto-signals live/logs/crypto_signals.jsonl \
  --paper \
  --once
```

## Notes
- Signals are logged to `live/logs/crypto_signals.jsonl`.
- Crypto uses notional orders (minimum $10 on Alpaca).
- Shorts on crypto are disabled by default.

## Support
This is a template. You own the deployment and data. Use at your own risk.
