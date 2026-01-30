# Quickstart

## 1) Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2) Env
```
cp .env.example .env.bridge
# Set Alpaca paper keys
```

## 3) Train models
```
python scripts/train_crypto_xgb.py --symbols BTC/USD ETH/USD --days 365 --timeframe 5Min
```

## 4) Generate signals
```
python scripts/ops/run_crypto_signals.sh
```

## 5) Paper bridge
```
python scripts/execution/run_alpaca_bridge.py --enable-crypto --paper --once
```

## 6) Validate
- Check `live/logs/crypto_signals.jsonl`
- Check `live/execution/crypto/orders.jsonl`
