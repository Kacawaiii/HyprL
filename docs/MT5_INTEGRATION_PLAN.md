# MT5 Integration Plan

## Overview

Integration of MetaTrader 5 for:
1. Historical data (better quality than yfinance)
2. Live execution (replace/complement Alpaca)
3. Real-time quotes

## Architecture

```
Windows Side                    WSL2/Linux Side
─────────────                   ───────────────
MT5 Terminal                    Python (HyprL)
    │                               │
    ├── HyprL_Bridge.mq5           │
    │   (Enhanced EA)               │
    │                               │
    └────── HTTP/WebSocket ────────┘
            localhost:5555
```

## Phase 1: Data Pipeline (No Live Trading)

### Goal
Use MT5 historical data for backtesting and model training.

### Files to Create

1. **`src/hyprl/data/mt5_fetcher.py`**
   - Connect to MT5 bridge
   - Fetch OHLCV bars for any symbol/timeframe
   - Cache locally like yfinance

2. **`mt5/HyprL_DataServer.mq5`**
   - Enhanced EA that serves historical data
   - HTTP endpoint: GET /bars?symbol=NVDA&timeframe=H1&count=10000

3. **`scripts/data/download_mt5_history.py`**
   - Bulk download historical data
   - Save to parquet files

### Data Storage
```
data/
├── mt5/
│   ├── NVDA_H1.parquet
│   ├── MSFT_H1.parquet
│   └── META_H1.parquet
```

## Phase 2: Enhanced Model Training

### Goal
Train XGBoost with more data, proper walk-forward validation.

### Process
```
1. Download 5+ years of H1 data from MT5
2. Feature engineering (same 28 features)
3. Walk-forward validation:
   - Train on 3 years
   - Validate on 6 months
   - Test on 6 months
   - Roll forward, repeat
4. Ensemble: XGBoost + LightGBM + CatBoost
5. Early stopping on validation loss
```

### Files to Create

1. **`scripts/train_model_v5.py`**
   - Uses MT5 data
   - Walk-forward cross-validation
   - Hyperparameter tuning with Optuna

2. **`configs/training/training_config.yaml`**
   ```yaml
   data_source: mt5
   symbols: [NVDA, MSFT]
   timeframe: H1
   train_years: 3
   val_months: 6
   test_months: 6

   model:
     type: ensemble
     estimators:
       - xgboost: 0.5
       - lightgbm: 0.3
       - catboost: 0.2

   regularization:
     max_depth: 3
     reg_lambda: 3.0
     reg_alpha: 0.5
     early_stopping: 50
   ```

## Phase 3: Live Execution via MT5

### Goal
Execute trades through MT5 instead of/alongside Alpaca.

### Files to Create

1. **`src/hyprl/broker/mt5_bridge.py`**
   ```python
   class MT5BridgeBroker(BrokerBase):
       """Broker that communicates with MT5 via HTTP bridge."""

       def __init__(self, bridge_url: str = "http://localhost:5555"):
           self.bridge_url = bridge_url

       async def submit_order(self, symbol, side, qty, ...):
           # POST to bridge
           pass
   ```

2. **`mt5/HyprL_Bridge_v2.mq5`**
   - Enhanced EA with:
     - Order execution endpoint
     - Position reporting
     - Account info
     - WebSocket for real-time updates

### Symbol Mapping
```json
// configs/mt5_symbols.json
{
  "NVDA": "NVDA.US",
  "MSFT": "MSFT.US",
  "META": "META.US",
  "BTCUSD": "BTCUSD",
  "ETHUSD": "ETHUSD"
}
```

## Phase 4: Optional LLM Sentiment

### Goal
Add news sentiment as one additional feature.

### Architecture
```
News API (Alpaca/NewsAPI)
         │
         ▼
   LLM (Mistral 7B local)
         │
         ▼
   sentiment_score: float (-1 to +1)
         │
         ▼
   Added as feature #29 to XGBoost
```

### Files to Create

1. **`src/hyprl/sentiment/llm_analyzer.py`**
   ```python
   class LLMSentimentAnalyzer:
       def __init__(self, model="mistral"):
           self.model = load_model(model)

       def analyze(self, headlines: List[str]) -> float:
           prompt = f"Analyze sentiment: {headlines}"
           response = self.model.generate(prompt)
           return parse_sentiment(response)
   ```

2. **`configs/sentiment/llm_config.yaml`**
   ```yaml
   enabled: true
   model: mistral-7b-instruct
   update_frequency: 1h  # Don't run on every bar
   cache_ttl: 3600
   ```

## Installation Checklist

### WSL2 (Python)
```bash
# Core dependencies
pip install websockets aiofiles httpx

# For LLM (optional)
pip install transformers torch
# OR
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
```

### Windows (MT5)
1. Install MT5 from broker
2. Enable algo trading in settings
3. Copy `mt5/*.mq5` to `MQL5/Experts/`
4. Compile and attach to chart
5. Configure bridge port (default 5555)

## Timeline

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | MT5 Data Pipeline | TODO |
| 2 | Model Training v5 | TODO |
| 3 | Live Execution | TODO |
| 4 | LLM Sentiment | Optional |

## Risk Considerations

1. **MT5 runs on Windows only** - Need bridge architecture
2. **Symbol naming differs by broker** - Need mapping config
3. **Timezone handling** - MT5 server time vs market time
4. **Connection stability** - Need reconnect logic

## Current Assets to Leverage

- Existing `BrokerBase` abstraction
- Existing feature engineering pipeline
- Existing risk management (ATR stops, circuit breakers)
- Existing MQL5 bridge (needs enhancement)
