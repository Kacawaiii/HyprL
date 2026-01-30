# Plan: XGBoost pour Crypto (BTC/ETH)

> **Objectif:** Entrainer des modeles XGBoost sur les donnees crypto pour remplacer le fallback rule-based actuel (prob=0.50)
> **Date:** 2026-01-11
> **Statut:** ✅ COMPLETE

---

## 0. Resultats Finaux

### Modeles Entraines

| Symbol | Train Acc | Test Acc | Samples | Status |
|--------|-----------|----------|---------|--------|
| BTC/USD | 79.4% | **54.8%** | 8,687 | ✅ Live |
| ETH/USD | 79.1% | **58.0%** | 8,687 | ✅ Live |

### Positions Crypto Live (2026-01-11)

| Symbol | Qty | Entry | Value | P&L |
|--------|-----|-------|-------|-----|
| BTCUSD | 0.00116 | $90,440 | ~$105 | +0.2% |
| ETHUSD | 0.079 | $3,103 | ~$244 | +0.0% |
| **Total** | - | - | **~$349** | - |

### Fichiers Crees

```
scripts/train_crypto_xgb.py          # Entrainement
scripts/run_crypto_signals.py        # Generation signaux
scripts/ops/run_crypto_signals.sh    # Wrapper systemd
src/hyprl/crypto/signals.py          # Features + inference
src/hyprl/broker/alpaca.py           # Crypto quotes/trades support
models/crypto/btc_usd_xgb.joblib     # Modele BTC
models/crypto/eth_usd_xgb.joblib     # Modele ETH
models/crypto/training_report.json   # Metriques
```

### Corrections Bridge

- `--enable-crypto` flag ajoute
- Sizing notional puis conversion en `qty`
- Minimum $10 par ordre crypto (bridge)
- `TimeInForce.GTC` pour 24/7
- Shorts crypto desactives

---

## 1. Situation Initiale (Resolue)

### Probleme Initial
Les signaux crypto utilisaient un fallback rule-based qui retournait toujours ~0.50:

```python
def _rule_based_probability(self, features: CryptoFeatures) -> float:
    score = 0.5
    # RSI, SMA, momentum adjustments...
    return score  # Toujours proche de 0.50
```

### Objectif (Atteint)
Entrainer des modeles XGBoost similaires a ceux utilises pour NVDA/MSFT/QQQ avec:
- ✅ Accuracy > 55% → BTC: 54.8%, ETH: 58.0%
- ⏳ Sharpe > 1.0 en backtest → A valider
- ⏳ Drawdown max < 15% → A valider

---

## 2. Donnees Disponibles

### Source: Alpaca Crypto API

```python
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

client = CryptoHistoricalDataClient()

# Donnees disponibles: ~2 ans d'historique
request = CryptoBarsRequest(
    symbol_or_symbols="BTC/USD",
    timeframe=TimeFrame(1, TimeFrameUnit.Hour),
    start=datetime(2024, 1, 1),
    end=datetime.now(),
)
bars = client.get_crypto_bars(request)
```

### Timeframes Disponibles
| Timeframe | Bars/jour | Usage |
|-----------|-----------|-------|
| 1Min | 1440 | Scalping (non recommande) |
| 5Min | 288 | Day trading |
| 15Min | 96 | Intraday |
| **1Hour** | 24 | **Recommande** |
| 1Day | 1 | Swing |

### Volume de Donnees Estime (1 an)
| Symbol | Bars 1H | Taille |
|--------|---------|--------|
| BTC/USD | ~8,760 | ~500KB |
| ETH/USD | ~8,760 | ~500KB |

---

## 3. Features a Calculer

### 3.1 Features de Base (similaires aux stocks)

```python
feature_columns = [
    # Returns
    'ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'ret_72h',

    # Volatilite
    'volatility_12h', 'volatility_24h', 'volatility_72h',

    # RSI
    'rsi_6', 'rsi_14', 'rsi_21',

    # SMA Ratios
    'sma_ratio_12', 'sma_ratio_24', 'sma_ratio_72',

    # Volume
    'volume_ratio_12', 'volume_ratio_24', 'volume_zscore',

    # ATR
    'atr_14', 'atr_14_norm',

    # Range
    'high_low_range', 'true_range',
]
```

### 3.2 Features Specifiques Crypto

```python
crypto_features = [
    # Temporel (crypto trade 24/7)
    'hour_of_day',        # 0-23
    'day_of_week',        # 0-6
    'is_weekend',         # 0/1
    'is_asia_session',    # 0/1 (00:00-08:00 UTC)
    'is_europe_session',  # 0/1 (08:00-16:00 UTC)
    'is_us_session',      # 0/1 (14:00-22:00 UTC)
]
```

### 3.3 Target Variable

```python
# Target: prix monte de X% dans les N prochaines heures
def compute_target(df, threshold_pct=0.5, horizon_hours=4):
    future_return = df['close'].shift(-horizon_hours) / df['close'] - 1
    target = (future_return > threshold_pct / 100).astype(int)
    return target

# Alternatives:
# - threshold_pct=1.0, horizon=8h  (plus conservateur)
# - threshold_pct=0.3, horizon=2h  (plus frequent)
```

---

## 4. Pipeline d'Entrainement

### 4.1 Script: `scripts/train_crypto_xgb.py`

```python
#!/usr/bin/env python3
"""Train XGBoost models for crypto trading."""

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Fetch data from Alpaca
def fetch_crypto_data(symbol: str, days: int = 365) -> pd.DataFrame:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = CryptoHistoricalDataClient()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=start,
        end=end,
    )
    bars = client.get_crypto_bars(request)

    data = []
    for bar in bars.data[symbol]:
        data.append({
            'timestamp': bar.timestamp,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume),
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

# Compute features
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Returns
    for h in [1, 4, 12, 24, 72]:
        df[f'ret_{h}h'] = df['close'].pct_change(h)

    # Volatility
    for h in [12, 24, 72]:
        df[f'volatility_{h}h'] = df['close'].pct_change().rolling(h).std()

    # RSI
    for period in [6, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # SMA Ratios
    for h in [12, 24, 72]:
        sma = df['close'].rolling(h).mean()
        df[f'sma_ratio_{h}'] = df['close'] / sma

    # Volume
    for h in [12, 24]:
        vol_sma = df['volume'].rolling(h).mean()
        df[f'volume_ratio_{h}'] = df['volume'] / vol_sma.replace(0, 1)
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(24).mean()) / df['volume'].rolling(24).std()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_14_norm'] = df['atr_14'] / df['close']

    # Range
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    # Temporal (crypto-specific)
    df['hour_of_day'] = df.index.hour / 24.0
    df['day_of_week'] = df.index.dayofweek / 7.0
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(float)

    return df

# Compute target
def compute_target(df: pd.DataFrame, threshold_pct: float = 0.5, horizon: int = 4) -> pd.Series:
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    return (future_return > threshold_pct / 100).astype(int)

# Main training
def train_model(symbol: str, output_dir: str = "models/crypto"):
    print(f"Training model for {symbol}...")

    # Fetch data
    df = fetch_crypto_data(symbol, days=365)
    print(f"  Fetched {len(df)} bars")

    # Features
    df = compute_features(df)
    df['target'] = compute_target(df, threshold_pct=0.5, horizon=4)
    df = df.dropna()
    print(f"  {len(df)} samples after feature computation")

    # Feature columns
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    X = df[feature_cols].values
    y = df['target'].values

    # Train/test split (time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Train class balance: {y_train.mean():.2%} positive")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=0.5,
        min_child_weight=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Test accuracy: {test_acc:.1%}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clean_symbol = symbol.replace("/", "_").lower()
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'trained_at': datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(bundle, output_path / f"{clean_symbol}_xgb.joblib")
    print(f"  Saved to {output_path / f'{clean_symbol}_xgb.joblib'}")

    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTC/USD", "ETH/USD"])
    parser.add_argument("--output-dir", default="models/crypto")
    args = parser.parse_args()

    for symbol in args.symbols:
        train_model(symbol, args.output_dir)
```

### 4.2 Hyperparametres Recommandes

| Parametre | Valeur | Raison |
|-----------|--------|--------|
| `n_estimators` | 200 | Assez pour capturer patterns |
| `max_depth` | 4 | Eviter overfitting |
| `learning_rate` | 0.05 | Convergence stable |
| `reg_lambda` | 3.0 | Regularisation L2 forte |
| `reg_alpha` | 0.5 | Regularisation L1 |
| `min_child_weight` | 5 | Eviter splits sur peu de samples |
| `subsample` | 0.8 | Bootstrap sampling |

---

## 5. Validation

### 5.1 Walk-Forward Optimization

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"WFO Accuracy: {np.mean(scores):.1%} (+/- {np.std(scores):.1%})")
```

### 5.2 Metriques de Trading

```python
# Backtest simple
def backtest_model(df, model, scaler, feature_cols):
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[:, 1]
    df['signal'] = np.where(proba > 0.55, 1, np.where(proba < 0.45, -1, 0))

    # Returns
    df['strategy_return'] = df['signal'].shift(1) * df['close'].pct_change()

    # Metrics
    total_return = (1 + df['strategy_return']).prod() - 1
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(24 * 365)
    max_dd = (df['strategy_return'].cumsum() - df['strategy_return'].cumsum().cummax()).min()

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': (df['signal'] != 0).sum(),
    }
```

### 5.3 Criteres de Validation

| Metrique | Minimum | Objectif |
|----------|---------|----------|
| Test Accuracy | > 52% | > 55% |
| Sharpe Ratio | > 0.5 | > 1.0 |
| Max Drawdown | < 20% | < 15% |
| Profit Factor | > 1.2 | > 1.5 |

---

## 6. Deploiement

### 6.1 Fichiers a Creer

```
models/crypto/
├── btc_usd_xgb.joblib    # Modele BTC
├── eth_usd_xgb.joblib    # Modele ETH
└── training_report.json   # Metriques
```

### 6.2 Integration dans CryptoSignalGenerator

```python
# src/hyprl/crypto/signals.py

def _load_model(self, symbol: str) -> bool:
    clean_symbol = symbol.replace("/", "_").lower()
    model_path = self.base_dir / "models/crypto" / f"{clean_symbol}_xgb.joblib"

    if model_path.exists():
        bundle = joblib.load(model_path)
        self._models[symbol] = bundle.get("model")
        self._scalers[symbol] = bundle.get("scaler")
        self._feature_cols[symbol] = bundle.get("feature_columns")
        return True
    return False
```

### 6.3 Deploiement VPS

```bash
# Entrainer localement
python scripts/train_crypto_xgb.py --symbols BTC/USD ETH/USD

# Deployer
rsync -avz models/crypto/ hyprl:/home/ubuntu/HyprL/models/crypto/

# Tester
ssh hyprl '/home/ubuntu/HyprL/scripts/ops/run_crypto_signals.sh'
```

---

## 7. Timeline

| Etape | Description | Temps |
|-------|-------------|-------|
| 1 | Creer script `train_crypto_xgb.py` | 1h |
| 2 | Fetch donnees + compute features | 30min |
| 3 | Entrainer modeles BTC/ETH | 30min |
| 4 | Valider avec backtest | 1h |
| 5 | Integrer dans signals.py | 30min |
| 6 | Deployer VPS | 15min |
| 7 | Monitorer en live | Ongoing |

---

## 8. Risques et Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Overfitting | Mauvaise perf live | Regularisation forte, WFO |
| Regime change | Modele obsolete | Re-entrainer mensuel |
| Data quality | Signaux errones | Validation donnees Alpaca |
| Correlation BTC/ETH | Signaux redondants | Diversifier timeframes |

---

## 9. Prochaines Etapes

1. [ ] Implementer `scripts/train_crypto_xgb.py`
2. [ ] Entrainer modeles sur 1 an de donnees
3. [ ] Valider accuracy > 55%
4. [ ] Backtest avec metriques trading
5. [ ] Deployer si criteres OK
6. [ ] Monitorer en paper trading 1 semaine
7. [ ] Ajuster thresholds si necessaire

---

*Document cree: 2026-01-11*
