# MT5 Data Workflow - HyprL

Guide pour télécharger des données historiques MT5 et entraîner des modèles sur timeframes courts.

## Prérequis

### Windows (avec MT5 installé)

```powershell
pip install MetaTrader5 pandas pyarrow polars xgboost scikit-learn joblib
```

### Structure des données

```
D:/hyprl_data/mt5/
├── EURUSD_1m.parquet      # 1.5M+ barres (10 ans)
├── EURUSD_5m.parquet      # 300k+ barres
├── EURUSD_15m.parquet     # 100k+ barres
├── EURUSD_1h.parquet      # 25k+ barres
├── NVDA_15m.parquet
├── MSFT_15m.parquet
└── ...
```

## Étape 1: Télécharger les données

### Tous les symboles par défaut (10 ans)

```powershell
python scripts/data/mt5_downloader.py --output D:/hyprl_data/mt5 --years 10
```

### Symboles spécifiques

```powershell
python scripts/data/mt5_downloader.py \
    --output D:/hyprl_data/mt5 \
    --symbols EURUSD,GBPUSD,NVDA,MSFT,QQQ \
    --timeframes 1m,5m,15m,1h \
    --years 10
```

### Voir les symboles disponibles

```powershell
python scripts/data/mt5_downloader.py --list-symbols
```

## Étape 2: Entraîner les modèles

### Un modèle spécifique

```powershell
python scripts/data/train_from_mt5.py \
    --data D:/hyprl_data/mt5 \
    --symbol EURUSD \
    --timeframe 15m \
    --output models/mt5
```

### Tous les modèles disponibles

```powershell
python scripts/data/train_from_mt5.py \
    --data D:/hyprl_data/mt5 \
    --all \
    --output models/mt5
```

## Étape 3: Backtest avec moteur Rust

```python
from hyprl.native.supercalc import run_fast_backtest
import polars as pl

# Charger données
df = pl.read_parquet("D:/hyprl_data/mt5/EURUSD_15m.parquet")

# Générer signaux avec le modèle
signals = model.predict_proba(features)[:, 1]
signals = (signals > 0.55).astype(float) - (signals < 0.45).astype(float)

# Backtest Rust (100x plus rapide que Python)
report = run_fast_backtest(
    candles=df,
    signal=signals,
    risk_pct=0.01,
    atr_mult_stop=2.0,
    atr_mult_tp=4.0,
)
print(f"Sharpe: {report['sharpe']:.2f}")
print(f"Profit Factor: {report['profit_factor']:.2f}")
```

## Estimation des données

| Timeframe | Barres/an | 10 ans | Taille Parquet |
|-----------|-----------|--------|----------------|
| 1m | 525,600 | 5.2M | ~200 MB |
| 5m | 105,120 | 1.0M | ~40 MB |
| 15m | 35,040 | 350k | ~15 MB |
| 1h | 8,760 | 87k | ~4 MB |

Pour 20 symboles × 4 timeframes × 10 ans ≈ **5 GB total**

## Symboles recommandés

### Forex (24/7, très liquide)
- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD

### Indices US
- US30 (Dow), US500 (S&P), NAS100 (Nasdaq)

### Commodities
- XAUUSD (Gold), XAGUSD (Silver)

### US Stocks (si dispo sur ton broker)
- NVDA, MSFT, AAPL, GOOGL, TSLA, QQQ

## Performance attendue

Avec 100k+ barres d'entraînement:

| Timeframe | Accuracy attendue | Note |
|-----------|-------------------|------|
| 1H (actuel) | 60-65% | Déjà bon |
| 15min | 58-62% | Devrait s'améliorer |
| 5min | 55-60% | Plus de bruit |
| 1min | 52-57% | Très bruité, scalping |

## Troubleshooting

### "MT5 initialize failed"
- MT5 doit être ouvert et connecté à un compte
- Vérifier que le compte est actif (démo ou réel)

### "Symbol not found"
- Les noms varient selon le broker
- Utiliser `--list-symbols` pour voir les noms exacts
- NVDA peut être "NVDA.US", "NVDA.r", etc.

### Données incomplètes
- Certains brokers limitent l'historique
- Essayer un autre broker avec plus d'historique
- Les données forex sont généralement plus complètes
