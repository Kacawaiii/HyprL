# Modèles ML HyprL

## Vue d'ensemble

HyprL utilise des modèles d'ensemble (XGBoost + LightGBM + CatBoost) pour prédire la direction du marché.

## Modèles disponibles

### V3 (XGBoost seul - Production actuelle)

| Fichier | Symbole | Timeframe |
|---------|---------|-----------|
| `models/nvda_1h_xgb_v3.joblib` | NVDA | 1h |
| `models/msft_1h_xgb_v3.joblib` | MSFT | 1h |
| `models/qqq_1h_xgb_v3.joblib` | QQQ | 1h |
| `models/nvda_15m_xgb_v3.joblib` | NVDA | 15m |
| `models/msft_15m_xgb_v3.joblib` | MSFT | 15m |
| `models/qqq_15m_xgb_v3.joblib` | QQQ | 15m |

### V4 (Ensemble - En test)

| Fichier | Symbole | Composition |
|---------|---------|-------------|
| `models/nvda_1h_ensemble_v4.joblib` | NVDA | XGB + LGB + CAT |
| `models/msft_1h_ensemble_v4.joblib` | MSFT | XGB + LGB + CAT |
| `models/qqq_1h_ensemble_v4.joblib` | QQQ | XGB + LGB + CAT |

## Hyperparamètres

### XGBoost V3

```python
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}
```

### LightGBM V4

```python
LGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'seed': 42
}
```

### CatBoost V4

```python
CAT_PARAMS = {
    'iterations': 100,
    'depth': 5,
    'learning_rate': 0.1,
    'l2_leaf_reg': 3.0,
    'random_seed': 42,
    'loss_function': 'Logloss',
    'verbose': False
}
```

## Features utilisées

### Features de base (returns)

```python
RETURN_FEATURES = [
    'ret_1h',      # Return 1 heure
    'ret_3h',      # Return 3 heures
    'ret_6h',      # Return 6 heures
    'ret_24h',     # Return 24 heures
    'ret_5d',      # Return 5 jours
]
```

### Features techniques

```python
TECHNICAL_FEATURES = [
    'atr_14',          # ATR 14 périodes
    'atr_72',          # ATR 72 périodes
    'atr_14_norm',     # ATR normalisé
    'atr_72_norm',     # ATR normalisé
    'rsi_14',          # RSI 14 périodes
    'macd',            # MACD
    'macd_signal',     # Signal MACD
    'macd_hist',       # Histogramme MACD
    'bb_upper',        # Bollinger Upper
    'bb_lower',        # Bollinger Lower
    'bb_width',        # Bollinger Width
]
```

### Features de volume

```python
VOLUME_FEATURES = [
    'volume_sma_20',   # Volume moyenne 20
    'volume_ratio',    # Ratio volume/moyenne
]
```

## Entraînement

### Script d'entraînement

```bash
# V3 (XGBoost seul)
python scripts/train_model_nvda_1h_v3.py

# V4 (Ensemble)
python scripts/train_model_v4_ensemble.py --symbol NVDA --timeframe 1h
```

### Walk-Forward Optimization

```python
WFO_CONFIG = {
    'train_days': 365,      # 1 an de training
    'test_days': 30,        # 1 mois de test
    'step_days': 30,        # Avancer de 1 mois
    'n_splits': 12,         # 12 splits
    'purge_days': 2,        # Gap pour éviter leakage
}
```

## Calibration des seuils

### Seuils de décision

```python
THRESHOLDS = {
    'long_threshold': 0.55,    # Prob > 55% → LONG
    'short_threshold': 0.45,   # Prob < 45% → SHORT
    'min_confidence': 0.10,    # |prob - 0.5| > 10%
}
```

### Configuration par symbole (Aggressive)

```yaml
symbols:
  NVDA:
    long_threshold: 0.52
    short_threshold: 0.48
  MSFT:
    long_threshold: 0.52
    short_threshold: 0.48
  QQQ:
    long_threshold: 0.52
    short_threshold: 0.48
```

## Performance des modèles

### Backtest V3 (2024)

| Symbole | Sharpe | Win Rate | Max DD |
|---------|--------|----------|--------|
| NVDA | 1.42 | 54% | -8.2% |
| MSFT | 1.15 | 52% | -6.8% |
| QQQ | 1.28 | 53% | -7.1% |

## Code de prédiction

### Chemin: `src/hyprl/model/probability.py`

```python
class ProbabilityModel:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

    def predict(self, features: pd.DataFrame) -> float:
        """Return probability of price going UP."""
        proba = self.model.predict_proba(features)
        return proba[0][1]  # Classe 1 = UP
```

### Utilisation

```python
from src.hyprl.model.probability import ProbabilityModel

model = ProbabilityModel("models/nvda_1h_xgb_v3.joblib")
prob_up = model.predict(features_df)

if prob_up > 0.55:
    signal = "long"
elif prob_up < 0.45:
    signal = "short"
else:
    signal = "flat"
```
