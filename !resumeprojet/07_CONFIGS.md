# Configurations HyprL

## Structure des configs

```
configs/
├── NVDA-1h_v3.yaml          # Config NVDA 1h
├── MSFT-1h_v3.yaml          # Config MSFT 1h
├── QQQ-1h_v3.yaml           # Config QQQ 1h
├── NVDA-15m_v3.yaml         # Config NVDA 15m
├── portfolio_core_15m_v3.yaml
└── runtime/                  # Configs de production
    ├── strategy_aggressive.yaml
    ├── strategy_normal.yaml
    └── strategy_mix.yaml
```

## Config symbole (NVDA-1h_v3.yaml)

```yaml
# Symbole et timeframe
ticker: NVDA
group: tech
interval: 1h
period: 730d  # 2 ans de données

# Modèle
model:
  type: xgboost
  artifact: models/nvda_1h_xgb_v3.joblib
  calibration: none
  seed: 42
  preset: nvda_v2

  # Features utilisées
  feature_columns:
    - ret_1h
    - ret_3h
    - ret_6h
    - ret_24h
    - atr_14
    - atr_72
    - atr_14_norm
    - atr_72_norm
    - rsi_14
    - macd
    - macd_signal
    - macd_hist
    - bb_width
    - volume_ratio

# Seuils de décision
thresholds:
  long: 0.55       # Prob > 55% → LONG
  short: 0.45      # Prob < 45% → SHORT

# Gestion du risque
risk:
  stop_loss_atr: 3.0      # Stop à 3 ATR
  take_profit_atr: 6.0    # TP à 6 ATR (2:1 R/R)
  trailing_stop_atr: 2.0  # Trailing à 2 ATR
  position_size_pct: 0.25 # 25% du capital par position

# Walk-Forward Optimization
wfo:
  train_days: 365
  test_days: 30
  step_days: 30
  n_splits: 12
  purge_days: 2
```

## Config stratégie (strategy_aggressive.yaml)

```yaml
# ============================================================
# STRATEGY: AGGRESSIVE v2 (Higher Risk)
# ============================================================

strategy:
  name: "aggressive"
  description: "High risk, maximum capital deployment"
  version: "2.0.0"

# Allocation de capital
capital:
  position_size_pct: 0.40      # 40% par trade
  max_position_pct: 0.60       # 60% max par symbole
  max_total_exposure: 1.80     # 180% levier
  min_cash_reserve: 0.02       # 2% cash minimum

# Gestion du risque
risk:
  stop_loss_atr: 2.5           # Stop à 2.5 ATR (serré)
  take_profit_atr: 5.0         # TP à 5 ATR
  max_daily_loss_pct: 0.10     # -10% max/jour
  max_drawdown_pct: 0.20       # -20% drawdown = pause
  use_trailing_stop: true

# Filtres
filters:
  enable_smart_filter: true    # Filtre RSI/momentum
  enable_sentiment: true       # Filtre news/social
  enable_quality: false
  enable_mtf: false

# Symboles et seuils personnalisés
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

# Sentiment filter config
sentiment:
  min_confidence: 0.3
  bullish_block_threshold: 0.4  # Bloquer short si sentiment > 0.4
  bearish_block_threshold: -0.4 # Bloquer long si sentiment < -0.4
  sources:
    - reddit
    - stocktwits
    - finviz
```

## Config Normal

```yaml
strategy:
  name: "normal"

capital:
  position_size_pct: 0.20      # 20% par trade
  max_position_pct: 0.40       # 40% max par symbole
  max_total_exposure: 1.00     # 100% pas de levier
  min_cash_reserve: 0.05       # 5% cash minimum

risk:
  stop_loss_atr: 3.0
  take_profit_atr: 6.0
  max_daily_loss_pct: 0.05
  max_drawdown_pct: 0.15

filters:
  enable_smart_filter: true
  enable_sentiment: true
```

## Config Mix

```yaml
strategy:
  name: "mix"

capital:
  position_size_pct: 0.30      # 30% par trade (entre normal et agg)
  max_position_pct: 0.50
  max_total_exposure: 1.40     # 140% levier modéré

risk:
  stop_loss_atr: 2.75          # Entre 2.5 et 3.0
  take_profit_atr: 5.5         # Entre 5.0 et 6.0
```

## Variables d'environnement (.env)

```bash
# .env.aggressive
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_PAPER=true

# Optional
LOG_LEVEL=INFO
ENABLE_DISCORD_ALERTS=false
```

## Hyperparamètres ML

### XGBoost V3

```python
XGB_PARAMS = {
    # Structure
    'n_estimators': 100,
    'max_depth': 5,
    'min_child_weight': 3,

    # Learning
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,

    # Regularization
    'gamma': 0.1,
    'reg_alpha': 0.1,   # L1
    'reg_lambda': 1.0,  # L2

    # Objective
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',

    # Reproducibility
    'seed': 42,
    'n_jobs': -1,
}
```

### LightGBM V4

```python
LGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
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
    'verbose': False,
    'thread_count': -1,
}
```

## Paramètres de risque

### Position Sizing

```python
SIZING_PARAMS = {
    # Kelly Criterion (fraction)
    'kelly_fraction': 0.25,  # 25% du Kelly optimal

    # ATR-based sizing
    'risk_per_trade_pct': 0.02,  # 2% du capital risqué par trade

    # Fixed fractional
    'fixed_fraction': 0.20,  # 20% du capital par trade
}
```

### Stop Loss / Take Profit

```python
SL_TP_PARAMS = {
    # ATR multiples
    'stop_loss_atr_mult': 2.5,
    'take_profit_atr_mult': 5.0,

    # Trailing stop
    'trailing_activation_atr': 2.0,  # Activer à 2 ATR de profit
    'trailing_distance_atr': 1.5,    # Distance de 1.5 ATR

    # Time-based
    'max_hold_hours': 24,
}
```

## Paramètres Sentiment

```python
SENTIMENT_PARAMS = {
    # Seuils
    'bullish_threshold': 0.3,
    'bearish_threshold': -0.3,
    'min_confidence': 0.4,

    # Poids des sources
    'source_weights': {
        'finviz_news': 0.65,
        'reddit_investing': 0.60,
        'reddit_stocks': 0.55,
        'stocktwits': 0.50,
        'reddit_wsb': 0.45,
        'fear_greed': 0.70,
    },

    # Refresh interval
    'refresh_minutes': 15,
}
```

## Paramètres Exit Monitor

```python
EXIT_MONITOR_PARAMS = {
    # P/L thresholds
    'profit_take_pct': 2.0,
    'loss_cut_pct': -1.5,

    # Sentiment reversal
    'sentiment_reversal_threshold': 0.3,

    # Time
    'max_hold_hours': 24,
    'check_interval_seconds': 300,
}
```
