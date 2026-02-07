# HyprL - Documentation Complète

## Résumé Exécutif

**HyprL** est un système de trading algorithmique ML complet qui:
- Prédit la direction du marché US (NVDA, MSFT, QQQ) avec XGBoost/LightGBM/CatBoost
- Filtre les signaux avec sentiment analysis (Reddit, StockTwits, Finviz)
- Exécute automatiquement via Alpaca et MetaTrader 5
- Gère le risque avec stops dynamiques, position tracking et exits intelligents

### Performance

| Métrique | Valeur |
|----------|--------|
| **Sharpe Ratio** | 1.42 |
| **Win Rate** | 54% |
| **Max Drawdown** | -8.2% |
| **Annual Return** | +28% |

---

## Structure du Projet

```
HyprL/
├── src/hyprl/                    # Code source
│   ├── backtest/                 # Backtesting
│   ├── broker/                   # Alpaca integration
│   ├── features/                 # Feature engineering
│   ├── indicators/               # Indicateurs techniques
│   ├── model/                    # Modèles ML
│   ├── native/                   # Bindings Rust
│   ├── sentiment/                # Sentiment analysis
│   │   ├── multi_source.py       # Scraping multi-source
│   │   └── trading_filter.py     # Filtre de trading
│   ├── strategy/                 # Stratégies
│   │   ├── core.py               # Stratégie principale
│   │   ├── position_aware.py     # Position tracking
│   │   ├── smart_exit_monitor.py # Exit intelligent
│   │   └── smart_filter.py       # Filtre RSI/momentum
│   └── risk/                     # Gestion risque
├── scripts/
│   ├── execution/
│   │   ├── run_strategy_bridge.py  # Bridge de trading
│   │   └── mt5_signal_api.py       # API MT5
│   └── run_live_hour.py            # Générateur signaux
├── models/                       # Modèles entraînés
├── configs/                      # Configurations
├── native/hyprl_supercalc/       # Calculateur Rust
├── apps/
│   ├── landing/                  # Site web
│   └── track_record/             # Dashboard
└── deploy/systemd/               # Services
```

---

## Modèles ML

### Fichiers

| Modèle | Chemin |
|--------|--------|
| NVDA 1h V3 | `models/nvda_1h_xgb_v3.joblib` |
| MSFT 1h V3 | `models/msft_1h_xgb_v3.joblib` |
| QQQ 1h V3 | `models/qqq_1h_xgb_v3.joblib` |
| NVDA 1h V4 | `models/nvda_1h_ensemble_v4.joblib` |
| MSFT 1h V4 | `models/msft_1h_ensemble_v4.joblib` |
| QQQ 1h V4 | `models/qqq_1h_ensemble_v4.joblib` |

### Hyperparamètres XGBoost

```python
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

### Features

```python
FEATURES = [
    'ret_1h', 'ret_3h', 'ret_6h', 'ret_24h',  # Momentum
    'atr_14', 'atr_72', 'atr_14_norm',        # Volatilité
    'rsi_14', 'macd', 'macd_hist',            # Techniques
    'bb_width', 'volume_ratio'                 # Volume
]
```

---

## Filtres de Trading

### Smart Filter
- Bloque LONG si RSI > 70
- Bloque SHORT si RSI < 30
- Bloque si momentum contraire (ret_1h > 2%)

### Sentiment Filter
Sources avec scores de fiabilité:

| Source | Fiabilité |
|--------|-----------|
| Fear & Greed Index | 70% |
| Finviz News | 65% |
| Reddit Investing | 60% |
| Reddit Stocks | 55% |
| StockTwits | 50% |
| Reddit WSB | 45% |

- Bloque si sentiment contradictoire (delta > 0.3)

### Position-Aware Exit Rules

| Situation | Action | Urgence |
|-----------|--------|---------|
| Profit > 1% + sentiment retourne | CLOSE | Soon |
| Profit > 2% | CLOSE | Soon |
| Perte + sentiment empire | CLOSE | Immediate |
| Perte > 1.5% | CLOSE | Immediate |
| Position > 24h + profit | CLOSE | Soon |

---

## Configurations

### Stratégies

| Stratégie | Position Size | Levier | Stop ATR |
|-----------|--------------|--------|----------|
| Normal | 20% | 1.0x | 3.0 |
| Mix | 30% | 1.4x | 2.75 |
| Aggressive | 40% | 1.8x | 2.5 |

### Seuils de décision

```yaml
symbols:
  NVDA:
    long_threshold: 0.52   # >52% → LONG
    short_threshold: 0.48  # <48% → SHORT
```

---

## Calculateur Rust

### Chemin: `native/hyprl_supercalc/`

Fonctions optimisées (10-50x plus rapide que Python):
- `sma()`, `ema()` - Moyennes mobiles
- `compute_atr_fast()` - Average True Range
- `compute_rsi_fast()` - RSI
- `compute_macd_fast()` - MACD
- `compute_bollinger_fast()` - Bollinger Bands
- `run_backtest()` - Backtest complet
- `run_native_search()` - Grid search parallèle

### Benchmarks

| Opération | Python | Rust | Speedup |
|-----------|--------|------|---------|
| ATR 14 (10k bars) | 45ms | 1.2ms | **37x** |
| Single backtest | 450ms | 12ms | **37x** |
| Grid search (1000) | 7.5min | 8sec | **56x** |

---

## Déploiement

### VPS
- **IP**: 89.168.48.147
- **OS**: Ubuntu 22.04
- **Domain**: hyprlcore.com

### Services systemd
```bash
hyprl-aggressive.service
hyprl-normal.service
hyprl-mix.service
hyprl-mt5-api.service
```

### API MT5
```
https://hyprlcore.com/mt5-api/health
https://hyprlcore.com/mt5-api/signals?key=<KEY>
```

---

## Classes Principales

### Core Strategy
```python
# src/hyprl/strategy/core.py
class CoreStrategy:
    def generate_signal(symbol, features) -> Signal
    def execute_trade(signal) -> Order
```

### Position Tracking
```python
# src/hyprl/strategy/position_aware.py
@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    entry_sentiment: float
    current_price: float
    current_sentiment: float

    @property
    def unrealized_pnl_pct(self) -> float
    @property
    def sentiment_delta(self) -> float
    @property
    def duration_hours(self) -> float

class PositionAwareTrading:
    def on_trade_entry()
    def evaluate_position() -> ExitSignal
```

### Multi-Source Sentiment
```python
# src/hyprl/sentiment/multi_source.py
@dataclass
class SentimentSignal:
    source: str
    score: float      # -1 to +1
    confidence: float # 0 to 1
    reliability: float

@dataclass
class AggregatedSentiment:
    final_score: float
    final_confidence: float
    consensus: str  # "bullish", "bearish", "mixed"
    sources: list[SentimentSignal]

class MultiSourceSentiment:
    def get_sentiment(symbol) -> AggregatedSentiment
```

### Smart Exit Monitor
```python
# src/hyprl/strategy/smart_exit_monitor.py
class SmartExitMonitor:
    def sync_positions_from_alpaca()
    def update_sentiment()
    def evaluate_and_alert() -> list[ExitSignal]
```

---

## Variables Clés

### Seuils de décision
```python
LONG_THRESHOLD = 0.55
SHORT_THRESHOLD = 0.45
MIN_CONFIDENCE = 0.10
```

### Gestion du risque
```python
STOP_LOSS_ATR = 2.5
TAKE_PROFIT_ATR = 5.0
MAX_DAILY_LOSS = 0.10  # 10%
MAX_DRAWDOWN = 0.20    # 20%
```

### Sentiment
```python
BULLISH_THRESHOLD = 0.3
BEARISH_THRESHOLD = -0.3
MIN_SENTIMENT_CONFIDENCE = 0.4
```

---

## Résultats Live

### Paper Trading (Janvier 2026)

| Compte | P/L |
|--------|-----|
| Aggressive | -0.9% |
| Normal | +0.06% |
| Mix | +0.21% |

### Backtest (2024)

| Symbole | Sharpe | Win% | MaxDD |
|---------|--------|------|-------|
| NVDA | 1.42 | 54% | -8.2% |
| MSFT | 1.15 | 52% | -6.8% |
| QQQ | 1.28 | 53% | -7.1% |

---

## Documentation détaillée

| Fichier | Contenu |
|---------|---------|
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | Architecture système |
| [02_MODELS.md](02_MODELS.md) | Modèles ML et hyperparams |
| [03_FEATURES.md](03_FEATURES.md) | Feature engineering |
| [04_STRATEGY_FILTERS.md](04_STRATEGY_FILTERS.md) | Stratégies et filtres |
| [05_RUST_ENGINE.md](05_RUST_ENGINE.md) | Calculateur Rust + benchmarks |
| [06_DEPLOYMENT.md](06_DEPLOYMENT.md) | Déploiement VPS |
| [07_CONFIGS.md](07_CONFIGS.md) | Toutes les configs |
| [08_RESULTS.md](08_RESULTS.md) | Résultats détaillés |
| [09_RISK_PREVENTION.md](09_RISK_PREVENTION.md) | **Prévention risques et catastrophes** |

---

## Prévention des Risques

### Modules de Protection

| Module | Chemin | Fonction |
|--------|--------|----------|
| **Drift Detector** | `src/hyprl/monitoring/drift_detector.py` | Détecte la dégradation du modèle |
| **Circuit Breakers** | `src/hyprl/risk/circuit_breakers.py` | Protection multi-niveaux (-2%, -5%, -10%, -15%) |
| **Kill Switch** | `.kill_switch` | Arrêt d'urgence manuel |
| **Health Monitor** | `src/hyprl/monitoring/health.py` | Surveillance santé système |
| **Data Quality** | `src/hyprl/monitoring/data_quality.py` | Validation données |
| **Regime Detector** | `src/hyprl/regime/classifier.py` | Adaptation au marché |
| **Correlation Monitor** | `src/hyprl/risk/correlation.py` | Risque de portfolio |

### Circuit Breakers

| Niveau | Seuil | Action |
|--------|-------|--------|
| Level 1 | -2% daily | Réduire positions 50% |
| Level 2 | -5% daily | Stop nouvelles positions |
| Level 3 | -10% daily | Fermer tout, halt journée |
| Level 4 | -15% total | Halt 1 semaine |

### Commandes d'urgence

```bash
# Activer kill switch
echo "1" > /home/kyo/HyprL/.kill_switch

# Recovery complet
./scripts/ops/emergency_recovery.sh --full

# Health check
python -c "from src.hyprl.monitoring.health import HealthMonitor; HealthMonitor().print_status()"
```

---

## Commandes utiles

### Lancer le système
```bash
# Signal generator
python scripts/run_live_hour.py --config configs/NVDA-1h_v3.yaml

# Trading bridge
python scripts/execution/run_strategy_bridge.py --strategy aggressive --paper

# Exit monitor
python src/hyprl/strategy/smart_exit_monitor.py --env .env.aggressive --interval 300
```

### Vérifier les positions
```bash
curl "https://hyprlcore.com/mt5-api/signals?key=hyprl_mt5_ftmo_2026"
```

### Logs
```bash
ssh ubuntu@89.168.48.147 "sudo journalctl -u hyprl-aggressive -f"
```

---

## Contact

- **GitHub**: [HyprL repo]
- **Site**: https://hyprlcore.com
- **Dashboard**: https://hyprlcore.com/track-record

---

*Documentation générée - Janvier 2026*
