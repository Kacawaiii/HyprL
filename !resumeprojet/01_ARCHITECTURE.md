# Architecture HyprL

## Vue d'ensemble

HyprL est un système de trading algorithmique complet utilisant le Machine Learning pour générer des signaux sur les actions US (NVDA, MSFT, QQQ).

```
┌─────────────────────────────────────────────────────────────────┐
│                         HyprL System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Data    │───▶│ Features │───▶│  Model   │───▶│ Signals  │  │
│  │ Ingestion│    │ Engine   │    │ Predict  │    │ Generator│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                               │        │
│       ▼                                               ▼        │
│  ┌──────────┐                                   ┌──────────┐   │
│  │  Alpaca  │                                   │ Filters  │   │
│  │   API    │                                   │ & Gates  │   │
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
│              │  Alpaca  │  │   MT5    │  │  Track   │        │
│              │  Paper   │  │  Bridge  │  │  Record  │        │
│              └──────────┘  └──────────┘  └──────────┘        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Structure des dossiers

```
HyprL/
├── src/hyprl/                    # Code source principal
│   ├── backtest/                 # Moteur de backtesting
│   ├── broker/                   # Intégrations brokers (Alpaca)
│   ├── features/                 # Feature engineering
│   ├── indicators/               # Indicateurs techniques
│   ├── live/                     # Trading live
│   ├── model/                    # Modèles ML
│   ├── native/                   # Bindings Rust
│   ├── regime/                   # Détection de régime marché
│   ├── risk/                     # Gestion du risque
│   ├── sentiment/                # Analyse de sentiment
│   └── strategy/                 # Stratégies de trading
│
├── scripts/                      # Scripts d'exécution
│   ├── execution/                # Scripts de trading live
│   ├── ops/                      # Scripts opérationnels
│   └── analysis/                 # Scripts d'analyse
│
├── configs/                      # Fichiers de configuration
│   ├── runtime/                  # Configs runtime (VPS)
│   └── *.yaml                    # Configs par symbole
│
├── models/                       # Modèles entraînés (.joblib)
│
├── native/                       # Code Rust (supercalc)
│   └── hyprl_supercalc/
│
├── apps/                         # Applications
│   ├── landing/                  # Site web
│   └── track_record/             # Dashboard Streamlit
│
├── deploy/                       # Déploiement
│   └── systemd/                  # Services systemd
│
├── tests/                        # Tests unitaires
│
├── live/logs/                    # Logs de trading
│
└── data/                         # Données (cache, etc.)
```

## Flux de données

### 1. Collecte des données
```
Alpaca API → OHLCV (1h bars) → Cache Parquet → Feature Engine
```

### 2. Feature Engineering
```
Raw OHLCV → Technical Indicators → Feature Columns → Normalized Features
```

### 3. Prédiction
```
Features → XGBoost/LightGBM/CatBoost → Probability → Signal Decision
```

### 4. Filtrage
```
Raw Signal → Smart Filter → Sentiment Filter → Quality Filter → Final Signal
```

### 5. Exécution
```
Final Signal → Risk Check → Position Sizing → Order → Alpaca/MT5
```

## Composants principaux

| Composant | Fichier | Rôle |
|-----------|---------|------|
| Core Strategy | `src/hyprl/strategy/core.py` | Logique principale de trading |
| Signal Generator | `scripts/run_live_hour.py` | Génère les signaux horaires |
| Trading Bridge | `scripts/execution/run_strategy_bridge.py` | Exécute les trades |
| MT5 API | `scripts/execution/mt5_signal_api.py` | API pour MetaTrader 5 |
| Sentiment | `src/hyprl/sentiment/multi_source.py` | Analyse multi-source |
| Position Monitor | `src/hyprl/strategy/smart_exit_monitor.py` | Monitoring intelligent |

## Environnement de production

- **VPS**: Oracle Cloud (89.168.48.147)
- **OS**: Ubuntu 22.04
- **Python**: 3.12
- **Services**: systemd (hyprl-aggressive, hyprl-normal, hyprl-mix, hyprl-mt5-api)
- **Web**: Nginx + Let's Encrypt (hyprlcore.com)
