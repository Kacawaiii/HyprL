# HyprL - Documentation Technique Complète

> **Version:** v0r1 (Production Launch)
> **Dernière mise à jour:** 2026-01-11
> **Statut:** Paper Trading Actif

---

## Table des Matières

1. [Vue d'Ensemble](#1-vue-densemble)
2. [Architecture du Système](#2-architecture-du-système)
3. [Modèles XGBoost v3](#3-modèles-xgboost-v3)
4. [Configuration des Tickers](#4-configuration-des-tickers)
5. [Gestion du Risque](#5-gestion-du-risque)
6. [Régime de Marché](#6-régime-de-marché)
7. [Résultats & Performance](#7-résultats--performance)
8. [Infrastructure VPS](#8-infrastructure-vps)
9. [Intégration Alpaca](#9-intégration-alpaca)
10. [Opérations Quotidiennes](#10-opérations-quotidiennes)
11. [Sécurité](#11-sécurité)
12. [Commandes Utiles](#12-commandes-utiles)
13. [Outils pour Plus de Gains](#13-outils-pour-plus-de-gains)
    - [Options Overlay](#131-options-overlay-covered-calls-income)
    - [Sentiment Filter](#132-sentiment-filter-newssocial)
    - [Crypto Extension](#133-crypto-extension-btceth-247)

---

## 1. Vue d'Ensemble

### Qu'est-ce que HyprL?

HyprL est un système de trading algorithmique utilisant des modèles XGBoost pour générer des signaux probabilistes sur des actions tech US (NVDA, MSFT) et un ETF (QQQ).

### Stack Technologique

| Composant | Technologie |
|-----------|-------------|
| ML/Models | XGBoost, scikit-learn |
| Backend | Python 3.12, Flask |
| Frontend | Streamlit |
| Broker | Alpaca (Paper/Live) |
| Infra | Docker, Caddy, systemd |
| VPS | OVH (Ubuntu 24.04) |
| DNS | hyprlcore.com |

### Flux de Données

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────┐
│   yfinance  │────▶│ Core V3      │────▶│ Alpaca      │────▶│ Orders  │
│   (Data)    │     │ (Signals)    │     │ Bridge      │     │ Filled  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Regime       │
                    │ Classifier   │
                    └──────────────┘
```

---

## 2. Architecture du Système

### Composants Principaux

```
HyprL/
├── models/                    # Modèles XGBoost (.joblib)
├── configs/                   # Configurations YAML par ticker
├── src/hyprl/
│   ├── strategy/              # Logique de décision
│   ├── risk/                  # Kelly, ATR, allocation
│   ├── regime/                # Classification marché
│   └── broker/                # Abstraction Alpaca
├── scripts/
│   ├── execution/             # Bridge Alpaca
│   └── ops/                   # Scripts opérationnels
├── apps/
│   ├── landing/               # Site web statique
│   └── track_record/          # Dashboard Streamlit
├── deploy/
│   ├── systemd/               # Services Linux
│   └── docker-compose.yml     # Stack Docker
└── live/
    ├── logs/                  # Signaux, régime
    ├── execution/             # Orders Alpaca
    └── state/                 # États persistants
```

### Services Docker (Production)

| Service | Image | Port | Rôle |
|---------|-------|------|------|
| hyprl_caddy | caddy:2-alpine | 80, 443 | Reverse proxy + TLS |
| hyprl-track_record | python:3.12-slim | 8501 | Dashboard Streamlit |
| hyprl-status_api | python:3.12-slim | 8080 | API status pour landing |
| hyprl_api | deploy-api:v2-dev | 8000 | API principale |
| hyprl_portal | deploy-portal:v2-dev | 8501 | Portal utilisateur |
| hyprl_redis | redis:7-alpine | 6379 | Cache |

### Services systemd

| Service | Fonction |
|---------|----------|
| `hyprl_alpaca_bridge` | Exécution des ordres Alpaca |
| Cron job horaire | Génération signaux Core V3 |

---

## 3. Modèles XGBoost v3

### Artifacts

| Fichier | Taille | Ticker |
|---------|--------|--------|
| `nvda_1h_xgb_v3.joblib` | 2.3 MB | NVDA |
| `msft_1h_xgb_v3.joblib` | 2.5 MB | MSFT |
| `qqq_1h_xgb_v3.joblib` | 2.3 MB | QQQ |

### Features (19 colonnes)

```python
feature_columns = [
    # Returns
    'ret_1h', 'ret_3h', 'ret_6h', 'ret_24h',

    # ATR (Average True Range)
    'atr_14', 'atr_72', 'atr_14_norm', 'atr_72_norm',

    # RSI (Relative Strength Index)
    'rsi_7', 'rsi_14', 'rsi_21',

    # Volume
    'vol_ratio_10_30', 'vol_regime_high', 'volume_zscore_24', 'volume_surge',

    # Range
    'range_pct', 'true_range',

    # Distribution
    'ret_skew_20', 'ret_kurt_20'
]
```

### Spécifications

| Paramètre | Valeur |
|-----------|--------|
| Type | XGBoost Classifier |
| Calibration | None (probabilités directes) |
| Seed | 42 |
| Intervalle | 1 heure |
| Période données | 60 jours (yfinance) |

### Sortie du Modèle

```python
probability_up = model.predict_proba(features)[0, 1]
# Valeur entre 0.0 et 1.0
# > threshold_long  → LONG
# < threshold_short → SHORT
# entre les deux   → FLAT
```

---

## 4. Configuration des Tickers

### NVDA (Semiconductors)

**Fichier:** `configs/NVDA-1h_v3.yaml`

```yaml
ticker: NVDA
group: semiconductor
interval: 1h
period: 60d

model:
  artifact: models/nvda_1h_xgb_v3.joblib
  type: xgboost
  calibration: none
  seed: 42
  preset: nvda_v2

thresholds:
  long: 0.53       # Entrée long si prob > 53%
  short: 0.45      # Entrée short si prob < 45%

risk:
  risk_pct: 0.01   # 1% du capital par trade
  atr_multiplier: 1.5  # Stop = 1.5 × ATR
  reward_multiple: 1.9
  min_position_size: 5

trailing:
  enabled: true
  stop_activation: 1.0   # Active à +1%
  stop_distance: 0.04    # Trail de 4%

enable_trend_filter: true
trend_long_min: -0.01
trend_short_min: -0.0025

multi_timeframes_enabled: true
multi_timeframes_frames: "15m,1h,4h"
fusion_method: min
fusion_weights:
  15m: 0.2
  1h: 0.6
  4h: 0.2

max_daily_trades: 3
min_bars_between_entries: 6
min_ev_multiple: 0.0
```

### MSFT (Technology)

**Fichier:** `configs/MSFT-1h_v3.yaml`

```yaml
ticker: MSFT
group: tech

thresholds:
  long: 0.63       # Plus conservateur que NVDA
  short: 0.55

risk:
  risk_pct: 0.01
  atr_multiplier: 1.0  # Stop plus serré
  reward_multiple: 1.5

trend_long_min: -0.01
trend_short_min: -0.0025

max_daily_trades: 3
```

### QQQ (ETF Tech)

**Fichier:** `configs/QQQ-1h_v3.yaml`

```yaml
ticker: QQQ
group: etf

thresholds:
  long: 0.60       # Ajusté récemment (était 0.73)
  short: 0.53

risk:
  risk_pct: 0.01
  atr_multiplier: 1.0
  reward_multiple: 1.5

max_daily_trades: 1   # Très conservateur
min_ev_multiple: 0.2  # Filtre EV actif
```

### Comparatif des Seuils

| Ticker | Long | Short | ATR Mult | Daily Trades |
|--------|------|-------|----------|--------------|
| NVDA | 0.53 | 0.45 | 1.5 | 3 |
| MSFT | 0.63 | 0.55 | 1.0 | 3 |
| QQQ | 0.60 | 0.53 | 1.0 | 1 |

---

## 5. Gestion du Risque

### 5.1 ATR-Based Position Sizing

**Formule:**
```
stop_distance = ATR_14 × atr_multiplier
risk_amount = capital × risk_pct
position_size = risk_amount / stop_distance
```

**Exemple (NVDA):**
```
Capital: $100,000
Risk: 1%
ATR_14: $3.50
atr_multiplier: 1.5

stop_distance = $3.50 × 1.5 = $5.25
risk_amount = $100,000 × 0.01 = $1,000
position_size = $1,000 / $5.25 = 190 shares
```

### 5.2 Kelly Criterion

**Module:** `src/hyprl/risk/kelly.py`

```python
# Formule Kelly
kelly_full = (win_rate × payoff_ratio - loss_rate) / payoff_ratio

# Application (Half-Kelly pour prudence)
kelly_half = kelly_full / 2

# Paramètres
lookback_trades = 50
min_trades = 5
max_multiplier = 2.0   # Cap à 2%
min_multiplier = 0.25  # Floor à 0.25%
```

### 5.3 Trailing Stop

```yaml
trailing:
  enabled: true
  stop_activation: 1.0   # Active quand profit >= 1%
  stop_distance: 0.04    # Trail 4% sous le high
```

**Logique:**
1. Prix monte de 1% → trailing activé
2. Nouveau high → stop remonte à (high - 4%)
3. Prix tombe sous stop → exit

### 5.4 Portfolio Risk Limits

```yaml
portfolio_risk:
  max_total_risk_pct: 0.05   # Max 5% risque total
  max_ticker_risk_pct: 0.03  # Max 3% par ticker
  max_group_risk_pct: 0.04   # Max 4% par groupe
  max_positions: 5           # Max 5 positions
```

### 5.5 Dynamic Allocation (Risk Parity)

**Module:** `src/hyprl/risk/dynamic_allocation.py`

```python
# Allocation inverse à la volatilité
weight[ticker] = 1 / volatility[ticker]
weights = normalize(weights)

# Contraintes
min_weight = 0.10  # Min 10% par ticker
max_weight = 0.50  # Max 50% par ticker
target_vol = 0.15  # Target 15% annualisé

# Rééquilibrage
rebalance_days = 5
smoothing_factor = 0.3  # 30% nouveau, 70% ancien
```

---

## 6. Régime de Marché

### Classification

**Module:** `src/hyprl/regime/classifier.py`

| Régime | Volatilité | VIX | Trend 20d |
|--------|------------|-----|-----------|
| CALM | < 15% | < 18 | > -5% |
| VOLATILE | 15-30% | 18-28 | -5% à -10% |
| CRASH | > 30% | > 28 | < -10% |

### Ajustements par Régime

| Régime | Risk Mult | Threshold+ | Max Pos | Tickers Actifs |
|--------|-----------|------------|---------|----------------|
| CALM | 1.0× | +0% | 6 | NVDA, MSFT, QQQ |
| VOLATILE | 0.5× | +5% | 4 | NVDA, MSFT, QQQ |
| CRASH | 0.25× | +10% | 2 | QQQ seulement |

### État Actuel (Live)

```json
{
  "regime": "calm",
  "risk_multiplier": 1.0,
  "threshold_tighten": 0.0,
  "enabled_tickers": ["NVDA", "MSFT", "QQQ"],
  "weights": {
    "NVDA": 0.328,
    "MSFT": 0.336,
    "QQQ": 0.336
  },
  "features": {
    "realized_vol": 0.089,  // ~9% annualisé
    "vix": 8.92,
    "trend_20d": 0.0001     // quasi flat
  }
}
```

---

## 7. Résultats & Performance

### Backtest OOS (Mars-Décembre 2024)

| Métrique | Portfolio | NVDA | MSFT | QQQ |
|----------|-----------|------|------|-----|
| Profit Factor | 7.79 | 5.93 | 2.95 | 2.77 |
| Sharpe Ratio | 17.91 | - | - | - |
| Max Drawdown | 0.82% | - | - | - |
| Win Rate | 76.3% | 79.7% | 74.7% | 74.4% |
| Trades | 390 | 118 | 182 | 168 |

### Paper Trading (Décembre 2024)

| Métrique | Valeur |
|----------|--------|
| Equity | ~$99,811 |
| Total Return | +23% |
| Max Drawdown | -2.5% |
| Trades | 474 |
| Win Rate | 73.4% |
| Profit Factor | 2.23 |

### Distribution des Exits

| Type | % |
|------|---|
| Trailing Stop | 73.2% |
| Stop Loss | 26.6% |
| Time Exit | 0.2% |

### Quality Gates (Tous Passés)

- [x] PF > 1.5 → 3.57
- [x] MaxDD < 5% → 2.62%
- [x] Win% > 60% → 76.3%
- [x] Trailing dominance > 70% → 75%

---

## 8. Infrastructure VPS

### Serveur

| Paramètre | Valeur |
|-----------|--------|
| Provider | OVH |
| IP | 54.36.183.78 |
| OS | Ubuntu 24.04 LTS |
| SSH | `ssh hyprl` |
| Path | /home/ubuntu/HyprL |

### DNS (hyprlcore.com)

| Sous-domaine | Cible | Usage |
|--------------|-------|-------|
| hyprlcore.com | Landing page | Site vitrine |
| www.hyprlcore.com | → hyprlcore.com | Redirect |
| api.hyprlcore.com | hyprl_api:8000 | API REST |
| app.hyprlcore.com | hyprl_portal:8501 | Dashboard |

### Configuration Caddy

```caddyfile
api.hyprlcore.com {
    reverse_proxy hyprl_api:8000
}

app.hyprlcore.com {
    reverse_proxy hyprl_portal:8501
}

hyprlcore.com, www.hyprlcore.com {
    handle /api/status {
        reverse_proxy hyprl-status_api-1:8080
    }

    handle /api/* {
        reverse_proxy hyprl_beta:8090
    }

    handle {
        root * /srv/landing
        file_server
    }
}
```

### Ports & Services

| Port | Service | Accès |
|------|---------|-------|
| 80 | Caddy HTTP | Public |
| 443 | Caddy HTTPS | Public |
| 8080 | Status API | Internal |
| 8501 | Streamlit | Internal |
| 8000 | API | Internal |
| 6379 | Redis | Internal |

### Docker Network

```bash
# Réseau principal
docker network ls
# hyprl_hyprl_net - bridge network pour tous les services

# Connexion d'un container
docker network connect hyprl_hyprl_net <container>
```

---

## 9. Intégration Alpaca

### Configuration API

**Fichier:** `.env.broker.alpaca`

```bash
ALPACA_API_KEY_ID=PKTZNMGD...
ALPACA_API_SECRET_KEY=GoFeWb...
ALPACA_API_KEY=${ALPACA_API_KEY_ID}
ALPACA_SECRET_KEY=${ALPACA_API_SECRET_KEY}
```

### Bridge Service

**Fichier:** `/etc/systemd/system/hyprl_alpaca_bridge.service`

```ini
[Service]
ExecStart=/home/ubuntu/HyprL/.venv/bin/python \
    scripts/execution/run_alpaca_bridge.py \
    --signals live/logs/live_signals.jsonl \
    --out live/execution/alpaca/orders.jsonl \
    --state live/execution/alpaca/state.json \
    --symbols NVDA,MSFT,QQQ \
    --paper \
    --allow-short \
    --max-orders-per-day 20 \
    --max-qty 50 \
    --poll-seconds 60 \
    --kill-switch /tmp/hyprl_kill_switch
```

### Options du Bridge

| Option | Valeur | Description |
|--------|--------|-------------|
| `--paper` | true | Mode paper trading |
| `--allow-short` | true | Shorts autorisés |
| `--max-orders-per-day` | 20 | Limite journalière |
| `--max-qty` | 50 | Max shares par ordre |
| `--poll-seconds` | 60 | Intervalle vérification |
| `--kill-switch` | /tmp/hyprl_kill_switch | Fichier arrêt urgence |

### Arrêt d'Urgence

```bash
# Créer le fichier kill-switch
touch /tmp/hyprl_kill_switch

# Le bridge s'arrête automatiquement

# Pour reprendre
rm /tmp/hyprl_kill_switch
sudo systemctl restart hyprl_alpaca_bridge
```

### API Alpaca - Méthodes Clés

```python
from hyprl.broker.alpaca import AlpacaBroker

broker = AlpacaBroker(paper=True)

# Compte
broker.get_account()        # Equity, buying power
broker.get_balance()        # Cash disponible

# Positions
broker.get_position(symbol) # Position ouverte
broker.list_positions()     # Toutes les positions
broker.close_position(sym)  # Fermer position
broker.close_all_positions()# Liquidation totale

# Orders
broker.submit_order(...)    # Passer ordre
broker.is_market_open()     # Marché ouvert?
```

---

## 10. Opérations Quotidiennes

### Cron Jobs (VPS)

```bash
# Signal generation - toutes les heures pendant marché US
0 14-21 * * 1-5 /home/ubuntu/HyprL/scripts/ops/run_core_v3_with_regime.sh

# Track record snapshot - 22h UTC
0 22 * * 1-5 python scripts/ops/alpaca_track_record_snapshot.py --paper
```

### Commandes de Monitoring

```bash
# Status bridge
sudo systemctl status hyprl_alpaca_bridge

# Logs bridge
sudo journalctl -u hyprl_alpaca_bridge -f

# Derniers signaux
tail -10 live/logs/live_signals.jsonl | jq .

# Dernier régime
tail -1 live/logs/regime_history.jsonl | jq .

# Orders récents
tail -10 live/execution/alpaca/orders.jsonl | jq .

# Containers Docker
docker ps
```

### GitHub Actions

**Workflow:** `.github/workflows/track-record-daily.yml`

- Schedule: 18:00 Europe/Paris (weekdays)
- Secrets requis: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- Artifacts: Snapshots JSON, reports MD

---

## 11. Sécurité

### Secrets Management

| Secret | Location | Jamais committer |
|--------|----------|------------------|
| Alpaca API | `.env.broker.alpaca` | ✓ |
| Discord webhook | `.env.ops` | ✓ |
| Stripe keys | `.env.stripe` | ✓ |

### .gitignore

```gitignore
.env*
*.env
secrets/
live/
*.joblib
```

### Accès SSH

```bash
# Config locale (~/.ssh/config)
Host hyprl
  HostName 54.36.183.78
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
```

### TLS/HTTPS

- Certificats Let's Encrypt automatiques via Caddy
- Renouvellement auto tous les 60 jours
- HTTPS forcé sur tous les domaines

### Firewall (UFW)

```bash
# Ports ouverts
22/tcp    # SSH
80/tcp    # HTTP (redirect)
443/tcp   # HTTPS
```

---

## 12. Commandes Utiles

### Déploiement

```bash
# Sync local → VPS
rsync -avz --exclude='.venv' --exclude='live' \
    /home/kyo/HyprL/ hyprl:/home/ubuntu/HyprL/

# Restart bridge
ssh hyprl 'sudo systemctl restart hyprl_alpaca_bridge'

# Rebuild containers
ssh hyprl 'cd /opt/hyprl && docker compose up -d --build'
```

### Debug

```bash
# Test signal generation
python scripts/ops/run_core_v3_with_regime.sh

# Test bridge (dry-run)
python scripts/execution/run_alpaca_bridge.py \
    --signals live/logs/live_signals.jsonl \
    --dry-run --once

# Vérifier modèle
python -c "import joblib; m=joblib.load('models/nvda_1h_xgb_v3.joblib'); print(m)"
```

### Backtest

```bash
# Run backtest
python scripts/run_backtest_v3.py \
    --config configs/portfolio_core_1h_v3.yaml \
    --start 2024-03-01 \
    --end 2024-12-01

# Quality gates
python scripts/validate_gates.py --results results/backtest_latest.json
```

### Maintenance

```bash
# Logs Docker
docker logs hyprl-status_api-1 --tail 100

# Restart container
docker restart hyprl-track_record-1

# Disk usage
df -h /opt/hyprl

# Cleanup old logs
find live/logs -name "*.jsonl" -mtime +30 -delete
```

---

## 13. Outils pour Plus de Gains

### 13.1 Options Overlay (Covered Calls Income)

**Module:** `src/hyprl/options/income.py`

Génère du revenu additionnel via des options sur les positions existantes.

#### Stratégies Disponibles

| Stratégie | Description | Usage |
|-----------|-------------|-------|
| **Covered Call** | Vendre des calls sur positions longues | Revenu mensuel |
| **Cash-Secured Put** | Vendre des puts pour entrer à prix réduit | Accumulation |
| **Collar** | Protection downside + cap upside | Défensif |

#### Configuration

```python
from hyprl.options.income import OptionsIncomeAnalyzer, IncomeConfig

config = IncomeConfig(
    min_delta_call=0.20,      # Delta min pour calls OTM
    max_delta_call=0.35,      # Delta max
    min_dte=7,                # Min jours avant expiry
    max_dte=45,               # Max jours
    preferred_dte=30,         # Target ~30 DTE (theta optimal)
    min_premium_pct=0.5,      # Min 0.5% premium
    min_annualized_return=12.0,  # Min 12% annualisé
    max_positions_pct=0.50,   # Max 50% portfolio en options
    require_100_shares=True,  # Seulement si 100+ shares
)

analyzer = OptionsIncomeAnalyzer(config)
```

#### Exemple d'Utilisation

```python
# Analyser opportunité covered call
opp = analyzer.analyze_covered_call(
    symbol="NVDA",
    stock_price=130.00,
    shares=200,
    volatility=0.35,
    days_to_expiry=30,
)

if opp and opp.recommendation in ("strong", "moderate"):
    print(f"Strike: ${opp.legs[0].strike}")
    print(f"Premium: ${opp.total_premium:.2f}")
    print(f"Annualized: {opp.annualized_return:.1f}%")
    print(f"Prob Profit: {opp.probability_profit:.0%}")
```

#### Recommandations

| Niveau | Critères |
|--------|----------|
| **strong** | Return ≥20%, Prob OTM ≥70% |
| **moderate** | Return ≥12%, Prob OTM ≥60% |
| **weak** | Return ≥8% |
| **avoid** | Premium trop faible |

---

### 13.2 Sentiment Filter (News/Social)

**Module:** `src/hyprl/sentiment/trading_filter.py`

Filtre et ajuste les signaux de trading basé sur le sentiment des news.

#### Sources de Données

- **Alpaca News API** - News financières en temps réel
- **Yahoo Finance** - Headlines et articles
- **VADER Sentiment** - Analyse NLP du sentiment

#### Classification du Sentiment

| Niveau | Score | Multiplicateur |
|--------|-------|----------------|
| VERY_BEARISH | ≤ -0.5 | 0.0x (bloqué) |
| BEARISH | -0.5 à -0.2 | 0.5x |
| NEUTRAL | -0.2 à +0.2 | 1.0x |
| BULLISH | +0.2 à +0.5 | 1.2x |
| VERY_BULLISH | ≥ +0.5 | 1.5x |

#### Configuration

```python
from hyprl.sentiment.trading_filter import TradingSentimentFilter, SentimentConfig

config = SentimentConfig(
    # Seuils de classification
    very_bearish_threshold=-0.5,
    bearish_threshold=-0.2,
    bullish_threshold=0.2,
    very_bullish_threshold=0.5,

    # Options de filtrage
    min_news_for_signal=2,        # Min 2 news pour signal
    news_lookback_hours=24,       # Fenêtre 24h
    block_on_very_bearish=True,   # Bloquer si très bearish
    reduce_on_sentiment_conflict=True,  # Réduire si conflit
)

filter = TradingSentimentFilter(config)
```

#### Exemple d'Utilisation

```python
# Analyser sentiment
result = filter.analyze("NVDA")
print(f"Sentiment: {result.level.value}")
print(f"Score: {result.score:.2f}")
print(f"News count: {result.news_count}")
print(f"Should trade: {result.should_trade}")

# Filtrer un signal
adj_size, adj_prob, should_trade, reason = filter.filter_signal(
    symbol="NVDA",
    signal_direction="long",
    base_size=100,
    base_probability=0.65,
)
```

#### Logique de Filtrage

1. **Signal LONG + Sentiment BEARISH** → Size × 0.5
2. **Signal SHORT + Sentiment BULLISH** → Size × 0.5
3. **Sentiment VERY_BEARISH** → Trade bloqué
4. **Sentiment aligné** → Size × multiplicateur

---

### 13.3 Crypto Extension (BTC/ETH 24/7)

**Module:** `src/hyprl/crypto/`

Trading 24/7 pour Bitcoin et Ethereum via Alpaca Crypto API avec modèles XGBoost.

#### Architecture

```
scripts/train_crypto_xgb.py     → Entraînement modèles
scripts/run_crypto_signals.py   → Génération signaux
scripts/ops/run_crypto_signals.sh → Wrapper systemd
src/hyprl/crypto/
├── trader.py                   → Client Alpaca, orders
├── signals.py                  → Features, ML inference
└── __init__.py                 → Exports
models/crypto/
├── btc_usd_xgb.joblib         → Modèle BTC
├── eth_usd_xgb.joblib         → Modèle ETH
└── training_report.json        → Métriques
```

#### Modèles XGBoost Crypto

**Entraînement:** `python scripts/train_crypto_xgb.py --symbols BTC/USD ETH/USD`

| Modèle | Train Acc | Test Acc | Samples | Features |
|--------|-----------|----------|---------|----------|
| BTC/USD | 79.4% | **54.8%** | 8,687 | 27 |
| ETH/USD | 79.1% | **58.0%** | 8,687 | 27 |

**Comparaison Stocks vs Crypto:**

| Asset | Test Accuracy | Volatilité | Difficulté |
|-------|---------------|------------|------------|
| NVDA (1h) | 63% | Moyenne | Facile |
| MSFT (1h) | 68% | Basse | Facile |
| QQQ (1h) | 65% | Basse | Facile |
| **BTC (1h)** | 54.8% | Haute | Difficile |
| **ETH (1h)** | 58.0% | Haute | Moyen |

> Note: Crypto est plus difficile à prédire (volatilité, 24/7, moins de patterns réguliers)

#### Hyperparamètres XGBoost

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,      # Régularisation L2 forte
    reg_alpha=0.5,       # Régularisation L1
    min_child_weight=5,  # Évite overfitting
    random_state=42,
)
```

#### Features (27 colonnes)

```python
FEATURE_COLUMNS = [
    # Returns (5)
    'ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'ret_72h',

    # Volatilité (3)
    'volatility_12h', 'volatility_24h', 'volatility_72h',

    # RSI (3)
    'rsi_6', 'rsi_14', 'rsi_21',

    # SMA Ratios (3)
    'sma_ratio_12', 'sma_ratio_24', 'sma_ratio_72',

    # Volume (3)
    'volume_ratio_12', 'volume_ratio_24', 'volume_zscore',

    # ATR/Range (4)
    'atr_14', 'atr_14_norm', 'high_low_range', 'true_range',

    # Temporel (6) - spécifique crypto 24/7
    'hour_of_day', 'day_of_week', 'is_weekend',
    'is_asia_session', 'is_europe_session', 'is_us_session',
]
```

#### Configuration Trading

```python
CryptoConfig(
    symbols=["BTC/USD", "ETH/USD"],
    timeframe="1Hour",

    # Seuils ML
    threshold_long=0.58,    # prob > 58% → LONG
    threshold_short=0.42,   # prob < 42% → SHORT (désactivé)

    # Risk management
    max_position_pct=0.10,      # Max 10% par crypto
    max_crypto_allocation=0.20, # Max 20% du portfolio total
    stop_loss_pct=0.03,         # 3% stop loss
    take_profit_pct=0.06,       # 6% take profit
)
```

#### Ordres Crypto (Notional)

Le bridge calcule un **notional** cible puis convertit en `qty` (min $10).

```python
# Bridge: notional sizing -> qty
notional = max(min_notional, equity * size_pct)  # min_notional=10 par defaut
qty = notional / price
broker.submit_order(
    symbol="ETH/USD",
    qty=qty,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,  # 24/7
)
```

#### Positions Live (2026-01-11)

| Symbol | Qty | Entry | Value | P&L |
|--------|-----|-------|-------|-----|
| BTCUSD | 0.00116 | $90,440 | ~$105 | +0.2% |
| ETHUSD | 0.079 | $3,103 | ~$244 | +0.0% |
| **Total Crypto** | - | - | **~$349** | - |

#### Service systemd (24/7)

**Timer:** `hyprl-crypto.timer` - toutes les heures

```bash
# Status
sudo systemctl status hyprl-crypto.timer

# Logs
sudo journalctl -u hyprl-crypto -f

# Test manuel
/home/ubuntu/HyprL/scripts/ops/run_crypto_signals.sh
```

#### Bridge Crypto Mode

```bash
python scripts/execution/run_alpaca_bridge.py \
    --enable-crypto \
    --signals live/logs/crypto_signals.jsonl \
    --out live/execution/crypto/orders.jsonl \
    --state live/execution/crypto/state.json \
    --paper \
    --max-signal-age-seconds 300
```

**Flags spécifiques crypto:**
- `--enable-crypto` : Active le mode crypto
- Pas de `--allow-short` (shorts crypto désactivés)
- `TimeInForce.GTC` automatique (pas de market hours)
- Sizing notional via `size_pct` (min $10)

#### Sortie Exemple

```
==================================================
Crypto Scan - 2026-01-11 11:24 UTC
==================================================

LONG SIGNALS:
  BTC/USD  prob=0.63 size=8.6% @ $90,375.55

SHORT SIGNALS:
  ETH/USD  prob=0.31 size=10.0% @ $3,101.42  (désactivé)

NEUTRAL:
  (aucun)
```

---

### 13.4 Intégration avec le Bridge

Ces outils peuvent être intégrés au bridge principal:

```python
# Dans run_alpaca_bridge.py

# 1. Sentiment filter sur chaque signal
from hyprl.sentiment.trading_filter import TradingSentimentFilter
sentiment_filter = TradingSentimentFilter()

for signal in pending_signals:
    adj_size, _, should_trade, reason = sentiment_filter.filter_signal(
        symbol=signal.symbol,
        signal_direction=signal.direction,
        base_size=signal.qty,
        base_probability=signal.probability,
    )
    if not should_trade:
        log(f"Signal blocked: {reason}")
        continue
    signal.qty = adj_size

# 2. Options overlay sur positions existantes
from hyprl.options.income import OptionsIncomeAnalyzer
options_analyzer = OptionsIncomeAnalyzer()

for position in broker.list_positions():
    if position.qty >= 100:
        opp = options_analyzer.analyze_covered_call(
            symbol=position.symbol,
            stock_price=position.current_price,
            shares=int(position.qty),
            volatility=0.30,
        )
        if opp and opp.recommendation == "strong":
            log(f"Covered call opportunity: {position.symbol}")
```

---

## Annexes

### A. Fichiers de Configuration Clés

| Fichier | Usage |
|---------|-------|
| `configs/NVDA-1h_v3.yaml` | Config NVDA |
| `configs/MSFT-1h_v3.yaml` | Config MSFT |
| `configs/QQQ-1h_v3.yaml` | Config QQQ |
| `configs/portfolio_core_1h_v3.yaml` | Portfolio weights |
| `deploy/systemd/hyprl_alpaca_bridge.service` | Service bridge |
| `docker-compose.yml` | Stack Docker |
| `.env.broker.alpaca` | Credentials Alpaca |

### B. Contacts & Support

- **Repo:** github.com/[private]/HyprL
- **VPS:** 54.36.183.78
- **Domain:** hyprlcore.com

### C. Versions

| Composant | Version |
|-----------|---------|
| Python | 3.12 |
| XGBoost | Latest |
| Streamlit | 1.40+ |
| Docker | 24+ |
| Caddy | 2 |
| alpaca-py | Latest |

---

*Document généré automatiquement - 2025-01-01*
