# HyprL — Document Complet du Projet
**Version:** 1.0 Final  
**Date:** 2025-12-31  
**Status:** Production (Paper Trading Live)  
**URL:** https://hyprlcore.com

---

# 📋 TABLE DES MATIÈRES

1. [Executive Summary](#1-executive-summary)
2. [Proposition de Valeur](#2-proposition-de-valeur)
3. [Performance & Track Record](#3-performance--track-record)
4. [Architecture Technique](#4-architecture-technique)
5. [Infrastructure Déployée](#5-infrastructure-déployée)
6. [API Reference](#6-api-reference)
7. [Modèles ML](#7-modèles-ml)
8. [Risk Management](#8-risk-management)
9. [État Actuel](#9-état-actuel)
10. [Fichiers Clés](#10-fichiers-clés)
11. [Commandes Opérationnelles](#11-commandes-opérationnelles)
12. [Business Model](#12-business-model)
13. [Roadmap](#13-roadmap)
14. [Troubleshooting](#14-troubleshooting)
15. [Contacts & Liens](#15-contacts--liens)

---

# 1. EXECUTIVE SUMMARY

## Qu'est-ce que HyprL ?

**HyprL** est un système de trading algorithmique SaaS qui génère des signaux d'achat/vente sur actions US (NVDA, MSFT, QQQ) avec une **transparence totale** — chaque trade est loggé, auditable, et les métriques sont publiques.

## Chiffres Clés

| Métrique | Valeur |
|----------|--------|
| **Profit Factor (OOS)** | 7.79 |
| **Sharpe Ratio (OOS)** | 17.91 |
| **Max Drawdown (OOS)** | -0.82% |
| **Win Rate moyen** | 76% |
| **Trades (9 mois OOS)** | 390 |
| **Capital Paper Trading** | ~$100,000 |

## Différenciateur

> **"No hidden results. No cherry-picking. Every trade logged."**

Contrairement aux 99% des services de signaux Telegram/Discord qui:
- Cachent leurs pertes
- N'ont pas de track record vérifiable
- Utilisent des "black boxes" opaques

HyprL offre:
- ✅ Track record public en temps réel
- ✅ Méthodologie ML documentée (XGBoost, 19 features)
- ✅ Audit trail SHA256
- ✅ Paper trading live avec widget temps réel

---

# 2. PROPOSITION DE VALEUR

## Problèmes du Marché

| Problème | Impact |
|----------|--------|
| 🙈 **Hidden Losses** | Services ne montrent que les winners |
| 🎰 **No Risk Management** | Positions 100x sans stops |
| 🔒 **Black Boxes** | Aucune explication de méthode |
| ⏰ **Signal Lag** | Exécution manuelle = slippage |

## Solution HyprL

| Feature | Bénéfice |
|---------|----------|
| 📊 **Live Track Record** | Dashboard public, équity en temps réel |
| 🧠 **ML Transparent** | XGBoost, 19 features documentées |
| 🛡️ **Risk-First** | Kelly sizing, ATR stops, trailing |
| ⚡ **Exécution Auto** | Bridge Alpaca, signaux toutes les heures |
| 🔐 **Audit Trail** | Logs JSONL, hash SHA256 |

---

# 3. PERFORMANCE & TRACK RECORD

## 3.1 Out-of-Sample (Mars 2024 — Décembre 2024)

> **Données jamais vues pendant l'entraînement. Pas de curve-fitting.**

| Métrique | Valeur |
|----------|--------|
| **Profit Factor** | 7.79 |
| **Sharpe Ratio** | 17.91 |
| **Max Drawdown** | -0.82% |
| **Trades** | 390 |
| **Période** | 9 mois |

### Stabilité Trimestrielle

| Quarter | Profit Factor | Sharpe | Max DD | Trades |
|---------|---------------|--------|--------|--------|
| Q1 2024 | 3.18 | 17.46 | -0.60% | 29 |
| Q2 2024 | 8.20 | 20.64 | -0.67% | 155 |
| Q3 2024 | 7.70 | 21.81 | -0.58% | 155 |
| Q4 2024 | 6.49 | 21.73 | -0.96% | 162 |

## 3.2 Full Period (2+ ans de données)

| Métrique | Valeur |
|----------|--------|
| **Profit Factor** | 3.01 |
| **Sharpe Ratio** | 7.32 |
| **Max Drawdown** | -4.53% |
| **Trades** | 1,582 |
| **Coûts inclus** | 0.1% round-trip (0.05% commission + 0.05% slippage) |

## 3.3 Performance par Ticker

| Ticker | Profit Factor | Win Rate | Allocation | Threshold Long/Short |
|--------|---------------|----------|------------|----------------------|
| **NVDA** | 5.88 | 82% | 40% | 0.53 / 0.45 |
| **MSFT** | 3.24 | 75% | 30% | 0.63 / 0.55 |
| **QQQ** | 2.41 | 71% | 30% | 0.73 / 0.53 |

## 3.4 Paper Trading Live (Actuel)

```
Date: 2025-12-31
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Equity:         $99,834.72
Today P/L:      -$165.28 (-0.17%)
Position:       96 QQQ LONG
Avg Entry:      $620.30
Unrealized:     -$120.48
Last Signal:    NVDA LONG @ 2025-12-26 15:45
Status:         SYNCED ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# 4. ARCHITECTURE TECHNIQUE

## 4.1 Stack Technologique

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
├─────────────────────────────────────────────────────────────┤
│  Landing Page (HTML/CSS/JS)    │    Dashboard (Streamlit)   │
│  https://hyprlcore.com         │    https://app.hyprlcore.com│
│  - Widget live status          │    - Equity curve          │
│  - Beta signup form            │    - Trade history         │
│  - Performance metrics         │    - Analytics             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      REVERSE PROXY                           │
├─────────────────────────────────────────────────────────────┤
│                    Caddy (TLS auto)                          │
│  hyprlcore.com → Landing + /api/* → Beta Service             │
│  app.hyprlcore.com → Portal (Streamlit)                      │
│  api.hyprlcore.com → API principale                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        BACKEND                               │
├─────────────────────────────────────────────────────────────┤
│  Beta Service (FastAPI :8090)                                │
│  - /api/status    → Live Alpaca data                         │
│  - /api/sync      → Reconciliation check                     │
│  - /api/sync/fix  → Force sync                               │
│  - /api/beta-signup → Waitlist                               │
├─────────────────────────────────────────────────────────────┤
│  Main API (FastAPI :8000)                                    │
│  - Endpoints techniques                                      │
│  - Backtest on-demand                                        │
├─────────────────────────────────────────────────────────────┤
│  Signal Generator (Cron)                                     │
│  - Toutes les heures 14-21 UTC (Lun-Ven)                    │
│  - XGBoost inference → JSONL signals                         │
├─────────────────────────────────────────────────────────────┤
│  Alpaca Bridge (Systemd)                                     │
│  - Lit signals JSONL                                         │
│  - Exécute via Alpaca Paper API                              │
│  - Log orders.jsonl                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        BROKER                                │
├─────────────────────────────────────────────────────────────┤
│                 Alpaca Paper Trading API                     │
│  - Account management                                        │
│  - Order execution                                           │
│  - Position tracking                                         │
│  - Market data                                               │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 Flow de Signal

```
1. CRON (chaque heure 14-21 UTC)
   │
   ▼
2. Signal Generator
   - Fetch OHLCV (yfinance)
   - Calculate 19 features
   - XGBoost predict → probability
   - Compare vs threshold
   │
   ▼
3. Signal JSONL
   {timestamp, symbol, decision, probability, size, ...}
   │
   ▼
4. Alpaca Bridge
   - Read new signals
   - Check limits (daily orders, notional)
   - Submit to Alpaca
   │
   ▼
5. Alpaca API
   - Execute market order
   - Return fill
   │
   ▼
6. Logs
   - orders.jsonl (execution log)
   - state.json (positions + equity)
```

## 4.3 Features ML (19 total)

```python
FEATURES = {
    # Momentum (4)
    "ret_1h":       "Return 1 hour",
    "ret_3h":       "Return 3 hours",
    "ret_6h":       "Return 6 hours",
    "ret_24h":      "Return 24 hours",

    # Volatility (6)
    "atr_14":       "ATR 14 periods",
    "atr_72":       "ATR 72 periods",
    "atr_14_norm":  "ATR 14 normalized",
    "atr_72_norm":  "ATR 72 normalized",
    "range_pct":    "High-Low range %",
    "true_range":   "True range",

    # Oscillators (3)
    "rsi_7":        "RSI 7 periods",
    "rsi_14":       "RSI 14 periods",
    "rsi_21":       "RSI 21 periods",

    # Volume (4)
    "vol_ratio":    "Volume ratio 10/30",
    "vol_regime":   "Volume regime (high/low)",
    "volume_zscore":"Volume z-score 24h",
    "volume_surge": "Volume surge detector",

    # Distribution (2)
    "ret_skew_20":  "Return skewness 20 periods",
    "ret_kurt_20":  "Return kurtosis 20 periods"
}
```

---

# 5. INFRASTRUCTURE DÉPLOYÉE

## 5.1 Serveur

| Attribut | Valeur |
|----------|--------|
| **Provider** | OVH VPS |
| **IP** | 54.36.183.78 |
| **OS** | Ubuntu 22.04 |
| **Docker** | 24.x + Compose |
| **SSL** | Auto (Caddy) |

## 5.2 Services Docker

```yaml
services:
  hyprl_caddy:      # Reverse proxy + TLS
    ports: 80, 443
    status: ✅ Running

  hyprl_api:        # API principale
    port: 8000 (internal)
    status: ✅ Running

  hyprl_portal:     # Dashboard Streamlit
    port: 8501 (internal)
    status: ✅ Running

  hyprl_beta:       # Beta service (status, sync, signup)
    port: 8090
    status: ✅ Running

  hyprl_bridge:     # Alpaca execution (systemd)
    status: ✅ Running
```

## 5.3 Domaines & Routing

| URL | Destination | Description |
|-----|-------------|-------------|
| `hyprlcore.com` | Landing HTML | Page marketing |
| `hyprlcore.com/api/*` | Beta Service :8090 | Status, sync, signup |
| `app.hyprlcore.com` | Portal :8501 | Dashboard Streamlit |
| `api.hyprlcore.com` | API :8000 | Endpoints techniques |

## 5.4 Cron Jobs

```cron
# Signal generation - Lun-Ven 14-21 UTC (heures marché US)
0 14-21 * * 1-5 /opt/hyprl/.venv/bin/python /opt/hyprl/scripts/ops/run_core_v3_hourly_batch.py >> /opt/hyprl/live/logs/cron.log 2>&1
```

## 5.5 Caddyfile

```caddyfile
api.hyprlcore.com {
    reverse_proxy hyprl_api:8000
}

app.hyprlcore.com {
    reverse_proxy hyprl_portal:8501
}

hyprlcore.com, www.hyprlcore.com {
    handle /api/* {
        reverse_proxy hyprl_beta:8090
    }
    handle {
        root * /srv/landing
        file_server
    }
}
```

---

# 6. API REFERENCE

## 6.1 Endpoints Publics

### GET /api/status
Widget live - données Alpaca temps réel.

**Response:**
```json
{
    "status": "live",
    "account_value": 99834.72,
    "today_pnl": -165.28,
    "today_pnl_pct": -0.165,
    "open_positions": 1,
    "positions": [
        {
            "symbol": "QQQ",
            "qty": 96,
            "side": "long",
            "pnl": -120.48,
            "pnl_pct": -0.20
        }
    ],
    "last_signal": "NVDA LONG (prob: 75.0%) @ 2025-12-26 15:45",
    "timestamp": "2025-12-31T00:30:00.000Z",
    "source": "alpaca_live"
}
```

### POST /api/beta-signup
Inscription waitlist beta.

**Body (FormData):**
```
name: string
email: string
experience: beginner|intermediate|advanced|professional
capital: <5k|5k-20k|20k-50k|50k-100k|100k+
interest_dashboard: yes|no
interest_signals: yes|no
interest_api: yes|no
interest_automation: yes|no
```

## 6.2 Endpoints Admin (secret requis)

### GET /api/sync?secret=<admin_secret>
Vérifie la synchronisation Alpaca ↔ State interne.

**Response (synced):**
```json
{
    "timestamp": "2025-12-31T00:23:04.689Z",
    "status": "synced",
    "alpaca_equity": 99834.72,
    "internal_equity": 99834.72,
    "equity_diff": 0.0,
    "equity_diff_pct": 0.0,
    "alpaca_positions": {"QQQ": 96.0},
    "internal_positions": {"QQQ": {...}},
    "position_mismatches": [],
    "alert": false
}
```

**Response (desync):**
```json
{
    "status": "DESYNC",
    "alert": true,
    "position_mismatches": [
        {"symbol": "QQQ", "alpaca_qty": 96, "internal_qty": 0}
    ]
}
```

### POST /api/sync/fix?secret=<admin_secret>
Force la synchronisation Alpaca → State interne.

**Response:**
```json
{
    "status": "synced",
    "equity": 99834.72,
    "positions": {
        "QQQ": {
            "qty": 96.0,
            "side": "long",
            "avg_entry": 620.298,
            "market_value": 59379.84
        }
    },
    "synced_at": "2025-12-31T00:22:00.641Z"
}
```

### GET /api/sync/history?secret=<admin_secret>&limit=50
Historique des reconciliations.

---

# 7. MODÈLES ML

## 7.1 Architecture Modèle

| Attribut | Valeur |
|----------|--------|
| **Type** | XGBoost Classifier |
| **Training** | Walk-forward 1 an hourly |
| **Features** | 19 (voir section 4.3) |
| **Output** | Probabilité directionnelle [0, 1] |
| **Calibration** | None (raw probability) |

## 7.2 Paramètres par Ticker

| Ticker | Threshold Long | Threshold Short | Max Daily Trades | Min Bars Between |
|--------|----------------|-----------------|------------------|------------------|
| NVDA | 0.53 | 0.45 | - | - |
| MSFT | 0.63 | 0.55 | 3 | 6 |
| QQQ | 0.73 | 0.53 | 1 | 6 |

## 7.3 Artifacts

```
/opt/hyprl/models/
├── nvda_1h_xgb_v3.joblib      # Modèle NVDA
├── nvda_1h_xgb_v3_features.json
├── msft_1h_xgb_v3.joblib      # Modèle MSFT
├── msft_1h_xgb_v3_features.json
├── qqq_1h_xgb_v3.joblib       # Modèle QQQ
└── qqq_1h_xgb_v3_features.json
```

## 7.4 Configs

```yaml
# /opt/hyprl/configs/NVDA-1h_v3.yaml
ticker: NVDA
interval: 1h
model_artifact: models/nvda_1h_xgb_v3.joblib
feature_preset: nvda_v2

thresholds:
  long: 0.53
  short: 0.45

risk:
  risk_pct: 0.01
  atr_multiplier: 1.0
  reward_multiple: 1.5

trailing:
  enabled: true
  stop_activation: 1.0
  stop_distance: 0.04
```

---

# 8. RISK MANAGEMENT

## 8.1 Position Sizing

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **Base Risk** | 1% | Risque par trade |
| **Sizing Method** | Kelly-based | Ajusté selon win rate |
| **Min Position** | 5 shares | Minimum exécutable |
| **Max Leverage** | 5x | Cap de sécurité |

## 8.2 Stop-Loss

| Paramètre | Valeur |
|-----------|--------|
| **Méthode** | ATR-based |
| **Multiplicateur** | 1.5 × ATR |
| **Trailing Activation** | +1.0R |
| **Trailing Distance** | 0.04R |

## 8.3 Limites Bridge

| Limite | Valeur | Description |
|--------|--------|-------------|
| **Max Orders/Day** | 50 | Ordres par jour |
| **Max Notional/Day** | $10,000 | Exposition max |
| **Kill Switch** | Configurable | Arrêt d'urgence |

## 8.4 Reconciliation

- **Check automatique** : `/api/sync` vérifie Alpaca vs interne
- **Seuil d'alerte** : >1% equity diff OU position mismatch
- **Fix manuel** : `/api/sync/fix` force la sync
- **Log** : Historique dans `reconciliation_log.json`

---

# 9. ÉTAT ACTUEL

## 9.1 Ce qui fonctionne ✅

| Composant | Status | Notes |
|-----------|--------|-------|
| Landing page | ✅ Live | https://hyprlcore.com |
| Widget live status | ✅ Live | Données Alpaca temps réel |
| API /status | ✅ Live | Public |
| API /sync | ✅ Live | Admin |
| API /sync/fix | ✅ Live | Admin |
| Beta signup | ✅ Live | FormData → JSON |
| Dashboard | ✅ Live | https://app.hyprlcore.com |
| Signal generator | ✅ Active | Cron 14-21 UTC |
| Alpaca bridge | ✅ Active | Paper trading |
| Paper trading | ✅ Active | ~$100k equity |

## 9.2 À faire (non critique)

| Item | Priorité | Effort |
|------|----------|--------|
| Auto-sync cron | Low | 30 min |
| Discord bot | Low | 2-4h |
| Email alerts | Medium | 2-4h |
| Healthchecks Docker | Low | 1h |

## 9.3 Métriques Live

```
Last Update: 2025-12-31 00:30 UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Account Equity:    $99,834.72
Cash Available:    $40,454.88
Open Positions:    1 (96 QQQ LONG)
Today's P/L:       -$165.28 (-0.17%)
Sync Status:       ✅ SYNCED
Last Signal:       NVDA LONG @ 2025-12-26
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

# 10. FICHIERS CLÉS

## 10.1 Configuration

```
/opt/hyprl/
├── .env.broker.alpaca          # Credentials Alpaca
├── docker-compose.yml          # Services Docker
├── Caddyfile                   # Reverse proxy
└── configs/
    ├── NVDA-1h_v3.yaml        # Config NVDA
    ├── MSFT-1h_v3.yaml        # Config MSFT
    └── QQQ-1h_v3.yaml         # Config QQQ
```

## 10.2 API & Services

```
/opt/hyprl/scripts/api/
├── beta_service.py            # FastAPI (status, sync, signup)
└── ...

/opt/hyprl/scripts/execution/
├── run_alpaca_bridge.py       # Bridge Alpaca
└── ...

/opt/hyprl/scripts/ops/
├── run_core_v3_hourly_batch.py # Signal generator
└── ...
```

## 10.3 State & Logs

```
/opt/hyprl/live/
├── execution/alpaca/
│   ├── state.json             # État interne (equity, positions)
│   └── orders.jsonl           # Log des ordres exécutés
├── logs/
│   ├── live_signals.jsonl     # Log des signaux générés
│   └── cron.log               # Log du cron
└── ...
```

## 10.4 Frontend

```
/opt/hyprl/apps/landing/
├── index.html                 # Landing page (854 lignes)
├── styles.css                 # Styles (1398 lignes)
├── bg.js                      # Animation canvas
└── thanks.html                # Page de confirmation
```

## 10.5 Data

```
/opt/hyprl/data/
├── beta_signups.json          # Inscriptions beta
├── reconciliation_log.json    # Historique sync
└── cache/                     # Cache prix yfinance
```

---

# 11. COMMANDES OPÉRATIONNELLES

## 11.1 SSH & Accès

```bash
# Connexion
ssh ubuntu@54.36.183.78

# Aller dans le projet
cd /opt/hyprl
```

## 11.2 Docker

```bash
# Status des services
docker compose ps

# Logs en temps réel
docker compose logs -f hyprl_beta

# Logs d'un service
docker compose logs --tail=50 hyprl_beta

# Restart un service
docker compose restart hyprl_beta

# Restart Caddy (après modif Caddyfile)
docker compose exec hyprl_caddy caddy reload --config /etc/caddy/Caddyfile
```

## 11.3 API Tests

```bash
# Status public
curl -s https://hyprlcore.com/api/status | python3 -m json.tool

# Check sync (admin)
curl -s "https://hyprlcore.com/api/sync?secret=<admin_secret>" | python3 -m json.tool

# Force sync (admin)
curl -s -X POST "https://hyprlcore.com/api/sync/fix?secret=<admin_secret>" | python3 -m json.tool

# Test direct (bypass Caddy)
curl -s http://127.0.0.1:8090/api/status | python3 -m json.tool
```

## 11.4 Logs & Debug

```bash
# Signaux récents
tail -20 /opt/hyprl/live/logs/live_signals.jsonl

# Ordres exécutés
tail -20 /opt/hyprl/live/execution/alpaca/orders.jsonl

# State actuel
cat /opt/hyprl/live/execution/alpaca/state.json | python3 -m json.tool

# Cron logs
tail -50 /opt/hyprl/live/logs/cron.log

# Beta signups
cat /opt/hyprl/data/beta_signups.json | python3 -m json.tool
```

## 11.5 Maintenance

```bash
# Backup state
cp /opt/hyprl/live/execution/alpaca/state.json ~/backups/state_$(date +%Y%m%d).json

# Vérifier permissions landing
ls -la /opt/hyprl/apps/landing/

# Fix permissions si nécessaire
chmod 644 /opt/hyprl/apps/landing/*

# Vérifier credentials Alpaca
cat /opt/hyprl/.env.broker.alpaca
```

---

# 12. BUSINESS MODEL

## 12.1 Pricing

| Tier | Prix | Features | Timeline |
|------|------|----------|----------|
| **Beta** | €19/mois | Dashboard, exports, Discord, support 48h | Now |
| **Starter** | €29/mois | + Email alerts, reports, support 24h | Q1 2026 |
| **Pro** | €79/mois | + API, webhooks, backtest export, support 12h | Q2 2026 |

## 12.2 Early Bird

- **30 places beta** disponibles
- **First 10** : 50% off forever (€9.50/mois)
- **Grandfathering** : Prix bloqué à vie

## 12.3 Target Market

| Segment | Description | Besoin |
|---------|-------------|--------|
| Retail traders avancés | 3+ ans exp, $20k-100k capital | Signaux fiables, track record |
| Développeurs quant | Veulent automatiser | API, webhooks |
| Crypto refugees | Déçus des scams | Transparence, audit |

## 12.4 Unit Economics (projections)

| Métrique | Valeur |
|----------|--------|
| CAC estimé | €50-100 |
| LTV (12 mois) | €228-468 |
| Churn estimé | 5-10%/mois |
| Break-even | 50-100 clients |

---

# 13. ROADMAP

## Phase 1: Paper Trading Validation ✅ (Actuel)

- [x] Infrastructure déployée
- [x] Signal generator (cron)
- [x] Alpaca bridge (paper)
- [x] Landing page V2
- [x] Widget live status
- [x] API status/sync
- [x] Dashboard Streamlit
- [x] Beta signup

## Phase 2: Beta Launch (Q1 2026)

- [ ] 30 early adopters
- [ ] Email alerts
- [ ] Discord community
- [ ] 3 mois de track record paper
- [ ] Weekly updates

## Phase 3: Public Launch (Q2 2026)

- [ ] Starter tier ($29)
- [ ] Monthly reports
- [ ] Advanced analytics
- [ ] Live trading (micro-capital)

## Phase 4: Scale (Q3 2026)

- [ ] Pro tier ($79)
- [ ] Public API
- [ ] Webhook integrations
- [ ] Custom portfolios
- [ ] 100+ clients

---

# 14. TROUBLESHOOTING

## Widget ne charge pas

```bash
# 1. Vérifier permissions
ls -la /opt/hyprl/apps/landing/
# Doit être 644 pour tous les fichiers

# 2. Fix si nécessaire
chmod 644 /opt/hyprl/apps/landing/*

# 3. Hard refresh navigateur
# Ctrl+Shift+R (ou Cmd+Shift+R sur Mac)

# 4. Vérifier console navigateur (F12)
# Chercher erreurs JS
```

## API retourne 404

```bash
# 1. Vérifier que le service tourne
docker compose ps hyprl_beta

# 2. Tester en direct
curl http://127.0.0.1:8090/api/status

# 3. Vérifier Caddy routing
cat /opt/hyprl/Caddyfile

# 4. Reload Caddy
docker compose exec hyprl_caddy caddy reload --config /etc/caddy/Caddyfile
```

## DESYNC détecté

```bash
# 1. Vérifier l'état
curl -s "https://hyprlcore.com/api/sync?secret=<admin_secret>"

# 2. Force sync
curl -s -X POST "https://hyprlcore.com/api/sync/fix?secret=<admin_secret>"

# 3. Vérifier le state.json
cat /opt/hyprl/live/execution/alpaca/state.json
```

## Alpaca unauthorized

```bash
# 1. Vérifier credentials
cat /opt/hyprl/.env.broker.alpaca

# 2. Régénérer sur https://app.alpaca.markets
# Paper Trading → API Keys → Regenerate

# 3. Mettre à jour .env.broker.alpaca
# ALPACA_API_KEY=<redacted>
# ALPACA_SECRET_KEY=<redacted>

# 4. Restart container
docker compose restart hyprl_beta
```

## Bridge ne s'exécute pas

```bash
# 1. Vérifier le service
systemctl status hyprl-bridge

# 2. Logs
journalctl -u hyprl-bridge -n 50

# 3. Test manuel
/opt/hyprl/.venv/bin/python /opt/hyprl/scripts/execution/run_alpaca_bridge.py \
  --signals /opt/hyprl/live/logs/live_signals.jsonl \
  --once --dry-run
```

---

# 15. CONTACTS & LIENS

## URLs Publiques

| Service | URL |
|---------|-----|
| Landing | https://hyprlcore.com |
| Dashboard | https://app.hyprlcore.com |
| API | https://api.hyprlcore.com |

## Technique

| Resource | Value |
|---------|-------|
| VPS IP | 54.36.183.78 |
| SSH | `ssh ubuntu@54.36.183.78` |
| Project Path | `/opt/hyprl` |

## Contact

| Channel | Value |
|---------|-------|
| Email | contact@hyprl.io |
| Twitter | @HyprLQuant |
| Discord | discord.gg/hyprl |

## Credentials (SÉCURISÉS)

> ⚠️ **Ne jamais committer ces valeurs**

```
ALPACA_API_KEY=<redacted>
ALPACA_SECRET_KEY=<redacted>
ADMIN_SECRET=<redacted>
```

---

# DISCLAIMER

> **Paper Trading Notice**: All performance metrics shown are from paper trading (simulated execution). Real trading results may differ due to slippage, market impact, and execution delays.

> **Risk Warning**: Trading involves substantial risk of loss and is not suitable for all investors. Do not trade with money you cannot afford to lose. Past performance is not indicative of future results.

> **Not Financial Advice**: HyprL is not a registered investment advisor. All signals and information are for educational purposes only. You are solely responsible for your trading decisions.

---

*Document généré le 2025-12-31*  
*Version 1.0 Final*  
*HyprL — Quantitative Trading Intelligence*
