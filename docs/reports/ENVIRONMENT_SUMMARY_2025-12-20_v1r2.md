# HyprL — Résumé Complet de l'Environnement
**Date :** 2025-12-20  
**Tag :** `portfolio_core_1h_v3_gate2_oos_v1r2`  
**Branche :** `release/portfolio_core_1h_v3_gate2_oos_v1r2`

## Source of truth
- **Tag :** `portfolio_core_1h_v3_gate2_oos_v1r2`
- **Branche :** `release/portfolio_core_1h_v3_gate2_oos_v1r2`
- **Commit (HEAD) :** `c8e17579d7fb5aecc86da3560453b5437f459dae`
- **Bundle SHA256 :** `docs/reports/core_v3_bundle.sha256`

---

## 1. Identité du Projet

| Attribut | Valeur |
|----------|--------|
| **Nom** | HyprL — core_v3 (portfolio_core_1h_v3) |
| **Type** | Système de trading algorithmique 1h (actions US) |
| **Modèle** | XGBoost (19 features, pas de calibration appliquée en prod (calibration report-only)) |
| **Tickers** | NVDA, MSFT, QQQ (core v3) + META, AMD, SPY (v2 legacy) |
| **Stack** | Python 3.11/3.12 (selon venv), pandas, scikit-learn, xgboost, FastAPI, Streamlit |
| **Accélérateur** | Rust (hyprl_supercalc via PyO3) |

---

## 2. État des Composants

| Composant | Statut | Notes |
|----------|--------|-------|
| **Modèles v3** | ✅ Entraînés | `models/*_1h_xgb_v3.joblib` (NVDA/MSFT/QQQ) |
| **Backtest** | ✅ Fonctionnel | Engine native + Python fallback |
| **Replay/Parity** | ✅ Validé | 0 mismatch BT↔Replay |
| **Live Runner** | ✅ Opérationnel | ~474 trades NVDA (full history) |
| **Portfolio Ops** | ✅ Cron hourly | Multi-ticker orchestration |
| **Monitoring** | ✅ Actif | Discord alerts + health checks |
| **API HyprL** | ✅ Déployable | FastAPI v2 + quotas/auth |
| **SENSE (audit)** | ✅ Audit-only | risk_multiplier=1.0 (neutralisé) |
| **Broker réel** | ⏳ Non connecté | PaperBrokerImpl uniquement |

---

## 3. Performance Validée (Runs du 2025-12-20)

### 3.1 NVDA Replay (full history, parity-mode)
```
Trades           : 474
Profit Factor    : 2.225
Win Rate         : 73.42%
Total PnL        : 330,253.92
Final Balance    : 340,253.92
Exit Mix         : trailing_stop 73.2% | stop_loss 26.6% | time_exit 0.2%
```

### 3.2 Portfolio Core v3 (OOS 2024-03-01 → 2024-12-01)
| Ticker | Trades | PF | Win% | PnL |
|--------|--------|-----|------|-----|
| NVDA | 118 | 5.929 | 79.66% | 22,179.57 |
| MSFT | 182 | 2.953 | 74.73% | 20,532.88 |
| QQQ | 168 | 2.765 | 74.40% | 16,612.67 |
| **Portfolio** | **468** | **3.573** | — | **~59,325** |

**Portfolio Aggregé :**
- PF : 3.57
- MaxDD : 2.62%
- Equity End : 30,015.49 (sur 10k initial)

### 3.3 Calibration NVDA (par bin de probabilité)
| Bin | Count | Mean Prob | Emp Win Rate | Mean Return |
|-----|-------|-----------|--------------|-------------|
| [0.40, 0.50) | 7 | 0.420 | 42.9% | 0.15% |
| [0.50, 0.60) | 15 | 0.552 | 40.0% | 0.00% |
| [0.60, 0.70) | 23 | 0.644 | 39.1% | -0.26% |
| [0.70, 0.80) | 54 | 0.757 | 64.8% | 0.45% |
| [0.80, 0.90) | 90 | 0.859 | 77.8% | 0.86% |
| [0.90, 1.00] | 83 | 0.936 | 95.2% | 1.26% |

**Observation :** Forte calibration au-dessus de 0.70 ; signaux < 0.70 moins fiables.

---

## 4. Architecture Fichiers

### 4.1 Configs (v3 frozen)
```
configs/
├── NVDA-1h_v3.yaml      # long_threshold 0.53, short 0.47
├── MSFT-1h_v3.yaml      # long_threshold 0.63, short 0.55
├── QQQ-1h_v3.yaml       # long_threshold 0.73, short 0.53
└── portfolio_core_1h_v3.yaml  # weights 0.4/0.3/0.3
```

### 4.2 Modèles
```
models/
├── nvda_1h_xgb_v3.joblib
├── msft_1h_xgb_v3.joblib
├── qqq_1h_xgb_v3.joblib
└── *_v3_features.json   # 19 colonnes par modèle
```

### 4.3 Scripts principaux
```
scripts/
├── run_backtest.py                 # Backtest single-ticker
├── run_live_replay.py              # Replay avec parity-mode
├── run_portfolio_replay.py         # Portfolio multi-ticker
├── analyze_trades.py               # Stats/calibration trades
├── repro/repro_core_v3_release.sh  # Intégrité + repro canonique release
└── ops/
    ├── run_live_multi_ticker_hourly.py
    ├── run_portfolio_monitor_live.py
    ├── push_core_v3_discord.py
    └── palier2_hourly.sh
```

### 4.4 Logs & Outputs
```
live/
├── exp/                      # Runs de reproduction (non trackés git)
├── logs/
│   ├── live_signals.jsonl
│   └── live_trades.jsonl
└── sense/                    # SENSE controls (audit)
```

---

## 5. Commandes Validées

```bash
# Backtest NVDA v3
.venv/bin/python scripts/run_backtest.py \
  --config configs/NVDA-1h_v3.yaml \
  --export-trades data/trades_NVDA_1h_v3.csv

# Replay NVDA v3 (parity)
.venv/bin/python scripts/run_live_replay.py \
  --config configs/NVDA-1h_v3.yaml \
  --parity-mode \
  --trade-log live/exp/trades_NVDA_replay_v3.csv

# Portfolio replay (OOS)
.venv/bin/python scripts/run_portfolio_replay.py \
  --configs configs/NVDA-1h_v3.yaml configs/MSFT-1h_v3.yaml configs/QQQ-1h_v3.yaml \
  --weights 0.4 0.3 0.3 \
  --start 2024-03-01 --end 2024-12-01 \
  --out-dir live/exp/trades_portfolio_v3 \
  --summary-out live/exp/trades_portfolio_v3/summary.json

# Analyze trades
.venv/bin/python scripts/analyze_trades.py \
  --trades live/exp/trades_NVDA_replay_v3.csv

# Repro canonique release (intégrité + repro)
./scripts/repro/repro_core_v3_release.sh
```

---

## 6. Gaps & Points d'Attention

### ❌ Scripts introuvables
| Script | Status | Alternative |
|--------|--------|-------------|
| `scripts/run_sense_backfill.py` | NOT FOUND | Localiser via `find . -name "*sense*"` |
| `scripts/exp/make_scorecard.py` | NOT FOUND | Localiser via `find . -name "*scorecard*"` |

### ⚠️ Logs live modifiés (si jamais trackés historiquement)
Si des `.jsonl` sous `live/logs/` sont trackés et bougent, solution clean :
```bash
git rm --cached live/logs/live_signals.jsonl live/logs/live_trades.jsonl
git commit -m "chore: stop tracking live JSONL logs"
```

### ⏳ Non implémenté / En cours
| Élément | État |
|---------|------|
| Broker réel (Alpaca/IBKR) | Non connecté — PaperBroker only |
| CI/CD GitHub Actions | Workflows présents, état non vérifié |
| OpenAPI/Swagger docs | Non visible |
| SENSE production | Neutralisé (audit-only) |

---

## 7. Feature Contract v3 (19 colonnes)

```python
[
  'ret_1h', 'ret_3h', 'ret_6h', 'ret_24h',
  'atr_14', 'atr_72', 'atr_14_norm', 'atr_72_norm',
  'rsi_7', 'rsi_14', 'rsi_21',
  'vol_ratio_10_30', 'vol_regime_high',
  'volume_zscore_24', 'volume_surge',
  'range_pct', 'true_range',
  'ret_skew_20', 'ret_kurt_20'
]
```

---

## 8. Risk/Signal Config (v3 defaults)

| Param | NVDA | MSFT | QQQ |
|-------|------|------|-----|
| long_threshold | 0.53 | 0.63 | 0.73 |
| short_threshold | 0.47 | 0.55 | 0.53 |
| risk_pct | 1% | 1% | 1% |
| trailing_stop_activation | 1.0 | 1.0 | 1.0 |
| trailing_stop_distance | 0.04 | 0.04 | 0.04 |
| max_daily_trades | — | 3 | 1 |
| min_bars_between_entries | — | 6 | 6 |

---

## 9. Déploiement

### Docker
```bash
docker build -f Dockerfile.runtime -t hyprl-runtime .
cd deploy && docker compose -f docker-compose.v2.prod.yml up -d
```

### Env templates
```
deploy/.env.api.prod.example
deploy/.env.portal.prod.example
deploy/.env.bot.prod.example
```

### Reverse proxy
- Caddy : `deploy/Caddyfile.example`
- NGINX : `deploy/nginx.conf.example`

---

## 10. Résumé Exécutif

| Métrique | Valeur |
|----------|--------|
| **Version** | core_v3 (frozen) |
| **PF Portfolio** | 3.57 |
| **MaxDD Portfolio** | 2.62% |
| **Trades Portfolio (OOS)** | 468 |
| **NVDA PF (full)** | 2.23 |
| **NVDA Trades** | 474 |
| **Win Rate moyen** | ~74–80% |
| **Exit dominant** | trailing_stop (~73%) |

**Conclusion :** Le système core_v3 est stable, validé en replay, et prêt pour live paper. L'intégration broker réel reste le principal gap pour passer en production réelle.
