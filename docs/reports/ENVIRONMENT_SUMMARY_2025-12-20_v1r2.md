# HyprL core_v3 — Environment Summary
**Date :** 2025-12-20  
**Tag :** `portfolio_core_1h_v3_gate2_oos_v1r2`  
**Branche :** `release/portfolio_core_1h_v3_gate2_oos_v1r2`

---

## 1. Source of Truth

| Artifact | Path |
|----------|------|
| **Tag** | `portfolio_core_1h_v3_gate2_oos_v1r2` |
| **Bundle SHA** | `docs/reports/core_v3_bundle.sha256` |
| **Factsheet** | `docs/reports/portfolio_core_1h_v3_factsheet_v1r2.md` |
| **Repro script** | `scripts/repro/repro_core_v3_release.sh` |

**Commande unique de reproduction :**
```bash
./scripts/repro/repro_core_v3_release.sh
# Outputs: live/exp/repro_<timestamp>/, hashes in docs/reports/
```

---

## 2. Identité du Projet

| Attribut | Valeur |
|----------|--------|
| **Nom** | HyprL — core_v3 (portfolio_core_1h_v3) |
| **Legacy** | Ascendant v2 (configs/*-1h_v2.yaml) |
| **Type** | Système de trading algorithmique 1h (actions US) |
| **Modèle** | XGBoost (19 features, calibration report-only) |
| **Tickers v3** | NVDA, MSFT, QQQ |
| **Stack** | Python 3.11/3.12 (selon venv), pandas, scikit-learn, xgboost, FastAPI |
| **Accélérateur** | Rust (hyprl_supercalc via PyO3) |

---

## 3. État des Composants

| Composant | Statut | Notes |
|-----------|--------|-------|
| **Modèles v3** | ✅ Entraînés | `models/*_1h_xgb_v3.joblib` (NVDA/MSFT/QQQ) |
| **Backtest** | ✅ Fonctionnel | Engine native + Python fallback |
| **Replay/Parity** | ✅ Validé | 0 mismatch BT↔Replay |
| **Live Runner** | ✅ Opérationnel | ~474 trades NVDA (full history) |
| **Portfolio Ops** | ✅ Cron hourly | Multi-ticker orchestration |
| **Monitoring** | ✅ Actif | Discord alerts + health checks |
| **API HyprL** | ✅ Déployable | FastAPI v2 + quotas/auth |
| **SENSE** | ✅ Audit-only | risk_multiplier=1.0 (neutralisé) |
| **Broker réel** | ⏳ Non connecté | PaperBrokerImpl uniquement |

---

## 4. Performance Validée (2025-12-20)

### 4.1 NVDA Replay (full history, parity-mode)
```
Trades           : 474
Profit Factor    : 2.225
Win Rate         : 73.42%
Total PnL        : 330,253.92
Final Balance    : 340,253.92
Exit Mix         : trailing_stop 73.2% | stop_loss 26.6% | time_exit 0.2%
```

### 4.2 Portfolio Core v3 (OOS 2024-03-01 → 2024-12-01)
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

### 4.3 Calibration Report (NVDA)
> Note : Calibration non appliquée en production — données à titre informatif.

| Bin | Count | Mean Prob | Emp Win Rate | Mean Return |
|-----|-------|-----------|--------------|-------------|
| [0.70, 0.80) | 54 | 0.757 | 64.8% | 0.45% |
| [0.80, 0.90) | 90 | 0.859 | 77.8% | 0.86% |
| [0.90, 1.00] | 83 | 0.936 | 95.2% | 1.26% |

**Observation :** Signaux fiables au-dessus de prob > 0.70.

---

## 5. Architecture Fichiers

### 5.1 Configs (v3 frozen)
```
configs/
├── NVDA-1h_v3.yaml      # long 0.53, short 0.47
├── MSFT-1h_v3.yaml      # long 0.63, short 0.55
├── QQQ-1h_v3.yaml       # long 0.73, short 0.53
└── portfolio_core_1h_v3.yaml  # weights 0.4/0.3/0.3
```

### 5.2 Modèles
```
models/
├── nvda_1h_xgb_v3.joblib
├── msft_1h_xgb_v3.joblib
├── qqq_1h_xgb_v3.joblib
└── *_v3_features.json   # 19 colonnes
```

### 5.3 Scripts principaux
```
scripts/
├── run_backtest.py
├── run_live_replay.py
├── run_portfolio_replay.py
├── analyze_trades.py
├── repro/
│   └── repro_core_v3_release.sh   # ← script canonique
└── ops/
    ├── run_live_multi_ticker_hourly.py
    ├── run_portfolio_monitor_live.py
    └── push_core_v3_discord.py
```

### 5.4 Logs & Outputs
```
live/
├── exp/                      # Runs de reproduction
└── logs/                     # Logs live (non trackés)
    ├── live_signals.jsonl
    └── live_trades.jsonl
```

---

## 6. Commandes Validées

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

# Full reproduction
./scripts/repro/repro_core_v3_release.sh
```

---

## 7. Gaps & Actions

### Scripts hors release
| Script | Status | Action |
|--------|--------|--------|
| `run_sense_backfill.py` | Hors release | SENSE audit-only, non requis |
| `make_scorecard.py` | Hors release | Expérimental, non inclus |

### Logs live (si encore trackés)
```bash
# Vérifier si trackés
git ls-files live/logs | head -5

# Si oui, untrack proprement
git rm --cached live/logs/live_signals.jsonl live/logs/live_trades.jsonl 2>/dev/null || true
echo "live/logs/*.jsonl" >> .gitignore
git add .gitignore
git commit -m "chore: stop tracking live JSONL logs"
```

### Non implémenté
| Élément | État |
|---------|------|
| Broker réel (Alpaca/IBKR) | PaperBroker only |
| CI/CD smoke test | À valider sur tag v1r2 |

---

## 8. Feature Contract v3 (19 colonnes)

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

## 9. Risk/Signal Config (v3)

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

## 10. Résumé Exécutif

| Métrique | Valeur |
|----------|--------|
| **Version** | core_v3 (frozen) |
| **Tag** | `portfolio_core_1h_v3_gate2_oos_v1r2` |
| **PF Portfolio (OOS)** | 3.57 |
| **MaxDD Portfolio** | 2.62% |
| **Trades Portfolio** | 468 |
| **NVDA PF (full)** | 2.23 |
| **NVDA Trades** | 474 |
| **Win Rate moyen** | ~74–80% |
| **Exit dominant** | trailing_stop (~73%) |

---

## 11. Intégration au Repo

### Option recommandée : branche docs (tag immuable)
```bash
git checkout -b docs/env_summary_v1r2 portfolio_core_1h_v3_gate2_oos_v1r2
mkdir -p docs/reports
# copier ce fichier dans docs/reports/ENVIRONMENT_SUMMARY_v1r2.md
git add docs/reports/ENVIRONMENT_SUMMARY_v1r2.md
git commit -m "docs(v1r2): add environment summary snapshot"
git push origin docs/env_summary_v1r2
```

### Option alternative : retagger (si doc doit faire partie de la release)
```bash
git checkout release/portfolio_core_1h_v3_gate2_oos_v1r2
# ajouter le doc, commit
git tag -d portfolio_core_1h_v3_gate2_oos_v1r2
git tag -a portfolio_core_1h_v3_gate2_oos_v1r2 -m "core_v3 r2 + env summary"
git push --force origin portfolio_core_1h_v3_gate2_oos_v1r2
```

---

**Score release :** ~95/100  
**Pour 97+ :** Valider CI/CD smoke sur tag + ajouter Operational Playbook minimal.
