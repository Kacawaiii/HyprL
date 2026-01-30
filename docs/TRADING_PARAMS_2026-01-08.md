# HyprL Trading Parameters Documentation
> Version: 2026-01-09 | Post-fix après pertes overnight

## Historique des Changements

| Date | Changement | Raison |
|------|------------|--------|
| 2026-01-08 | NVDA long 0.53→0.60 | Seuil trop bas, trades marginaux |
| 2026-01-08 | NVDA short 0.45→0.55 | Idem |
| 2026-01-08 | NVDA min_ev 0.0→0.15 | EV gate désactivé |
| 2026-01-08 | NVDA fusion min→mean | Pénalisé par pire timeframe |
| 2026-01-08 | Bridge slippage 5%→0.3% | Entrées trop hautes |
| 2026-01-08 | Bridge spread guard 0.5% | Nouveau |
| 2026-01-08 | Bridge brackets DAY→GTC | Expiration à 16h |
| 2026-01-08 | Bridge cancel ALL orders | Bracket legs non-annulables |
| **2026-01-09** | **NVDA long 0.60→0.68** | **Encore plus conservateur** |
| **2026-01-09** | **NVDA short 0.55→0.62** | **Idem** |
| **2026-01-09** | **risk_pct 1%→0.5%** | **Taille positions réduites** |
| **2026-01-09** | **min_ev_multiple 0.15→0.25** | **EV gate plus strict** |
| **2026-01-09** | **cooldown 3600s→300s** | **Plus de trades mais plus petits** |
| **2026-01-09** | **slippage 0.3%→0.15%** | **Encore plus strict** |
| **2026-01-09** | **spread 0.5%→0.3%** | **Idem** |
| **2026-01-09** | **close-eod-minutes=10** | **FERMETURE AUTO 15:50 ET** |

---

## 1. Paramètres par Ticker

### NVDA (40% du portfolio)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **interval** | 1h | Timeframe principal |
| **period** | 60d | Données historiques |
| **model** | XGBoost v3 | `models/nvda_1h_xgb_v3.joblib` |

#### Thresholds (MODIFIÉ 2026-01-08)
| Paramètre | Avant | Après | Marge vs 50% |
|-----------|-------|-------|--------------|
| **long** | 0.53 | **0.60** | +10% |
| **short** | 0.45 | **0.55** | +5% |

#### Risk Management
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| risk_pct | 0.01 | 1% du capital par trade |
| atr_multiplier | 1.5 | Stop = 1.5 × ATR |
| reward_multiple | 1.9 | TP = 1.9 × risque (R:R 1:1.9) |
| min_position_size | 5 | Minimum 5 actions |

#### Trailing Stop
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| enabled | true | Trailing activé |
| stop_activation | 1.0 | Active après +1R de profit |
| stop_distance | 0.04 | Trail de $0.04 |

#### Multi-Timeframe (MODIFIÉ 2026-01-08)
| Paramètre | Avant | Après | Description |
|-----------|-------|-------|-------------|
| frames | 15m,1h,4h | 15m,1h,4h | Inchangé |
| fusion_method | min | **mean** | Moyenne pondérée |
| weights | 0.2/0.6/0.2 | 0.2/0.6/0.2 | 15m/1h/4h |

#### EV Gate (MODIFIÉ 2026-01-08)
| Paramètre | Avant | Après | Description |
|-----------|-------|-------|-------------|
| min_ev_multiple | 0.0 | **0.15** | EV minimum 0.15R |

#### Trend Filter
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| enable_trend_filter | true | Filtre de tendance actif |
| trend_long_min | -0.01 | Long si trend > -1% |
| trend_short_min | -0.001 | Short si trend > -0.1% |

---

### MSFT (30% du portfolio)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **interval** | 1h | Timeframe principal |
| **model** | XGBoost v3 | `models/msft_1h_xgb_v3.joblib` |

#### Thresholds
| Paramètre | Valeur | Marge vs 50% |
|-----------|--------|--------------|
| **long** | 0.63 | +13% |
| **short** | 0.55 | +5% |

#### Risk Management
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| risk_pct | 0.01 | 1% du capital par trade |
| atr_multiplier | 1.0 | Stop = 1.0 × ATR |
| reward_multiple | 1.5 | R:R 1:1.5 |
| min_ev_multiple | 0.1 | EV minimum 0.1R |

#### Throttling
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| max_daily_trades | 3 | Max 3 trades/jour |
| min_bars_between_entries | 6 | 6h entre entrées |

#### Multi-Timeframe
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| fusion_method | min | Pire des timeframes |
| weights | 0.2/0.6/0.2 | 15m/1h/4h |

---

### QQQ (30% du portfolio)

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **interval** | 1h | Timeframe principal |
| **model** | XGBoost v3 | `models/qqq_1h_xgb_v3.joblib` |

#### Thresholds
| Paramètre | Valeur | Marge vs 50% |
|-----------|--------|--------------|
| **long** | 0.60 | +10% |
| **short** | 0.53 | +3% |

#### Risk Management
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| risk_pct | 0.01 | 1% du capital par trade |
| atr_multiplier | 1.0 | Stop = 1.0 × ATR |
| reward_multiple | 1.5 | R:R 1:1.5 |
| min_ev_multiple | 0.2 | EV minimum 0.2R |

#### Throttling
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| max_daily_trades | 1 | Max 1 trade/jour |
| min_bars_between_entries | 6 | 6h entre entrées |

---

## 2. Features du Modèle (19 features)

```
ret_1h          # Return 1h
ret_3h          # Return 3h
ret_6h          # Return 6h
ret_24h         # Return 24h
atr_14          # ATR 14 périodes
atr_72          # ATR 72 périodes
atr_14_norm     # ATR 14 normalisé
atr_72_norm     # ATR 72 normalisé
rsi_7           # RSI 7
rsi_14          # RSI 14
rsi_21          # RSI 21
vol_ratio_10_30 # Ratio volume 10/30
vol_regime_high # Régime haute volatilité
volume_zscore_24 # Z-score volume 24h
volume_surge    # Surge de volume
range_pct       # Range en %
true_range      # True Range
ret_skew_20     # Skewness returns 20
ret_kurt_20     # Kurtosis returns 20
```

---

## 3. Paramètres du Bridge (Exécution)

### Général
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| symbols | NVDA,MSFT,QQQ | Tickers tradés |
| paper | true | Mode paper trading |
| allow-short | true | Shorts autorisés |

### Limites
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| max-orders-per-day | 20 | Max ordres/jour |
| max-qty | 50 | Max actions par ordre |
| max-notional-total | $35,000 | Notionnel total max |
| max-notional-per-symbol | $15,000 | Notionnel par ticker |
| max-open-positions | 3 | Positions simultanées max |

### Polling
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| poll-seconds | 60 | Intervalle hors RTH |
| poll-seconds-rth | 10 | Intervalle pendant RTH |
| kill-switch-poll-seconds | 5 | Check kill switch |

### Cooldowns (MODIFIÉ 2026-01-09)
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| cooldown-seconds | **300** | 5min après trade (était 3600) |
| cooldown-stale-seconds | 1800 | 30min signal périmé |
| max-signal-age-seconds | 4500 | 1h15 max âge signal |

### Risk
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| max-daily-drawdown-pct | 2.0 | Stop trading si -2% jour |
| use-brackets | true | Ordres bracket (stop+TP) |

### Guards (MODIFIÉ 2026-01-09)
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| max-entry-slippage-pct | **0.15** | Reject si slippage > 0.15% (était 0.3%) |
| max-spread-pct | **0.3** | Reject si spread > 0.3% (était 0.5%) |
| skip-past-tp | **true** | Skip si prix > TP |

### EOD Auto-Close (NOUVEAU 2026-01-09)
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| close-eod-minutes | **10** | Ferme toutes positions 10min avant close (15:50 ET) |

**Pourquoi:** Évite le risque overnight. Exemple: -$40 à la clôture → -$150 le lendemain matin.

**Comportement:**
- Déclenché à 15:50 ET (10 min avant 16:00)
- Annule tous les ordres ouverts (brackets)
- Soumet des ordres MARKET pour fermer les positions
- Flag `eod_closed_today` empêche les re-triggers

**Note:** `use-brackets` gère les sorties (TP/SL) mais l'entrée reste **MARKET** tant que `--use-limit-orders` n'est pas activé.

---

## 4. Logique de Décision

### Flow d'un Signal

```
1. Signal Generator (cron toutes les heures RTH)
   ├── Fetch données OHLCV
   ├── Calcul 19 features
   ├── XGBoost predict_proba()
   ├── Multi-timeframe fusion (mean)
   └── Output: {symbol, decision, probability, entry_price, stop, tp}

2. Bridge (poll toutes les 10s RTH)
   ├── Read signal from JSONL
   ├── Check signal age (< 4500s)
   ├── Check cooldown
   ├── Check daily drawdown (< 2%)
   ├── Fetch quote (BID/ASK)
   ├── Spread guard (< 0.5%)
   ├── Slippage guard (< 0.2% from entry_price)
   ├── Skip-past-TP guard
   ├── EV gate (> min_ev_multiple × R)
   └── Submit bracket order (GTC)
```

### Calcul Expected Value

```
EV = (P_win × Reward) - (P_loss × Risk)
   = (prob × reward_multiple × R) - ((1-prob) × R)
   = R × [prob × reward_multiple - (1-prob)]

Exemple NVDA Long @ 60%:
  EV = R × [0.60 × 1.9 - 0.40]
     = R × [1.14 - 0.40]
     = 0.74R

  min_ev_multiple = 0.15R
  0.74R > 0.15R → PASS
```

### Calcul Slippage

```
Pour LONG: slippage = (ASK - entry_price) / entry_price
Pour SHORT: slippage = (entry_price - BID) / entry_price

Exemple:
  entry_price = $187.25
  ASK = $190.15
  slippage = (190.15 - 187.25) / 187.25 = 1.55%

  max_slippage = 0.3%
  1.55% > 0.3% → REJECT
```

---

## 5. Comparaison Avant/Après

### Trade NVDA du 2026-01-07 (Perte -$354)

| Étape | Avant Fix | Après Fix |
|-------|-----------|-----------|
| Signal prob | 70.6% | 70.6% |
| Long threshold | 0.53 ✓ | 0.60 ✓ |
| Entry price signal | $187.25 | $187.25 |
| Prix au fill | $190.15 | - |
| Slippage | 1.55% | **REJECT** (>0.3%) |
| Bracket TIF | DAY (expire 16h) | GTC (persist) |
| Résultat | -$354 | Pas de trade |

### Scénario Futur (avec fix)

```
Signal: NVDA LONG @ $185.00, prob=62%
Bridge:
  1. Fetch quote: BID=$184.98, ASK=$185.02
  2. Spread: 0.02% < 0.5% → OK
  3. Slippage: (185.02 - 185.00) / 185.00 = 0.01% < 0.2% → OK
  4. EV: 0.62 × 1.9 - 0.38 = 0.80R > 0.15R → OK
  5. Submit bracket GTC

Position:
  - Entry: ~$185.02 (ASK)
  - Stop: $182.50 (1.5 ATR)
  - TP: $189.79 (1.9R)
  - Bracket persist jusqu'à hit
```

---

## 6. Fichiers de Référence

| Fichier | Description |
|---------|-------------|
| `configs/NVDA-1h_v3.yaml` | Config NVDA |
| `configs/MSFT-1h_v3.yaml` | Config MSFT |
| `configs/QQQ-1h_v3.yaml` | Config QQQ |
| `configs/portfolio_core_1h_v3.yaml` | Portfolio |
| `scripts/execution/run_alpaca_bridge.py` | Bridge |
| `src/hyprl/broker/alpaca.py` | Broker (get_latest_quote) |
| `live/logs/live_signals.jsonl` | Signaux live |
| `live/execution/alpaca/orders.jsonl` | Logs ordres |

---

## 7. Monitoring

### Logs à surveiller

```bash
# Signaux générés
tail -f /home/ubuntu/HyprL/live/logs/live_signals.jsonl

# Ordres bridge
tail -f /home/ubuntu/HyprL/live/execution/alpaca/orders.jsonl | jq

# Cron signals
tail -f /home/ubuntu/HyprL/live/logs/cron_signals.log
```

### Events importants

| Event | Signification |
|-------|---------------|
| `slippage_reject_long` | Slippage > 0.15%, trade rejeté |
| `spread_reject` | Spread > 0.3%, trade rejeté |
| `skip_past_tp_long` | Prix déjà > TP, trade inutile |
| `expected_value_below_min` | EV < min_ev, trade rejeté |
| `open_long_bracket` | Ordre bracket soumis |
| `cancel_order_success` | Ordre annulé (bracket leg) |
| `eod_close_triggered` | Fermeture EOD déclenchée (15:50 ET) |
| `eod_close_position` | Position fermée avant EOD |
| `eod_close_error` | Erreur fermeture EOD |

---

## 8. Risques Résiduels

| Risque | Mitigation | Status |
|--------|------------|--------|
| Model drift | Retrain mensuel | TODO |
| Feature mismatch | Vérifier 19 features live | TODO |
| Gap overnight | **EOD auto-close 15:50 ET** | **OK** |
| Rate limits Alpaca | Backoff si 429 | TODO |
| DST changement heure | ZoneInfo America/New_York | OK |

---

*Document mis à jour le 2026-01-09 avec EOD auto-close et paramètres plus conservateurs.*
