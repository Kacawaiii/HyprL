# HyprL — Plan de Finalisation Complet

**Date:** 2025-12-21  
**Version:** v1r3  
**Tag:** `portfolio_core_1h_v3_gate2_oos_v1r3r1`  
**Commit:** `6d8f24f`  
**Objectif:** Projet commercialisable en 3-4 mois

---

# PARTIE 1 — ÉTAT ACTUEL (Ce qu'on a)

## 1.1 Core Trading Engine (FROZEN ✅)

| Composant | Status | Fichiers |
|-----------|--------|----------|
| Models XGBoost v3 | ✅ Frozen | `models/{nvda,msft,qqq}_1h_xgb_v3.joblib` |
| Feature pipeline (19 cols) | ✅ Frozen | `src/hyprl/features/nvda_v2.py` |
| Backtest engine (native Rust) | ✅ Frozen | `native/hyprl_supercalc/` |
| Configs production | ✅ Frozen | `configs/{NVDA,MSFT,QQQ}-1h_v3.yaml` |
| Portfolio config | ✅ Frozen | `configs/portfolio_core_1h_v3.yaml` |

### Métriques validées (OOS 2024-03 → 2024-12)
| Ticker | Trades | PF | Win% | MaxDD |
|--------|--------|-----|------|-------|
| NVDA | 118 | 5.93 | 79.7% | ~2% |
| MSFT | 182 | 2.95 | 74.7% | ~3% |
| QQQ | 168 | 2.77 | 74.4% | ~3% |
| **Portfolio** | 468 | 3.57 | 76.3% | 2.62% |

## 1.2 Broker Integration (DONE ✅)

| Composant | Status | Fichiers |
|-----------|--------|----------|
| BrokerBase contract | ✅ | `src/hyprl/broker/base.py` |
| AlpacaBroker (paper+live) | ✅ | `src/hyprl/broker/alpaca.py` |
| DryRunBroker | ✅ | `src/hyprl/broker/dryrun.py` |
| Execution Bridge | ✅ | `scripts/execution/run_alpaca_bridge.py` |
| Tests unitaires | ✅ | `tests/broker/test_alpaca_broker_mock.py` |
| Documentation | ✅ | `docs/ALPACA_BROKER.md` |
| Env template | ✅ | `.env.broker.alpaca.example` |

### Features broker
- ✅ Order submission/cancel (MARKET, LIMIT, STOP, STOP_LIMIT)
- ✅ Position management
- ✅ Idempotency (client_order_id)
- ✅ Retry + rate limit handling
- ✅ Emergency close all
- ✅ Market clock awareness
- ✅ Risk limits (`--max-notional-per-day`, `--max-orders-per-day`)
- ✅ Kill switch (`--kill-switch`)

## 1.3 Legal & Compliance (DONE ✅)

| Document | Status | Fichier |
|----------|--------|---------|
| Disclaimer (risques) | ✅ | `docs/legal/DISCLAIMER.md` |
| CGU (conditions) | ✅ | `docs/legal/CGU.md` |
| Privacy (RGPD) | ✅ | `docs/legal/PRIVACY.md` |

### Stratégie AMF
- ✅ Service générique (pas de conseil personnalisé)
- ✅ Signaux algorithmiques (pas de recommandations)
- ✅ Education/information framing
- ✅ Disclaimers sur tous les touchpoints

## 1.4 Infrastructure Ops (PARTIAL ⚠️)

| Composant | Status | Fichiers |
|-----------|--------|----------|
| Live runners | ✅ | `scripts/run_live_hour.py` |
| Portfolio replay | ✅ | `scripts/run_portfolio_replay.py` |
| Health monitor | ✅ | `scripts/monitor/monitor_portfolio_health.py` |
| Discord poster | ✅ | `scripts/ops/push_core_v3_discord.py` |
| Heartbeat check | ✅ | `scripts/ops/check_heartbeat.py` |
| Alert webhook | ✅ | `scripts/ops/alert_portfolio_health.py` |
| Track record generator | ⚠️ Partiel | `scripts/ops/run_monthly_track_record.py` |
| Dashboard Streamlit | ⚠️ Local | `scripts/dashboard/palier2_dashboard.py` |

## 1.5 API & Portal (PARTIAL ⚠️)

| Composant | Status | Fichiers |
|-----------|--------|----------|
| FastAPI v2 (auth/quotas) | ✅ | `api/` |
| /v2/predict | ✅ | `api/routers/v2_predict.py` |
| /v2/usage | ✅ | `api/routers/v2_usage.py` |
| /v2/sessions | ✅ | `api/routers/v2_sessions.py` |
| Portal Streamlit | ⚠️ Local | `portal/` |
| Discord Bot | ⚠️ Partiel | `bot/` |

## 1.6 SENSE (Audit-only ✅)

| Composant | Status | Fichiers |
|-----------|--------|----------|
| Pipeline (chrono/fear/fusion) | ✅ Neutralisé | `src/hyprl_sense/` |
| Shadow harness | ✅ | `scripts/run_sense_shadow.py` |
| Analytics | ✅ | `scripts/analysis/sense_*.py` |
| Config | ✅ | `configs/sense.yaml` (risk_multiplier=1.0) |

---

# PARTIE 2 — CE QUI MANQUE (Gaps)

## 2.1 Gaps Critiques (Bloqueurs commercialisation)

| Gap | Priorité | Effort | Impact |
|-----|----------|--------|--------|
| **Track record paper (20+ trades)** | P0 | 2 sem | Crédibilité |
| **Track record live (50+ trades)** | P0 | 6 sem | Preuve réelle |
| **Dashboard public** | P1 | 2 jours | Vitrine |
| **Landing page + waitlist** | P1 | 1 jour | Acquisition |
| **Personnalisation docs légaux** | P1 | 2h | Conformité |

## 2.2 Gaps Importants (Nice-to-have avant launch)

| Gap | Priorité | Effort | Impact |
|-----|----------|--------|--------|
| Tests E2E broker | P2 | 3 jours | Fiabilité |
| CI/CD pipeline | P2 | 2 jours | Automatisation |
| Billing (Stripe) | P2 | 3 jours | Revenus |
| Discord bot complet | P2 | 2 jours | Distribution |
| Multi-broker (IBKR) | P3 | 1 sem | Expansion EU |

## 2.3 Gaps Post-launch (V2)

| Gap | Priorité | Effort |
|-----|----------|--------|
| Mobile app | P4 | 2 mois |
| Options/Futures support | P4 | 1 mois |
| Backtest-as-a-service | P4 | 2 sem |
| White-label licensing | P4 | 1 mois |

---

# PARTIE 3 — PLAN D'ACTION DÉTAILLÉ

## Phase 0: Préparation (Cette semaine)

### 0.1 Personnaliser documents légaux
```bash
# Remplacer les [placeholders] dans:
# - docs/legal/CGU.md
# - docs/legal/PRIVACY.md
# - docs/legal/DISCLAIMER.md

# Informations requises:
# - Nom entreprise: [À définir]
# - Adresse: [À définir]
# - Email contact: contact@hyprl.io
# - Email privacy: privacy@hyprl.io
# - Ville juridiction: Paris
```

**Checklist:**
- [ ] Créer adresse email pro (contact@hyprl.io, privacy@hyprl.io)
- [ ] Décider structure juridique (auto-entrepreneur vs SAS)
- [ ] Remplir placeholders dans les 3 docs
- [ ] Commit + tag

### 0.2 Créer compte Alpaca
```bash
# 1. Inscription
# https://alpaca.markets/signup

# 2. Paper Trading activation (instantané)

# 3. Récupérer API keys
# Dashboard → Paper Trading → View API Keys

# 4. Configurer
cp .env.broker.alpaca.example .env.broker.alpaca
nano .env.broker.alpaca
# ALPACA_API_KEY=PKXXXXXXXXXX
# ALPACA_SECRET_KEY=XXXXXXXXXX
# ALPACA_PAPER=true
```

**Checklist:**
- [ ] Compte Alpaca créé
- [ ] Paper trading activé
- [ ] API keys récupérées
- [ ] .env.broker.alpaca configuré

### 0.3 Valider dry-run bridge
```bash
# Test parsing signaux
.venv/bin/python scripts/execution/run_alpaca_bridge.py \
  --signals live/logs/live_signals.jsonl \
  --out live/execution/alpaca/orders.jsonl \
  --state live/execution/alpaca/state.json \
  --symbols NVDA \
  --dry-run \
  --once

# Vérifier output
cat live/execution/alpaca/orders.jsonl | head -5
```

**DoD:**
- [ ] Bridge parse correctement les signaux
- [ ] Orders loggés en dry-run
- [ ] State checkpoint fonctionne

---

## Phase 1: Paper Trading Validation (2 semaines)

### Objectif
Valider le wiring technique : 0 crash, 0 doublon, signal → order → fill cohérent.

### 1.1 Lancer paper trading
```bash
# Terminal 1: Core v3 (génère les signaux)
.venv/bin/python scripts/run_live_hour.py \
  --config configs/NVDA-1h_v3.yaml \
  --trade-log live/logs/core_v3/trades_NVDA.csv \
  --summary-file live/logs/core_v3/summary_NVDA.json

# Terminal 2: Bridge Alpaca (exécute les ordres)
source .env.broker.alpaca
.venv/bin/python scripts/execution/run_alpaca_bridge.py \
  --signals live/logs/live_signals.jsonl \
  --paper \
  --symbols NVDA \
  --max-orders-per-day 10 \
  --max-notional-per-day 5000 \
  --state live/execution/alpaca/state.json \
  --out live/execution/alpaca/orders.jsonl
```

### 1.2 Monitoring quotidien
```bash
# Health check
.venv/bin/python scripts/monitor/monitor_portfolio_health.py \
  --trade-logs live/execution/alpaca/orders.jsonl \
  --weights NVDA=1.0 \
  --initial-equity 100000 \
  --summary-out live/execution/alpaca/health.json

# Alertes (si webhook configuré)
.venv/bin/python scripts/ops/alert_portfolio_health.py \
  --health live/execution/alpaca/health.json \
  --pf-alert 1.0 \
  --dd-alert 15 \
  --webhook $DISCORD_WEBHOOK
```

### 1.3 Reconciliation
```bash
# Comparer signaux vs orders vs fills
# Script à créer: scripts/ops/reconcile_alpaca.py

# Vérifier:
# - Chaque signal actionable a un order
# - Chaque order a un fill (ou rejection loggée)
# - Pas de doublons (idempotency OK)
```

### Critères de succès Phase 1
| Métrique | Cible | Bloquant |
|----------|-------|----------|
| Trades paper | ≥20 | Oui |
| Crashs bridge | 0 | Oui |
| Doublons orders | 0 | Oui |
| Signal→Order match | 100% | Oui |
| Uptime bridge | >95% | Oui |

### Livrables Phase 1
- [ ] `live/track_record/paper/YYYY-MM/` avec orders.jsonl + fills.jsonl
- [ ] Rapport Phase 1 (trades, PF, issues)
- [ ] Go/No-Go pour Phase 2

---

## Phase 2: Live Micro Trading (4-6 semaines)

### Objectif
Valider avec argent réel : coûts réels, slippage réel, psychology réelle.

### Prérequis
- ✅ Phase 1 réussie (20+ trades paper, 0 crash)
- Account Alpaca live approuvé
- Capital déposé: $500-1000

### 2.1 Approbation live Alpaca
```bash
# Alpaca Dashboard → Settings → Live Trading
# Remplir questionnaire (income, experience, etc.)
# Délai: 1-3 jours ouvrés

# Déposer capital
# Wire transfer ou ACH: $500-1000 initial
```

### 2.2 Configuration live micro
```yaml
# .env.broker.alpaca (live)
ALPACA_API_KEY=AKXXXXXXXXXX  # Live key (différente de paper)
ALPACA_SECRET_KEY=XXXXXXXXXX
ALPACA_PAPER=false

# Limits très restrictifs
max_notional_day: 200      # $200/jour max
max_orders_day: 5          # 5 trades/jour max
risk_per_trade: 0.005      # 0.5% par trade
```

### 2.3 Lancer live micro
```bash
# ATTENTION: Argent réel
source .env.broker.alpaca

# Confirmation manuelle requise (le bridge demande "CONFIRM")
.venv/bin/python scripts/execution/run_alpaca_bridge.py \
  --signals live/logs/live_signals.jsonl \
  --live \
  --symbols NVDA \
  --max-orders-per-day 5 \
  --max-notional-per-day 200 \
  --state live/execution/alpaca_live/state.json \
  --out live/execution/alpaca_live/orders.jsonl
```

### 2.4 Suivi quotidien
```bash
# Export equity depuis Alpaca
.venv/bin/python scripts/ops/export_alpaca_equity.py \
  --out live/track_record/live/YYYY-MM/equity.csv

# Réconciliation fills
.venv/bin/python scripts/ops/reconcile_alpaca.py \
  --orders live/execution/alpaca_live/orders.jsonl \
  --out live/track_record/live/YYYY-MM/reconciliation.json

# Métriques
.venv/bin/python scripts/analyze_trades.py \
  --trades live/track_record/live/YYYY-MM/trades.csv
```

### Critères de succès Phase 2
| Métrique | Cible | Bloquant |
|----------|-------|----------|
| Trades live | ≥50 | Oui |
| PF live | ≥1.3 | Oui |
| MaxDD live | ≤20% | Oui |
| Slippage moyen | ≤0.1% | Non |
| Coûts réels | ≤0.2% RT | Non |

### Livrables Phase 2
- [ ] `live/track_record/live/YYYY-MM/` avec données réelles
- [ ] Rapport comparatif paper vs live
- [ ] Métriques slippage/coûts réels
- [ ] Go/No-Go pour Phase 3

---

## Phase 3: Production Setup (2-3 semaines, parallèle)

### 3.1 Dashboard public
```bash
# Option A: Streamlit Cloud (gratuit, rapide)
# 1. Créer compte streamlit.io
# 2. Connect GitHub repo
# 3. Deploy portal/main.py

# Option B: VPS (plus contrôle)
# Voir docs/DEPLOY_V2_PROD_DOCKER.md
```

**Contenu dashboard:**
- Equity curve (paper vs live séparés)
- Trades récents
- Métriques clés (PF, Sharpe, MaxDD)
- Statut système (dernière update)
- Disclaimers visibles

### 3.2 Landing page
```bash
# Carrd.co ou Notion (rapide, pas de code)
# Contenu:
# - Headline: "Signaux trading algorithmique"
# - Métriques clés (avec disclaimer "backtested")
# - Features
# - Pricing preview
# - Waitlist form (Google Forms ou Typeform)
# - Liens légaux (CGU, Privacy, Disclaimer)
```

### 3.3 Discord setup
```bash
# Serveur structure:
# - #welcome (rules, disclaimer)
# - #announcements (updates)
# - #alerts-nvda (signaux)
# - #alerts-msft
# - #alerts-qqq
# - #support
# - #feedback

# Bot commands:
# /predict NVDA → dernier signal
# /usage → quota restant
# /subscribe → lien Stripe
```

### 3.4 Email setup
```bash
# Zoho Mail (gratuit pour 1 user)
# ou Google Workspace ($6/mois)

# Adresses requises:
# - contact@hyprl.io
# - privacy@hyprl.io
# - support@hyprl.io
```

---

## Phase 4: Beta Launch (2-4 semaines)

### 4.1 Recrutement beta
```bash
# Sources:
# - Waitlist (landing page)
# - Reddit r/algotrading, r/quant
# - Twitter/X #algotrading
# - Discord serveurs trading

# Cible: 10-20 beta users
# Critères:
# - Intérêt trading algo
# - Willing to give feedback
# - Comprend les risques
```

### 4.2 Beta pricing
| Tier | Contenu | Prix beta (-50%) | Prix final |
|------|---------|------------------|------------|
| Starter | Discord NVDA only | €24.50/mois | €49/mois |
| Pro | Multi-ticker + API | €49.50/mois | €99/mois |
| Premium | Full API + priority | €99.50/mois | €199/mois |

### 4.3 Onboarding flow
```
1. Inscription via landing page
2. Email de bienvenue + disclaimer
3. Lien Discord (accept rules)
4. Channel #getting-started
5. Premier signal → DM explicatif
6. Feedback form après 1 semaine
```

### 4.4 Feedback collection
```bash
# Google Form hebdomadaire:
# - Qualité signaux (1-5)
# - Clarté alertes (1-5)
# - Bugs rencontrés
# - Features souhaitées
# - NPS (0-10)

# Objectifs:
# - NPS ≥30
# - 5+ testimonials écrits
```

### Critères de succès Phase 4
| Métrique | Cible |
|----------|-------|
| Beta users | 10-20 |
| Retention 30j | ≥60% |
| NPS | ≥30 |
| Testimonials | ≥5 |
| Bugs critiques | 0 |

---

## Phase 5: Production Launch (Semaine 12+)

### 5.1 Billing integration
```bash
# Stripe (recommandé) ou Paddle (EU taxes simplifiées)

# Setup:
# 1. Compte Stripe
# 2. Products (Starter, Pro, Premium)
# 3. Subscriptions (monthly)
# 4. Webhook → API quota update
# 5. Customer portal (gestion abo)
```

### 5.2 API quotas
```python
# api/plan_limits.py
PLAN_LIMITS = {
    "free": {"daily_predictions": 10, "rate_limit_rpm": 10},
    "starter": {"daily_predictions": 100, "rate_limit_rpm": 30},
    "pro": {"daily_predictions": 500, "rate_limit_rpm": 60},
    "premium": {"daily_predictions": 2000, "rate_limit_rpm": 120},
}
```

### 5.3 Support SLA
| Tier | Response time | Channel |
|------|---------------|---------|
| Free | Best effort | Discord public |
| Starter | 48h | Email |
| Pro | 24h | Email + Discord DM |
| Premium | 4h | Email + Discord + Call |

### 5.4 Marketing push
```bash
# Channels:
# - Twitter/X (daily insights)
# - Reddit (weekly value posts, NO spam)
# - LinkedIn (professional angle)
# - YouTube (tutorial videos)
# - Blog (SEO, case studies)

# Budget: €0-500/mois initial
# Focus: organic + testimonials
```

---

# PARTIE 4 — TIMELINE GLOBALE

```
Semaine 0 (maintenant)
├── Personnaliser docs légaux
├── Créer compte Alpaca
└── Valider dry-run bridge

Semaines 1-2 (Paper Trading)
├── Lancer paper trading NVDA
├── Monitoring quotidien
├── Atteindre 20+ trades paper
└── Go/No-Go Phase 2

Semaines 3-8 (Live Micro) ← PARALLÈLE avec Phase 3
├── Approbation Alpaca live
├── Dépôt $500-1000
├── Trading live micro
├── Atteindre 50+ trades live
└── Rapport comparatif paper/live

Semaines 3-5 (Production Setup) ← PARALLÈLE
├── Dashboard Streamlit Cloud
├── Landing page + waitlist
├── Discord server structure
└── Email setup

Semaines 6-10 (Beta)
├── Recrutement 10-20 users
├── Beta pricing actif
├── Feedback collection
├── 5+ testimonials
└── Itérations produit

Semaines 10-12 (Production Launch)
├── Stripe integration
├── Full pricing actif
├── Marketing push
└── Support processes
```

---

# PARTIE 5 — CHECKLIST FINALE

## Avant Paper Trading (Semaine 0)
- [ ] Docs légaux personnalisés
- [ ] Compte Alpaca créé
- [ ] API keys paper configurées
- [ ] Dry-run bridge validé
- [ ] Discord server créé

## Avant Live Trading (Semaine 2)
- [ ] 20+ trades paper réussis
- [ ] 0 crash bridge
- [ ] Reconciliation OK
- [ ] Approbation Alpaca live demandée

## Avant Beta (Semaine 5)
- [ ] 50+ trades live
- [ ] PF live ≥1.3
- [ ] Dashboard public déployé
- [ ] Landing page live
- [ ] Beta pricing configuré

## Avant Production (Semaine 10)
- [ ] 100+ trades track record
- [ ] 10+ beta users actifs
- [ ] 5+ testimonials
- [ ] Stripe intégré
- [ ] Support process documenté

---

# PARTIE 6 — RISQUES ET MITIGATIONS

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Drawdown sévère live | Moyenne | Élevé | Kill switch, max 20% DD |
| Alpaca API down | Faible | Moyen | Retry/backoff, alerting |
| Pas assez de signaux | Moyenne | Moyen | Multi-ticker (NVDA+MSFT+QQQ) |
| Régulation AMF | Faible | Élevé | Disclaimers, pas de conseil perso |
| Pas de traction beta | Moyenne | Moyen | Itérer sur feedback, pivot pricing |
| Concurrence | Faible | Faible | Focus qualité signaux + UX |

---

# PARTIE 7 — BUDGET ESTIMÉ

| Poste | Coût mensuel | Notes |
|-------|--------------|-------|
| Streamlit Cloud | €0 | Tier gratuit suffisant |
| Carrd landing | €0-19 | Pro si domaine custom |
| Zoho Mail | €0 | 1 user gratuit |
| Discord | €0 | Gratuit |
| Alpaca | €0 | Pas de frais plateforme |
| Domaine hyprl.io | €15/an | Namecheap |
| VPS (optionnel) | €5-20 | Hetzner/OVH si besoin |
| Stripe fees | 2.9% + €0.25 | Par transaction |
| **Total initial** | **€0-50/mois** | Avant revenus |

---

# PARTIE 8 — COMMANDES CLÉS

```bash
# === PHASE 0: PRÉPARATION ===
# Dry-run test
.venv/bin/python scripts/execution/run_alpaca_bridge.py \
  --signals live/logs/live_signals.jsonl --symbols NVDA --dry-run --once

# === PHASE 1: PAPER TRADING ===
# Core v3 (Terminal 1)
.venv/bin/python scripts/run_live_hour.py \
  --config configs/NVDA-1h_v3.yaml

# Bridge paper (Terminal 2)
source .env.broker.alpaca
.venv/bin/python scripts/execution/run_alpaca_bridge.py \
  --signals live/logs/live_signals.jsonl --paper --symbols NVDA

# === PHASE 2: LIVE MICRO ===
# Bridge live (confirmation requise)
.venv/bin/python scripts/execution/run_alpaca_bridge.py \
  --signals live/logs/live_signals.jsonl --live --symbols NVDA \
  --max-notional-per-day 200 --max-orders-per-day 5

# === MONITORING ===
# Health
.venv/bin/python scripts/monitor/monitor_portfolio_health.py \
  --trade-logs live/execution/alpaca/orders.jsonl --weights NVDA=1.0

# Track record mensuel
.venv/bin/python scripts/ops/run_monthly_track_record.py \
  --live-root live/execution/alpaca --output-root reports
```

---

# CONCLUSION

Le projet HyprL est à **~80% de complétion** pour une commercialisation.

**Ce qui est solide:**
- ✅ Core v3 frozen avec métriques validées (PF 3.57, 468 trades OOS)
- ✅ Broker Alpaca complet (paper + live)
- ✅ Bridge execution avec safety (limits, kill switch)
- ✅ Legal pack RGPD-compliant

**Ce qui reste:**
- ⚠️ Track record réel (paper → live)
- ⚠️ Dashboard/landing public
- ⚠️ Beta users + testimonials

**Timeline réaliste:** 3-4 mois jusqu'au lancement production.

**Action immédiate:** Configurer Alpaca et lancer le dry-run test.

---

*Document généré le 2025-12-21 — Version v1r3*
