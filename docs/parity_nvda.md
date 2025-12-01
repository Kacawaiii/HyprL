# NVDA Parity Remediation Plan

## 1. Contexte
- Actif: NVDA 1h, fenêtre glissante 1 an (1 407 barres).
- Source des écarts: `data/signals/nvda_signal_diff.csv` (golden diff export backtest ↔ live replay).
- Objectif global: écart combiné (bt_only + replay_only + décisions divergentes) < 2 % et probabilité identique à 1e-6 sur toutes les barres.

## 2. Statistiques clés (CSV 2024-12-12 → 2025-11-26)
- `_merge`: 1 259 `both`, 139 `right_only` (replay-only), 9 `left_only` (BT-only).
- Flags internes: 49 BT-only décisions (3.48 %), 21 replay-only décisions (1.49 %).
- Décisions divergentes: 43/1 259 (3.4 %) `emit` BT vs `reject` replay; 42/86 (=48.8 %) des signaux émis BT sont rejetés live.
- Probabilité: |Δprob| moyenne 4.19e-3, max 6.36e-2; 78 % > 1e-3; divergences ciblées 2.16e-2 en moyenne.
- Expected P&L: 44 barres comparables → |ΔEV| moyenne 6.27, max 25.8 (replay ne remonte pas EV quand il rejette).
- Position sizing: |Δsize| max 8.64 (≈13 %), |Δrisk_amount| max 17.8 (≈12 %).
- Thresholds: identiques (diff=0).

## 3. Hypothèses classées
1. **Feature misalignment** (très probable) – 139 `right_only`, `source_bt` vide; caches décalés / fenêtrage.
2. **Warmup gaps** (très probable) – absence de pré-roll aligné.
3. **Model path divergence** (probable) – scaler/seeds non figés → Δprob uniformes.
4. **Sizing mismatch** (probable) – RiskManager live lit equity dynamique; BT utilise courbe historique figée.
5. **Floating-point/order** (faible) – conversions float32/64; n’impacte que la cible 1e-6.

## 4. Protocole de diagnostic
1. **Microscope obligatoire (3–5 jours)**
   - Rejouer NVDA 1h sur 3–5 jours via `HYPRL_PARITY_MODE=microscope`.
   - Exiger 100 % décisions identiques, max |Δprob| < 1e-4, sizing identique à 1e-6 avant tout rerun 1 an.
2. **Tests unitaires / golden fixtures**
   - Générer fixtures NVDA depuis le pipeline qui produit `nvda_signal_diff.csv`; committer sous `tests/fixtures/parity/nvda_microscope.parquet`.
   - `tests/parity/test_strategy_signal_parity.py`: recharge fixtures, appelle `StrategyEngine.decide()` BT vs live shim, assert prob/decision/flags à 1e-6.
   - `tests/parity/test_risk_position_size.py`: vérifie `calc_position_size`/`expected_pnl` sur ATR/equity partagés.
3. **Instrumentation SignalTrace**
   - Dataclass `SignalTrace` (timestamp, feature_hash, indicators, model_params_hash, risk_inputs, prob, EV, position_size).
   - Writer sous `data/parity_traces/NVDA_<ts>.jsonl` déclenché quand `HYPRL_PARITY_TRACE=1`. Replay et BT écrivent les mêmes points.
4. **Vérifications features**
   - Diff DataFrame complet entre `BacktestFeatureStore` et `LiveFeatureStore` (index, NaN, warmup) pour chaque microscope window.
   - Verrouiller `sentiment_ok/trend_ok` bool via cache commun.
5. **Vérifications modèle**
   - Snapshot unique `artifacts/model_prob_v2.pkl` (weights + scaler + seed). Les deux moteurs le chargent en lecture seule.
   - Instrumenter `ProbabilityModel.predict_proba` pour logguer |Δinputs| > 1e-6 entre BT et live durant le microscope.
6. **Vérifications sizing/risk**
   - `RiskManager` consomme un `EquityContext` injecté (BT curve). Plus de lecture broker pendant un replay.
   - `calc_position_size` arrondit en float64 + `round(6)` avant retour.
   - `StrategyEngine` peut recevoir une map `parity_equity` (timestamp → equity BT) pour forcer la même équité lors des appels `decide_signals_on_bar` en replay.

## 5. Correctifs code-level
- **Feature pipeline**: refactor `src/hyprl/features/pipeline.py` → `FeatureFrameSnapshot` partagé, dtype float64, caches alignés; hook warmup explicite.
- **Model sharing**: charger `ProbabilityModel` via `ProbabilityModel.load_artifact(...)` côté BT/live (seed fixe, scaler figé). Ajouter test de regression sur vecteur NVDA microscope.
- **Signal path**: exposer `StrategyInputs` structuré; `StrategyEngine.decide()` devient pure et commune; replay ne court-circuite plus EV.
- **Risk manager**: isoler calculs (position_size, risk_amount, expected_pnl) dans `risk.calculators` avec tests golden; aligner rounding.
- **Logging parity**: `scripts/check_parity.py` rejoue NVDA microscope + 1 an, produit stats identiques à `nvda_signal_diff.csv`, échoue si seuils dépassés.

## 6. Critères d’acceptation par étape
### Après instrumentation + features/model share
- bt_only ≤ 1 %, replay_only ≤ 1 % (microscope + 1 an).
- |Δprob| moyen < 1e-3, max < 1e-2 (microscope).
- 100 % décisions identiques sur la fenêtre microscope.

### Après alignement RiskManager
- max |position_size_bt − position_size_replay| < 1e-3.
- max |risk_amount_bt − risk_amount_replay| < 1e-2.
- Expected P&L toujours présent côté replay pour les signaux émis.

### Final (live-ready)
- (bt_only + replay_only + décision divergente) < 2 % sur 1 an NVDA.
- max |Δprob| ≤ 1e-4, moyenne ≤ 5e-5.
- `scripts/check_parity.py` passe en CI (fail si seuils violés).

## 7. Roadmap
1. **Semaine 1** – Implémenter SignalTrace, pipeline microscope, tests parity initiaux. Critère: microscope 3–5 jours = 100 % aligné.
2. **Semaine 2** – Refactor feature/model sharing; régénérer fixtures; cible: bt_only/replay_only ≤ 1 % et Δprob moyen < 1e-3.
3. **Semaine 3** – Harmoniser RiskManager + EV logging; cible sizing/risk diff sous 1e-3/1e-2.
4. **Semaine 4** – Lancer `scripts/check_parity.py` (microscope + 1 an) en CI; documenter résultat; activer mode live-ready.

## 8. Actions prochaines
1. Générer les fixtures microscope (`data/parity/nvda_backtest_signal_log_MICRO.csv`, `nvda_replay_signal_log_MICRO.csv`, `nvda_signal_diff_MICRO.csv`) via `python scripts/make_nvda_microscope_fixtures.py --start 2025-01-15 --end 2025-01-20`.
2. Construire `SignalTrace`, tests parity (`tests/parity/*`) et `scripts/check_parity.py`.
3. Appliquer refactors features/model/risk puis répéter microscope → 1 an jusqu’à respect des seuils.
