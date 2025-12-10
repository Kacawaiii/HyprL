# HyprL – Architecture & Design (V1.3)

Ce document décrit **l’architecture actuelle** de HyprL, la structure du dépôt, les principaux pipelines (données → features → modèle → backtest → supersearch), ainsi que l’intégration du moteur natif Rust (`hyprl_supercalc`).

Pour la **théorie générale de trading algorithmique** (formules d’indicateurs, concepts NLP, API externes, etc.), se référer au document d’annexe de théorie (PDF/MD séparé, non détaillé ici).

---

## 1. Objectifs & périmètre du projet

### 1.1 Objectif principal

HyprL est un moteur de recherche de stratégies et de backtests multi-actifs conçu pour :

- Construire des grilles de stratégies (seuils, risk, filtres, sentiments, régimes).
- Backtester rapidement ces stratégies avec un modèle de coûts réaliste.
- Appliquer une couche **risk & robustesse** (PF, Sharpe, MaxDD, RoR, bootstrap).
- Sélectionner un **panel** de stratégies robustes (Phase 1), puis suivre leurs performances (paper trading / realtime).

### 1.2 Stack technique

- **Orchestrateur** : Python (>= 3.11)
- **Moteur natif** : Rust (`native/hyprl_supercalc`, exposé via PyO3)
- **Données** : DataFrames (pandas/polars), fichiers CSV/Parquet, caches locaux
- **Tests** : `pytest` côté Python, `cargo test` côté Rust
- **Scripts CLI** : sous `scripts/` (supersearch, backtests, analyse, etc.)

---

## 2. Architecture & arborescence du dépôt

### 2.1 Structure haut niveau

```text
HyprL/
  src/hyprl/
    backtest/
    portfolio/
    risk/
    search/
    supercalc/
    sentiment/
    model/
    ...
  native/
    hyprl_supercalc/
      src/
      Cargo.toml
      scripts/build_supercalc.sh
  scripts/
    run_supersearch.py
    run_backtest.py
    run_analysis.py
    run_paper_trading.py
    ...
  tests/
    backtest/
    search/
    ...
  docs/
    PROJECT_BRAIN.md
    experiments/
    ...
2.2 Modules Python principaux
src/hyprl/backtest/runner.py
Moteur de backtest Python (référence) : exécution bar-par-bar, coûts, stops/TP, export des trades, métriques.

src/hyprl/risk/metrics.py
Calcul des métriques de performance : PF, Sharpe, MaxDD, CAGR, Sortino, risk of ruin, bootstrap, robustness, etc.

src/hyprl/portfolio/core.py
Agrégation multi-tickers, contraintes portefeuille (PF, Sharpe, MaxDD, RoR, corrélations, pondération).

src/hyprl/search/optimizer.py
Construction de la grille de candidats, appel du moteur (Python ou natif), application des hard constraints, tri et sélection.

src/hyprl/supercalc/__init__.py
Wrapper Python autour de hyprl_supercalc (Rust) :

run_batch_backtest_py

run_native_search_batch

helpers de contraintes et de signal (_normalize_constraints, _config_to_native_dict, _precompute_native_stats, etc.).

src/hyprl/sentiment/*.py
Hooks pour le sentiment et les features alternatifs (stubs ou premières intégrations).

src/hyprl/model/*.py
Modèles de probabilité, calibration, registry Meta-ML (XGBoost/RF/etc. selon l’état du projet).

2.3 Crate Rust hyprl_supercalc
native/hyprl_supercalc/src/core.rs
Types de base : Candle, BacktestConfig, EquityPoint, PerformanceMetrics, BacktestReport.

native/hyprl_supercalc/src/backtest/mod.rs
Boucle de backtest native (exposition, coût des trades, ATR sizing, stops/TP, calcul de l’equity curve).

native/hyprl_supercalc/src/metrics.rs
Calcul des métriques à partir de l’equity : PF, Sharpe, MaxDD, CAGR, Sortino, risk of ruin, bootstrap, robustness (selon progression W1.3).

native/hyprl_supercalc/src/indicators.rs
Indicateurs techniques natifs (SMA, EMA, RSI, MACD, Bollinger, ATR, trend_ratio, rolling_vol) avec parité visée avec Python (W1.1).

native/hyprl_supercalc/src/batch.rs
Moteur de recherche natif : SearchConstraint, run_native_search (grid parallélisée, contraintes, tri lexicographique, top-k).

native/hyprl_supercalc/src/ffi.rs
Bridge PyO3 :

run_batch_backtest_py(df, signal, configs)

run_native_search_py(df, signal, configs, constraints, top_k)

compute_indicators_py(df) (après W1.1)
et conversions IntoPy pour les structs.

3. Pipeline données & contrats
3.1 OHLCV & features
HyprL assume une table OHLCV enrichie de features :

Colonnes de base obligatoires :

timestamp (int ms ou datetime ISO),

open, high, low, close, volume.

Colonnes de features typiques (évolutives selon les presets) :

Tendance : sma_10, sma_50, ema_20, etc.

Momentum : rsi_14, roc_10, etc.

Volatilité : atr_14, rolling_std_20, etc.

Bandes : bb_up_20_2, bb_mid_20_2, bb_low_20_2 (Bollinger).

Régimes : trend_regime, vol_regime (labels tendance/range/high-vol).

Sentiment : sentiment_score, sentiment_bucket (par ex. fear/neutral/greed).

Les noms précis des colonnes sont définis par :

la pipeline de features temps-réel et backtest (ex. src/hyprl/rt/features.py ou modules équivalents),

les fichiers de configuration de features/presets (YAML sous configs/ le cas échéant).

3.2 Contrats de données
Input backtest/supersearch : DataFrame OHLCV+features :

index temporel croissant,

colonnes minimales ci-dessus,

pas de trous majeurs non gérés par le pré-traitement.

Configs YAML (presets) :

Emplacement : sous configs/ (ex. configs/supersearch_presets.yaml).

Contenu typique :

grilles : long_thresholds, short_thresholds, risk_pcts, min_ev_multiples,

flags : trend_filter_flags, sentiment_regimes, bornes min/max de sentiment,

mapping vers SearchConfig.

4. Backtest & Risk Layer
4.1 BacktestConfig & moteur Python
Paramètres clés :

risk_pct, commission_pct, slippage_pct, max_leverage, allow_short, etc.

Filtres : trend gating, sentiment gating, min_ev_multiple, etc.

Fonctionnement du backtest Python :

construction du signal/position,

application des coûts (commission + slippage),

gestion des stops/TP,

export des trades et de l’equity (pour métriques et analyse).

4.2 Moteur natif & parité
BacktestConfig & BacktestReport sont partagés entre Python et Rust.

_run_native_engine_v2 vise à reproduire la logique Python :

ATR-based sizing,

stops/take profits,

filtre d’Expected Value (min_ev),

trend/sentiment gating,

bilan des trades et equity curve.

Parité validée via tests/test_supercalc_parity.py et les tests natifs supplémentaires.

4.3 Risk & robustesse
Métriques de base :

PF (profit factor),

Sharpe,

MaxDD (max drawdown),

CAGR.

Métriques avancées (W1.3) :

Implémentées côté Rust dans metrics.rs (à différents niveaux de complétion) :

Sortino ratio,

risk_of_ruin,

bootstrap (p05/p95),

futur robustness_score (combinaison de PF / DD / volatilité / bootstrap).

Utilisées comme hard constraints et/ou comme features pour la sélection de stratégies.

Ces métriques alimentent :

les hard constraints du supersearch,

l’analyse de robustesse en Phase 1 / Phase 2.

4.4 Interprétation profitabilité & sizing du capital
Les métriques HyprL (PF, Sharpe, MaxDD, etc.) sont calculées à partir d’un capital initial et d’un risk_pct donnés. Pour interpréter “combien d’argent ça fait” :

Capital initial : la plupart des scripts utilisent un capital de référence (ex. 10 000 unités).

risk_pct : fraction du capital risquée par trade (ex. 0.01 = 1 % du capital courant).

Modèle de coûts : typiquement

commission_pct ≈ 0.0005 (0,05 % par côté),

slippage_pct ≈ 0.0005,

soit ~0,2 % round-trip sur un trade complet.

Pour un run donné, les CSV de trades (ex. data/trades_AAPL_1y_seed42.csv) contiennent :

chaque entrée/sortie,

PnL par trade,

equity après chaque trade.

Un analyseur de trades (ex. scripts/analyze_trades.py) peut produire un résumé :

total_return (Strat %) : gain/perte en % sur le capital initial,

pf : profit factor,

sharpe,

max_dd,

expectancy : espérance par trade.

Exemple indicatif
Supposons :

capital initial = 10 000,

risk_pct = 0.01,

modèle de coûts standard.

Un résultat du type :

Strat % ≈ 1.89 → ≈ +189 sur 10 000,

PF ≈ 1.02 → très proche du break-even,

Sharpe ≈ 0.19,

MaxDD ≈ 17.9 %,

indique une stratégie faiblement profitable avec un drawdown non négligeable.
L’objectif de cette section est de permettre la lecture :

“Pour ce risk_pct et ce modèle de coûts, combien je gagne/perds sur 10k, et pour quel risque (MaxDD, RoR) ?”

Des tableaux synthétiques peuvent être générés en parcourant les trades_*.csv et seront intégrables dans DOC_HYPRL.md ou dans le portail (portal/).

5. Moteur natif & supercalc (Rust + wrapper Python)
5.1 Fonctions Rust exposées
run_batch_backtest_py(df, signal, configs) -> [BacktestReport]
Backtests multiples en parallèle sur un même dataset.

run_native_search_py(df, signal, configs, constraints, top_k) -> [BacktestReport]
Grid search natif avec contraintes et tri lexicographique.

compute_indicators_py(df) -> dict/DataFrame (après W1.1)
Calcul des indicateurs natifs avec parité Python.

5.2 Wrapper Python hyprl.supercalc
Normalisation des contraintes Python → Rust :

_normalize_constraints(SearchConfig/hard constraints) -> SearchConstraint.

Conversion des configs vers le format natif :

_config_to_native_dict(CandidateConfig, label_mode, etc.).

Batch helper :

run_native_search_batch(df, signal, configs, constraints, top_k) utilisé par search/optimizer.py pour pré-calculer les stats natives (préfetch).

Intégration dans run_search via _precompute_native_stats :

groupement des candidats par signal/paramètres compatibles,

appel unique au natif par groupe,

réutilisation des BacktestReport dans la boucle d’évaluation Python.

6. Search & Phase 1 (supersearch)
6.1 run_search & SearchConfig
SearchConfig décrit :

ticker(s), période, intervalle,

grilles de paramètres : long_thresholds, short_thresholds, risk_pcts, min_ev_multiples, flags trend/sentiment, etc.

hard constraints : min_trades, min_pf, min_sharpe, max_dd, max_ror, min_expectancy, seuils de robustesse, etc.

run_search :

construit la grille de CandidateConfig,

choisit engine={python,native,auto},

collecte les BacktestReport (Python ou natif, avec préfetch natif si actif),

applique les hard constraints et le scoring,

renvoie une liste de SearchResult triés.

6.2 Mapping constraints YAML → hard gates
Les contraintes de robustesse sont fixées :

via les arguments CLI de scripts/run_supersearch.py :

--min-pf, --min-sharpe, --max-dd, --max-ror, --min-expectancy, etc.

--min-portfolio-pf, --min-portfolio-sharpe, --max-portfolio-dd, --max-portfolio-ror, --max-correlation, etc.

et/ou via des presets YAML (grilles + contraintes par défaut).

SearchConfig (Python) et SearchConstraint (Rust) appliquent ces bornes comme des hard gates :

une stratégie qui ne respecte pas min_pf ou max_dd est rejetée,

des contraintes portefeuille (Sharpe, DD, RoR, corrélations) peuvent invalider un candidat,

passer une valeur à 0.0 désactive en général la contrainte (ex. min_portfolio_sharpe = 0.0).

Bien documenter ce mapping permet de comprendre pourquoi :

une grille “ultralax” peut encore produire 0 survivors (ex. contrainte portefeuille non désactivée),

certaines contraintes s’appliquent au niveau stratégie vs portefeuille.

6.3 Mode Python vs natif
engine="python" : backtest + search 100 % Python (référence historique).

engine="native" :

utilise run_native_search_batch + _precompute_native_stats,

nécessite hyprl_supercalc compilé et importable,

si indisponible, doit lever une erreur explicite (mode forcé).

engine="auto" :

essaie le natif si disponible (native_available()),

fallback Python en cas de problème (import, runtime) avec log clair.

6.4 Phase 1 & expériences
Construction d’un panel de stratégies (ex. docs/experiments/PHASE1_PANEL.csv).

Lancement des sessions paper trading (ex. docs/experiments/PHASE1_SESSIONS.csv).

Analyse et scoring de robustesse (scripts d’analyse dans docs/experiments/).

Log des expériences et décisions dans docs/PROJECT_BRAIN.md.

7. Services périphériques & Ops
En plus du cœur “recherche / backtest / risk”, le dépôt contient plusieurs sous-systèmes autour de HyprL et du moteur natif.

7.1 api/ – Service FastAPI (API V2)
Répertoire : api/
Spécification détaillée : voir V2_API.md.

Rôle :

Exposer HyprL comme service HTTP (API V2).

Endpoints typiques :

/v2/predict : scoring / probas sur un échantillon de features.

/v2/usage : suivi de consommation / crédits.

/v2/sessions : gestion de sessions d’analyse.

/v2/autorank/start : déclenchement de jobs de ranking automatisés.

Paramètres principaux (ENV / config) :

HYPRL_API_HOST, HYPRL_API_PORT

HYPRL_DB_URL

éventuel HYPRL_REDIS_URL

chemins vers data / modèles / presets.

7.2 bot/ – ChatOps / intégrations bot
Répertoire : bot/

Rôle :

Fournir un bot (Discord/Telegram/autre) qui pilote HyprL ou l’API V2 :

lancer des supersearch,

consulter des résultats,

déclencher des analyses.

Dépendances typiques :

HYPRL_API_URL : URL de l’API.

Token du bot via variables d’environnement.

7.3 portal/ – Dashboard Streamlit / Web
Répertoire : portal/

Rôle :

Visualiser les résultats HyprL de façon interactive :

panels Phase 1,

sessions paper trading,

métriques de robustesse,

graphes equity / drawdown.

Entrées :

CSV/Parquet générés dans docs/experiments/ ou data/.

Config :

HYPRL_DATA_DIR

HYPRL_EXPERIMENTS_DIR

7.4 core_bridge/ – Glue temps réel
Répertoire : core_bridge/

Rôle :

Assurer la colle entre :

le moteur HyprL (backtest, supersearch),

les services temps réel (API V2, bots, portail).

Exemples :

Orchestration de runs récurrents (supersearch périodique, refresh de panel).

Pilotage de sessions temps réel/paper trading à partir de l’API ou du portail.

Abstraction des appels scripts/run_* pour qu’ils soient déclenchables par API.

7.5 deploy/ – Docker / Compose / infra
Répertoire : deploy/

Rôle :

Packaging et déploiement :

Dockerfile HyprL + hyprl_supercalc.

docker-compose.yml pour :

API FastAPI,

DB / Redis éventuels,

workers batch.

Variables d’environnement :

API host/port,

DB/Redis URLs,

chemins montés (HYPRL_DATA_DIR, HYPRL_EXPERIMENTS_DIR).

8. Milestones & état actuel (M1 – V1.3)
8.1 Work Packages W1.x
W1.1 – Indicators parity
Objectif : parité d’indicateurs techniques Python/Rust.

À faire :

Implémenter dans native/hyprl_supercalc/src/indicators.rs :

SMA, EMA, RSI, MACD, Bollinger Bands, ATR,

trend_ratio, rolling_vol.

Exposer :

IndicatorSet,

compute_indicators(candles: &[Candle]) -> IndicatorSet,

compute_indicators_py(df: PyDataFrame) dans ffi.rs.

Ajouter des tests de parité :

tests/test_supercalc_indicators_parity.py (DataFrame OHLCV synthétique, comparaison Python vs Rust).

Statut : TODO (spécifié, non implémenté).

W1.2 – Backtest parity
Parité logique Python/Rust (ATR sizing, stops/TP, min_ev, trend/sentiment gating).
Validée par test_supercalc_parity.py et les tests backtest existants.
Statut : DONE.

W1.3 – Advanced metrics

Objectif : aligner les métriques avancées Python/Rust.

Déjà fait (partiel) :

Sortino, risk_of_ruin, bootstrap (p05/p95) dans metrics.rs.

Intégration dans PerformanceMetrics.

Restant :

Définir et implémenter robustness_score côté Rust (même logique que Python).

S’assurer que toutes les métriques sont exposées via PyO3.

Ajouter un test dédié, ex. tests/test_supercalc_risk_metrics_parity.py.

Statut : PARTIAL.

W1.4 – Native search

Objectif : pipeline de recherche entièrement natif, branché sur optimizer.py.

Implémenté :

Côté Rust :

SearchConstraint,

run_native_search (batch, contraintes, tri lexicographique).

Côté Python :

run_native_search_batch dans supercalc.py,

_precompute_native_stats dans optimizer.py,

intégration dans run_search (engine natif + auto).

À valider :

pytest tests/test_supercalc_*.py -q,

pytest tests/search/test_optimizer_native.py -q,

puis plus largement pytest tests/search/test_optimizer.py tests/search/test_optimizer_native.py -q.

Statut : Implémenté, validation complète dépendante de la suite pytest.

8.2 Prochains milestones (exemple)
M2 – Sentiment + modèles boostés

Pipeline sentiment réel (source, cache, features),

Intégration XGBoost/LightGBM + calibration + ensembles.

M3 – API V2 + déploiement

Service FastAPI, usage/credits, sessions, autorank,

Docker/Compose, monitoring, métriques.

9. Opérations & commandes clés
9.1 Build du moteur natif
bash
Copier le code
cd ~/HyprL
bash scripts/build_supercalc.sh
9.2 Tests Python
Tests ciblés supercalc :

bash
Copier le code
source .venv/bin/activate
pytest tests/test_supercalc_*.py -q
Tests search (Python + natif) :

bash
Copier le code
pytest tests/search/test_optimizer.py tests/search/test_optimizer_native.py -q
Suite complète :

bash
Copier le code
pytest tests/ -q
9.3 Exemples de commandes CLI
Supersearch simple :

bash
Copier le code
python scripts/run_supersearch.py \
  --ticker AAPL \
  --period 1y \
  --interval 1d \
  --engine native \
  --min-portfolio-sharpe 0.0 \
  --output docs/experiments/SUPERCALC_NATIVE_AAPL_1y.csv
Backtest simple (ex.) :

bash
Copier le code
python scripts/run_backtest.py \
  --ticker AAPL \
  --period 1y \
  --interval 1d \
  --config-path configs/example_backtest.yaml
Paper trading (ex.) :

bash
Copier le code
python scripts/run_paper_trading.py \
  --session-config docs/experiments/PHASE1_PANEL.csv \
  --output docs/experiments/PHASE1_SESSIONS.csv
9.4 Pipelines Phase 1 & modes de recherche
Scripts Phase 1 :

scripts/build_phase1_panel.py
Construit un panel de stratégies à partir des sorties de supersearch.

scripts/run_phase1_experiments.py
Lance des sessions de paper trading / simulations sur le panel, exporte les résultats.

scripts/analyze_phase1_results.py
Agrège les performances, calcule des scores de robustesse, produit les CSV de synthèse.

Modes de moteur :

--engine python : backtest + search 100 % Python.

--engine native : utilise hyprl_supercalc (Rust) avec pré-calcul natif.

--engine auto : essaie le natif, fallback Python si nécessaire.

Ensemble, ces scripts forment le pipeline Phase 1 :
recherche → panel → sessions → analyse.

10. Références & historique
docs/PROJECT_BRAIN.md
Journal interne des évolutions (commits significatifs, tests, expériences).

Annexe théorique (trading, indicateurs, NLP, APIs, etc.)
Document séparé : conception d’un système de trading algorithmique avancé
(référence locale : /mnt/data/Conception d’un système de trading algorithmique avancé.pdf).

docs/experiments/
Exports des panels, sessions, résultats supersearch/backtests, etc.

pgsql
Copier le code

Si tu veux, étape suivante on peut effectivement préparer un petit tableau “perf par 10k” pour 2–3 CSV de trades existants, que tu pourras coller dans la section 4.4 ou dans le portail
::contentReference[oaicite:0]{index=0}