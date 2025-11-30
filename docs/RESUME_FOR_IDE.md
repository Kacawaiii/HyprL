# HyprL — Résumé complet pour IDE

Ce document résume l’ensemble du projet pour te permettre de naviguer rapidement dans le code, d’exécuter les workflows clés (recherche, backtest, supersearch, paper trading, API V2) et de déployer la stack V2 en Docker.

Sources: `README.md`, `AGENTS.md`, `docs/PROJECT_BRAIN.md`, `docs/V1_1_1_SUMMARY.md`, `docs/V2_API.md`, `docs/DEPLOY_V2_DOCKER.md`, `docs/DEPLOY_V2_PROD_DOCKER.md`, `docs/hyprl_export_overview.md`, `docs/experiments/PHASE1_LOG.md`.

## Vue d’ensemble
- Objectif: moteur d’analyse quantitatif (V1) + services V2 (API/tokens, portal, bot) pour prédiction probabiliste, backtests, recherche de stratégies et exécution «paper» en temps réel.
- Périmètre V1: pipeline déterministe ingestion → features → modèle proba (logistic par défaut) → risk ATR → backtest, plus recherche (supersearch), portefeuille multi‑tickers, paper trading et GUIs Streamlit.
- Périmètre V2: API FastAPI avec tokens/credits/usage/rate‑limit, endpoint predict + sessions + autorank orchestration; portal Streamlit; bot Discord; déploiement docker-compose.

## Architecture & Répertoires clés
- `src/hyprl/`: coeur V1 (data, indicators/features, model, risk, backtest, portfolio, execution, analysis, adaptive/regimes). Voir `docs/V1_1_1_SUMMARY.md` pour la cartographie détaillée des modules.
- `scripts/`: CLIs (analyse, backtest, sweeps, supersearch, paper, phase1 orchestration, realtime MVP, dashboards).
- `configs/`: presets YAML par ticker/intervalle (+ adaptive regimes).
- `data/`: artefacts produits (CSV trades, sweeps, résultats phase1, sessions live). Datasets bruts lourds hors Git.
- `native/` et `accelerators/`: moteurs natifs (Rust «supercalc») avec wrappers; fallback Python si absent.
- `api/`, `portal/`, `bot/`: composants V2 (FastAPI, Streamlit portal, Discord bot).
- `deploy/`: Dockerfiles, compose, exemples de reverse proxy et scripts build/push.
- `tests/`: suites pytest reflétant l’arborescence `src/`.

## Workflows V1 — Commandes rapides
- Analyse ponctuelle: `python scripts/run_analysis.py --ticker AAPL --period 5d`
- Backtest mono‑actif: `python scripts/run_backtest.py --ticker AAPL --period 1y --initial-balance 10000 --seed 42 --long-threshold 0.60 --short-threshold 0.40 --export-trades data/trades_AAPL_1y.csv`
- Sweeps: `run_threshold_sweep.py` (grille longs), `run_universe_sweep.py` (multi‑tickers → CSV comparatif)
- Supersearch (grille recherche): `scripts/run_supersearch.py` avec contraintes risque et mode `--engine auto` (Rust si dispo)
- Paper trading (replay): `scripts/run_paper_trading.py` à partir d’un CSV supersearch, logs sous `data/live/sessions/<id>/`
- Dashboards/GUI: `streamlit run scripts/hyprl_gui.py` (backtest interactif), `hyprl_replay_gui.py` (replay), `hyprl_dashboard.py` (sessions)
- Phase 1 (panel → sessions → résultats): `build_phase1_panel.py`, `run_phase1_experiments.py`, `analyze_phase1_results.py` + journal `docs/experiments/PHASE1_LOG.md`
- Realtime MVP (paper): `scripts/run_realtime_paper.py` + `scripts/analyze_live_session.py`

## Modélisation & Calibration
- Modèles: logistic (par défaut) + `random_forest` optionnel; calibration proba `--calibration {platt,isotonic}`.
- Features: frame technique vectorisée (SMA/EMA ratios, RSI, ATR, volatilité, returns) + stub sentiment.
- Comparaison modèles: `scripts/compare_models.py` (ex. NVDA 1y @1h).

## Risk Layer & Robustesse
- Sizing ATR, stops/TP par R‑multiples, EV filter, coûts/commission/slippage.
- Métriques et contraintes (Supersearch): PF, Sharpe, maxDD, Risk‑of‑Ruin, expectancy, bootstrap (`maxDD_p95`, `pnl_p05`).
- Rejets «hard constraints» avant ranking; score pénalise RoR et DD stressés.

## Portfolio Layer (multi‑tickers)
- Aligne equities, pondérations (égal ou inv_vol), calcule PF/Sharpe/DD/RoR portefeuille + corrélations.
- Contraintes portefeuille: `--min-portfolio-pf`, `--min-portfolio-sharpe`, `--max-portfolio-dd`, `--max-portfolio-ror`, `--max-correlation`.

## Execution & Paper Trading
- `PaperBroker` et engine de replay; logs NDJSON sous `data/live/sessions/<id>/` (events, bars, features, predictions, orders/fills, equity, manifest).
- Analyzer: `scripts/analyze_live_session.py` (PF/Sharpe/DD, exposure, avg_hold_bars, winrate, top rejection reasons).

## Realtime MVP — Incrément 1
- `scripts/run_realtime_paper.py`: features techniques complètes, `ProbabilityModel` local, clamps (rate/qty/cap), meta‑ML optionnel; raisons loggées pour chaque décision.
- Paramètres utiles: `--max-orders-per-min`, `--per-symbol-cap`, `--min-qty/--max-qty`, `--dry-run` (par défaut) et `--enable-paper`.

## API V2 — Tokens, Usage, Predict, Sessions, Autorank
Voir `docs/V2_API.md`.
- Stockage: SQLite par défaut (`HYPRL_DB_URL`), Argon2id pour hash des tokens (plain jamais persisté), events d’usage par appel.
- Auth: Bearer `tok_<id>.<secret>`; `HYPRL_ADMIN_TOKEN` pour bootstrapping.
- Endpoints principaux:
	- `POST /v2/tokens` (admin): crée un token et provisionne l’account (credits, scopes).
	- `POST /v2/predict` (scope `read:predict`): débite crédits (= nb de symboles), journalise usage.
	- `GET /v2/usage` (scope `read:usage`): credits_total/remaining + agrégats par endpoint.
	- `POST /v2/sessions` | `GET/DELETE /v2/sessions/{id}` | `GET /v2/sessions/{id}/report`: orchestration des sessions realtime «paper» et rapports.
	- `POST /v2/autorank/start` | `GET /v2/autorank/{job_id}`: autorank de CSV supersearch → top‑K sessions; coûts = `10 + 50*top_k`.
- Rate‑limit: 60 req/min; Redis optionnel (`REDIS_URL`) pour mutualiser.
- Bridge predict: `HYPRL_PREDICT_IMPL={stub,real}` (fallback auto en cas d’échec du «real»).

## Déploiement V2 — docker-compose
Voir `docs/DEPLOY_V2_DOCKER.md`.
- Préparer `.env.api`, `.env.portal`, `.env.bot` sous `deploy/` (tokens, titres, URLs) à partir des `.example`.
- Démarrage local:
```bash
cd deploy
docker compose -f docker-compose.v2.yml up --build -d
```
- Services: `redis` (rate‑limit/jobs), `api` (FastAPI, 8000), `portal` (Streamlit, 8501), `bot` (Discord).
- Vérifications: `http://localhost:8000/health` → `{"ok": true}`, portal accessible sur `:8501`.
- Arrêt/nettoyage: `docker compose -f docker-compose.v2.yml down [-v]`.

### Déploiement Prod — images & reverse proxy
Voir `docs/DEPLOY_V2_PROD_DOCKER.md`.
- Build & push:
```bash
HYPRL_REGISTRY_PREFIX=ghcr.io/YOUR_ORG/hyprl \
HYPRL_TAG=v2-prod \
sh deploy/build_and_push_v2.sh
```
- VPS: préparer `/opt/hyprl`, copier `deploy/`, `.env.*`, puis:
```bash
cd /opt/hyprl/deploy
export HYPRL_REGISTRY_PREFIX=ghcr.io/YOUR_ORG/hyprl
export HYPRL_TAG=v2-prod
docker compose -f docker-compose.v2.prod.yml pull
docker compose -f docker-compose.v2.prod.yml up -d
```
- Proxy TLS (Caddy/NGINX) d’après les exemples (`deploy/Caddyfile.example`, `deploy/nginx.conf.example`).

## Qualité, Tests & Style
- Installation locale:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
- Tests: `pytest -q` (ou `./scripts/run_tests.sh`).
- Lint/format: `ruff check src tests` et `black src tests`.
- Cibles critiques: parité supercalc Python↔Rust, risk metrics, portfolio core, execution (paper broker), configs.

## Accélération Native (Supercalc)
- Build Rust: `pip install maturin` puis `bash scripts/build_supercalc.sh`.
- Utilisation: `--engine auto|python|native` dans `run_supersearch.py` et autres appels.

## Données & Artefacts
- `data/`: trades exportés, sweeps/supersearch (`data/supersearch_*`), résultats phase1 (`data/experiments/phase1_results.csv`), sessions live (`data/live/sessions/<id>`).
- `backups/`: dumps/trades et dossiers de runs.
- Provenance/sidecars: CSV accompagnés de `.meta.json` + `provenance.json` pour traçabilité.

## Références utiles
- V1 snapshot & résultats: `docs/V1_1_1_SUMMARY.md`, `data/universe_scores_v1_2.csv`.
- Guide export USB: `docs/hyprl_export_overview.md`.
- Agent/Style de réponses: `AGENTS.md` (format d’action/résultat/vérification/next steps).
- Journal d’avancement: `docs/experiments/PHASE1_LOG.md` et `docs/PROJECT_BRAIN.md`.

## Points ouverts / Prochaines étapes
- Améliorer la calibration (isotonic/quantile) et modèles alternatifs (XGBoost/NN, ensembles).
- Renforcer l’adaptive/regimes et valider sur davantage de sessions paper.
- Réduire l’écart vs bench en bull markets (trend filters/macros, gating volatilité/sentiment réel).
- Accélérer les hot paths (features, Monte Carlo, supercalc) côté natif.

---
Pour toute exécution rapide, active `.venv` puis lance les commandes ci‑dessus. Pour la stack V2, complète d’abord les `.env` puis démarre `deploy/docker-compose.v2.yml`.

