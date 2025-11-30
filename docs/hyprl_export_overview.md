# HyprL — Fiche d'Export

Version du snapshot : 2025-11-10  
Objectif : fournir sur une clé USB tout le contexte utile (code, scripts, tests, GUI, état d’avancement, commandes critiques).

## 1. Résumé Technique
- Moteur de recherche quant HyprL : ingestion OHLCV + sentiment stub → indicateurs multi-frames → modèle probabiliste → plan de risque ATR → backtest avec coûts/benchmarks.  
- Stack : Python 3.11, venv + pip, pandas, scikit-learn, Streamlit pour les GUI, pybind11/pyo3 (future accélération).  
- Sorties principales : rapports CLI (`run_analysis`, `run_backtest`, sweeps), CSV de trades, GUI Streamlit, exports de seuils.

## 2. Arborescence à Copier
- `src/hyprl/` : coeur (data, indicators, model, risk, backtest, pipeline).  
- `scripts/` : CLIs (`run_analysis.py`, `run_backtest.py`, `run_threshold_sweep.py`, `run_universe_sweep.py`, `run_supersearch.py`, `analyze_trades.py`, `hyprl_gui.py`, `hyprl_replay_gui.py`).  
- `configs/` : presets YAML (seuils, adaptive).  
- `tests/` : miroir unitaire/integ (pytest).  
- `accelerators/` : squelettes C++/Rust (optionnel).  
- `data/` : exports/outils (trades, sweeps). Datasets lourds restent hors Git mais peuvent être ajoutés à la clé si besoin.  
- `README.md`, `AGENTS.md`, `pyproject.toml`, `docs/hyprl_export_overview.md` (ce fichier), `PROJECT_MEMORY/PROJECT_BRAIN` si présents.

## 3. Installation Locale (venv)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Supercalc natif (optionnel) :
```bash
pip install maturin
bash scripts/build_supercalc.sh
```
Accélération (optionnel) :
```bash
pip install maturin
cd accelerators/rust/hyprl_accel
maturin develop --release
```

## 4. Workflows & Commandes Clés
| Besoin | Commande | Notes |
| --- | --- | --- |
| Menu interactif (Linux/macOS) | `bash scripts/hyprl_launcher.sh` | Lance un menu couvrant GUI, replay, analysis/backtest paramétrés, pytest et commandes custom (requiert `.venv`). |
| Menu interactif (Windows) | `scripts\hyprl_launcher.bat` | Équivalent batch avec invites pour tickers/périodes/seuils et raccourcis GUI/tests. |
| Analyse rapide | `python scripts/run_analysis.py --ticker AAPL --period 5d` | Télécharge OHLCV, calcule features, affiche proba + plan trade. |
| Backtest mono-actif | `python scripts/run_backtest.py --ticker AAPL --period 1y --initial-balance 10000 --seed 42 --long-threshold 0.60 --short-threshold 0.40 --export-trades data/trades_AAPL_1y.csv` | Inclut EV filter, ATR sizing, benchmark buy&hold, export CSV. |
| Mode adaptatif | ajouter `--adaptive --adaptive-lookback 30` + presets YAML (`adaptive:`). | Bascule safe/normal/aggressive selon drawdown. |
| Sweep de seuils | `python scripts/run_threshold_sweep.py --ticker AAPL --period 1y --initial-balance 10000 --seed 42` | Long grid, short depuis config. |
| Sweep multi-tickers | `python scripts/run_universe_sweep.py --tickers AAPL,MSFT,NVDA --period 1y --initial-balance 10000 --seed 42 --output data/universe_1y.csv` | Produit CSV comparatif. |
| Supersearch (recherche) | `python scripts/run_supersearch.py --ticker AAPL --tickers AAPL,MSFT,GOOGL --period 1y --interval 1h --initial-balance 10000 --seed 42 --long-thresholds "0.5,0.55,0.6" --short-thresholds "0.35,0.4" --risk-pcts "0.01,0.015,0.02" --sentiment-min-values "-0.4,-0.2" --sentiment-max-values "0.2,0.5" --sentiment-regimes "off,neutral_only" --min-trades 50 --min-pf 1.2 --min-sharpe 0.8 --max-dd 0.35 --max-ror 0.1 --min-portfolio-pf 1.2 --min-portfolio-sharpe 0.8 --max-portfolio-dd 0.35 --max-portfolio-ror 0.1 --max-correlation 0.8 --engine auto --output data/supersearch_AAPL_1y.csv` | Balaye (seuils × risque + régimes sentiment) avec filtres PF/Sharpe/DD/RoR, puis agrège un portefeuille multi-tickers; engine `auto` ⇒ natif si dispo, sinon fallback Python. |
| Paper trading (replay) | `python scripts/run_paper_trading.py --tickers "AAPL,MSFT" --period 1y --interval 1h --initial-balance 10000 --config-csv data/supersearch_portfolio_AAPL_MSFT_1y.csv --config-index 0 --engine auto` | Rejoue une stratégie (CSV Supersearch) dans un broker mémoire, écrit `data/live/sessions/<session>/`. |
| Dashboard live | `streamlit run scripts/hyprl_dashboard.py` | Visualise equity/trades d'une session paper trading. |
| Panel Phase 1 | `python scripts/build_phase1_panel.py --csv-paths "data/supersearch_portfolio_AAPL_MSFT_GOOGL_1y.csv,data/supersearch_portfolio_BTC_ETH_1y.csv" --max-strategies 5` | Filtre les stratégies (PF/Sharpe/DD/RoR/corr) et produit `docs/experiments/PHASE1_PANEL.csv`. |
| Sessions Phase 1 | `python scripts/run_phase1_experiments.py --period 1y --interval 1h --initial-balance 10000` | Rejoue chaque ligne du panel via `PaperBroker`, loggue sous `data/live/sessions/<session_id>/`, écrit `docs/experiments/PHASE1_SESSIONS.csv`. |
| Analyse Phase 1 | `python scripts/analyze_phase1_results.py` | Compare backtest vs live (PF/Sharpe/DD/expectancy, ratios, robustness_score) et exporte `data/experiments/phase1_results.csv`. |
| Comparaison modèles | `python scripts/compare_models.py --ticker NVDA --period 1y --interval 1h --risk-profile normal --output data/nvda_model_compare.csv` | Logistic vs RandomForest + calibrations. |
| Analyse de trades | `python scripts/analyze_trades.py --trades data/trades_AAPL_1y.csv` | Stats (Sharpe, bins de calibration, PF). |
| GUI principale | `streamlit run scripts/hyprl_gui.py` | Exécution/backtest interactif, presets, toggle adaptive. |
| GUI Replay | `streamlit run scripts/hyprl_replay_gui.py` | Recharge CSV trades, slider temporel, snapshots zip. |

## 5. Pipeline Interne
1. `MarketDataFetcher` (yfinance) → OHLCV aligné + timezone fix.  
2. `indicators` → features vectorisées (SMA/EMA ratios, RSI, ATR, vol, returns).  
3. Sentiment stub (placeholder pour future intégration).  
4. `model` (LogisticRegression + StandardScaler) ou `random_forest`, calibrations (`--calibration platt|isotonic`).  
5. `risk` module → ATR risk_pct, reward multiples, EV filter, stop/TP dynamiques.  
6. `backtest` → walk-forward, coûts, slippage, regime reports, trade log CSV.  
7. CLI/GUI orchestrent via `AnalysisPipeline`.

## 6. Tests & Validation
- Lancer `source .venv/bin/activate && pytest -q` (ou `./scripts/run_tests.sh`).  
- Ruff + Black : `source .venv/bin/activate && ruff check src tests` et `source .venv/bin/activate && black src tests`.  
- Scénarios recommandés avant export :  
  1. `pytest -q` (sanity).  
  2. `python scripts/run_analysis.py --ticker AAPL --period 5d`.  
  3. `python scripts/run_backtest.py --ticker AAPL --period 3mo --long-threshold 0.55 --short-threshold 0.40 --initial-balance 5000 --seed 1 --export-trades data/trades_smoke.csv`.  
  4. `streamlit run scripts/hyprl_gui.py` (vérifier que la page monte).  
  5. `pytest tests/backtest/test_supercalc_native.py -q` (cohérence Python ↔ Rust pour Supercalc).  
  6. `pytest tests/risk/test_risk_metrics.py -q` (Risk Layer).  
  7. `pytest tests/portfolio/test_portfolio_core.py -q` (Portfolio Layer).  
  8. `pytest tests/execution/test_paper_broker.py tests/execution/test_execution_engine.py -q` (Execution Layer).  
- Couverture visée : ≥85 % sur `model` et `pipeline`.

### Risk Layer
- `hyprl/risk/metrics.py` fournit expectancy, variance, risk-of-ruin et bootstrap Monte Carlo.
- `run_search` rejette automatiquement les configs trop fragiles (min trades, PF, Sharpe, max DD, RoR, expectancy).
- Les CSV/Top-N affichent `expectancy_per_trade`, `risk_of_ruin`, `maxdd_p95`, `pnl_p05` pour suivre la robustesse.

## 7. Statut & Avancée (V1.2)
- Baseline déterministe validée (AAPL 1h logistic/platt, PF≈1.02, Sharpe≈0.19, 58 trades, drawdown ≈17.9 %).  
- Adaptive mode expérimental (AAPL) : ~5.9 % strat, PF≈1.17, Sharpe≈0.47, DD≈13.6 %.  
- Universe v1.2 (`data/universe_scores_v1_2.csv`) gelé : AAPL tradable, MSFT/NVDA en recherche.  
- Gap identifié : sous-performance vs bench sur marchés haussiers soutenus. Priorités : calibration avancée, features sentiment réels, moteurs RF/XGBoost, portage C++/Rust pour hot paths.

## 8. Conseils d’Export
- Inclure `.env` ou configs sensibles séparément (ne sont pas dans Git).  
- Conserver `data/` pour les CSV produits (sweeps, trades) afin de rejouer les analyses offline.  
- Vérifier la cohérence via `git status` avant copie; archiver sous `hyprl_<date>.tar.gz` si besoin :  
  ```bash
  tar czf hyprl_2025-11-10.tar.gz \
    AGENTS.md README.md pyproject.toml requirements.txt \
    src tests scripts configs accelerators data docs
  ```
- Copier l’archive + ce document + éventuels notebooks ou rapports externes sur la clé.

## 9. FAQ Rapide
- **Pourquoi venv/pip ?** Simplicité multiplateforme, contrôle explicite via `requirements.txt`.  
- **Peut-on brancher d’autres sources données ?** Oui via nouveaux fetchers dans `src/hyprl/data/`.  
|- **GPU/CUDA ?** Prévu via `accelerators/`, fallback Python par défaut.  
- **Live trading ?** Non, V1 = recherche/backtest uniquement.  
- **Logs & exports ?** `data/` accumule CSV (trades, sweeps); conserver pour audit.

## 10. Checklist avant remise
1. `source .venv/bin/activate && pytest -q` OK.  
2. `python scripts/run_analysis.py --ticker AAPL --period 5d` OK.  
3. `python scripts/run_backtest.py --ticker AAPL --period 1y --seed 42 --export-trades data/trades_AAPL_1y.csv` OK.  
4. `streamlit run scripts/hyprl_gui.py` lance sans erreur.  
5. Mettre à jour `docs/hyprl_export_overview.md` si nouveaux modules/CLIs.  
6. Copier archive + CSV critiques (trades, universe, supersearch) + configs sur la clé.

---
Contact : Kyo (HyprL). Toute modification majeure doit être notée dans `PROJECT_BRAIN.md`.
