# HyprL V1.1.1 Snapshot

Technical recap of the HyprL baseline as of release **V1.1.1**. This version focuses on a deterministic signal → risk → backtest engine suitable for research workflows (not live trading).

## 1. Core Capabilities
- **Ingestion** – `MarketDataFetcher` pulls OHLCV via yfinance with timezone fixes and column normalization; `NewsFetcher` + `SentimentScorer` stub sentiment headlines for optional features.
- **Feature stack** – `compute_feature_frame` builds SMA/EMA ratios, RSI, ATR-normalized range/volatility, rolling returns, and forward-return targets; `enrich_sentiment_features` injects sentiment-derived deltas.
- **Probability models** – `ProbabilityModel` wraps logistic regression (default) with deterministic scaling plus optional `random_forest` backend and Platt/Isotonic calibration flags.
- **Risk layer** – `RiskConfig` + `plan_trade` size positions via ATR multiples, enforce R-multiples, EV filters, and commission/slippage assumptions.
- **Backtest engine** – `BacktestRunner` performs walk-forward simulation with adaptive profiles, regime overrides, sentiment gating, benchmark curve, trade log, calibration metrics, and CSV export hooks.
- **Search & analytics** – Supercalc dataset pipelines feed `run_supersearch.py` pour les sweeps, registry Meta-ML + aliases (`scripts/registry_set_alias.py`, `scripts/registry_list.py`), calibrateurs (`scripts/calibrate_meta_predictions.py`), autorank constraints + Phase-1 auto-shortlist (`scripts/run_phase1_from_autorank.py`).
- **Execution & analysis** – Paper broker + dashboards/replay GUIs, Phase-1 orchestrateur, et **Realtime MVP (increment 1)** : `scripts/run_realtime_paper.py` utilise les features techniques HyprL + `ProbabilityModel` (Meta-ML optionnel), applique clamps légers (`--max-orders-per-min`, `--per-symbol-cap`, `--min-qty/--max-qty`), loggue `reason`/`meta_pred` dans `predictions.jsonl`, et `scripts/analyze_live_session.py` fournit PF/Sharpe/DD + exposure/hold/winrate/répartition des rejets.

## 2. Module Map
| Domain | Key Modules | Notes |
| --- | --- | --- |
| Data & Sentiment | `src/hyprl/data/market.py`, `src/hyprl/data/news.py`, `src/hyprl/sentiment/analyzer.py` | Download, cache, headline sentiment stub. |
| Indicators & Features | `src/hyprl/indicators/technical.py`, `src/hyprl/features/sentiment.py` | Vectorized technical frame + sentiment enrichments. |
| Modeling | `src/hyprl/model/probability.py`, `src/hyprl/model/ensemble.py` | Logistic default plus RF/ensemble scaffolding with calibration utilities. |
| Risk & Metrics | `src/hyprl/risk/manager.py`, `src/hyprl/risk/gates.py`, `src/hyprl/risk/metrics.py` | ATR sizing, EV checks, CVaR estimation, bootstrap drawdowns, risk-of-ruin calculators. |
| Adaptive/Regimes | `src/hyprl/adaptive/engine.py`, `src/hyprl/regimes/registry.py` | Regime definitions (safe/normal/aggressive) with overrides for risk, thresholds, and model choice. |
| Backtest & Snapshots | `src/hyprl/backtest/runner.py`, `src/hyprl/snapshots.py` | Walk-forward simulation, trade struct, universe sweeps, snapshot exports. |
| Search & Portfolio | `src/hyprl/search/*`, `src/hyprl/portfolio/core.py`, `src/hyprl/universe/` | Grid/portfolio optimizer, multi-ticker alignment, correlation-limited portfolios, weighting schemes (equal / inverse-vol). |
| Execution & Analysis | `src/hyprl/execution/*`, `src/hyprl/analysis/*` | Paper trading broker, dashboards, phase-1 analytics. |

## 3. CLI & Workflow Coverage
- `run_analysis.py` – single-ticker analysis with sentiment stub and ATR plan.
- `run_backtest.py` – deterministic walk-forward with EV/trend filters, adaptive toggle, CSV export.
- `run_threshold_sweep.py`, `run_universe_sweep.py` – threshold grids & multi-ticker comparison CSVs.
- `run_supersearch.py` – research-grade grid search spanning thresholds × risk × sentiment regimes (supports Rust `hyprl_supercalc` acceleration) with `--weighting-scheme {equal,inv_vol}` to control portfolio weights (inv_vol ⇒ weights ∝ 1/σ_20) and optional meta-robustness re-ranking via `--meta-robustness path/to/model.joblib --meta-weight 0.4`.
- `autorank_supersearch.py` – CI/offline utility to reload existing Supersearch CSVs, recompute meta predictions, and emit `*_autoranked.csv` + `*_autoranked.SUMMARY.txt` with top-K diagnostics.

Example:
```bash
python scripts/run_supersearch.py --tickers "AAPL,MSFT,GOOGL" \
  --ticker AAPL --period 1y --interval 1h \
  --weighting-scheme inv_vol \
  --min-trades 50 --min-pf 1.2 --min-sharpe 0.8 \
  --max-dd 0.35 --max-ror 0.1 --min-expectancy 0.0 \
  --engine auto --output data/supersearch_portfolio_AAPL_MSFT_GOOGL_1y.csv
```
- `analyze_trades.py`, `compare_models.py` – trade log statistics and model benchmarking.
- `run_paper_trading.py`, `run_phase1_experiments.py`, `analyze_phase1_results.py` – paper execution workflow + robustness analysis.
- `build_phase1_panel.py` – Phase 1 shortlist builder with tunable robustness filters (`--min-portfolio-pf`, `--min-portfolio-sharpe`, `--max-portfolio-dd`, etc.); defaults (PF≥1.05, Sharpe≥0.5, DD≤40 %, RoR≤20 %, trades≥30, corr≤0.90) ensure non-empty panels and print diagnostics if nothing survives.
- `analyze_phase1_results.py` + `export_meta_robustness_dataset.py` – compute advanced live/backtest ratios (PF/Sharpe/DD/winrate/equity vol) with a bounded robustness_score and export `data/experiments/meta_robustness_dataset.csv` for future Meta-ML (no training in V1.1.1).
- `train_meta_robustness.py` / `autorank_supersearch.py` / `hyprl_meta_dashboard.py` – Meta-ML v0 workflow covering training, CI autoranking, and Streamlit diagnostics (scatter base vs meta, filters, export autoranked).
- Streamlit apps: `hyprl_gui.py`, `hyprl_replay_gui.py`, `hyprl_dashboard.py`.
- Launcher scripts (`hyprl_launcher.sh/.bat`) wrap the above with venv activation.

## 4. Validation & Testing
- Pytest suites mirror the package tree (`tests/backtest`, `tests/risk`, `tests/portfolio`, etc.), covering runner metrics, adaptive regimes, supercalc parity, config parsing, paper broker, and WFO splits.
- Ruff & Black enforced via CI; recommended pre-flight: `ruff check src tests`, `black src tests`, then `pytest -q`.
- Target coverage: ≥85 % on `model` and `pipeline`; regression tests added for schema, risk, and accelerator changes.

## 5. Reference Metrics (V1.1.1 Baselines)
- **AAPL 1y @1h (non-adaptive)** – logistic+Platt, `short=0.40`, `long=0.52`; PF≈1.02, Sharpe≈0.19, max DD≈17.9 %, 58 trades (seed 42). Adaptive mode optional (≈5.9 % strat, PF≈1.17, Sharpe≈0.47, DD≈13.6 %).
- **NVDA 1y @1h** – logistic seed 42, `long=0.55`, `short=0.40`; Sharpe≈1.5, PF>1, yet still below buy-and-hold on strong uptrends.
- **Universe snapshot** – `data/universe_scores_v1_2.csv` tags AAPL as tradable, MSFT/NVDA research-only (PF<1, negative alpha).

## 6. Known Gaps / Next Focus
1. Improve probability calibration and feature richness (true sentiment feeds, alternative models such as XGBoost/NN).
2. Expand accelerators (pybind11/pyo3) for hot loops (feature calc, Monte Carlo, supercalc) to reduce runtime for large sweeps.
3. Enhance adaptive regime logic (drawdown-aware overrides, auto-threshold tuning) and validate with additional paper trading logs.
4. Bridge performance gap vs. benchmarks on persistent bull runs (trend filters, volatility regime gating, macro signals).

## 7. Deliverables
- Codebase (src, scripts, configs, tests, docs) plus generated CSV artifacts under `data/`.
- README + AGENTS instructions for operators, this V1.1.1 snapshot, and `docs/hyprl_export_overview.md` for USB exports.
- Optional Rust `hyprl_supercalc` build assets and paper trading logs (`data/live/sessions/*`) for reproducibility.

---
Contacts: Kyo (HyprL). Use `PROJECT_BRAIN.md` for ongoing decisions/logs after further iterations.
