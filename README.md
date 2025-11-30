# HyprL – V1 Signal & Backtest Engine

HyprL is a research-grade trading analysis engine that ingests OHLCV (and sentiment stubs), builds multi-timeframe indicators, trains a probabilistic model (logistic regression), and simulates trades with realistic ATR-based risk, commissions, slippage, and buy‑and‑hold benchmarks. V1 is a clean reference implementation for quant experimentation—not a production trading bot.

## Tech stack and entry points
- Language: Python (requires 3.11+)
- Package manager: pip + editable install via setuptools (pyproject.toml)
- Core libs: numpy, pandas, yfinance, scikit-learn, ta, vaderSentiment
- Optional apps: Streamlit (GUIs)
- API (present in repo): FastAPI + Uvicorn scaffolding under `api/` (offline tests included). See TODO below for serving instructions.
- Optional native accelerators:
  - Rust module via maturin in `accelerators/rust/hyprl_accel` (optional)
  - Native supercalc for search via `scripts/build_supercalc.sh` (optional)

Primary entry points (CLIs in scripts/):
- scripts/run_analysis.py — one-shot indicator/model analysis
- scripts/run_backtest.py — single-ticker backtest with costs + benchmark
- scripts/run_threshold_sweep.py — sweep long thresholds (short from config)
- scripts/run_universe_sweep.py — multi-ticker sweep summary
- scripts/run_supersearch.py — research grid search with risk/portfolio filters
- scripts/analyze_trades.py — analyze exported trades CSV
- scripts/compare_models.py — baseline vs alternative models
- scripts/run_paper_trading.py — paper-trading sessions from configs
- scripts/run_phase1_experiments.py, run_phase1_from_autorank.py — research workflows
- GUIs: scripts/hyprl_gui.py, scripts/hyprl_replay_gui.py, scripts/hyprl_dashboard.py (Streamlit)
- Launchers: scripts/hyprl_launcher.sh (Unix), scripts/hyprl_launcher.bat (Windows)

## Requirements
- Python 3.11 or newer
- pip 23+ recommended
- OS: Linux, macOS, or Windows (WSL supported). Streamlit GUIs require a desktop/browser.
- Optional:
  - Rust toolchain + maturin if you want the accelerators/supercalc
  - Redis if you intend to use the optional API rate-limit via Redis (see `api/`)

## Project Layout
- `src/hyprl/data/` – `MarketDataFetcher` (yfinance download, cache, timezone fixes, column flattening).
- `src/hyprl/indicators/` – enriched technical feature frame (SMA/EMA ratios, RSI, ATR, volatility, rolling returns).
- `src/hyprl/model/` – logistic regression probability model with StandardScaler + deterministic random_state.
- `src/hyprl/risk/` – ATR-based `RiskConfig` and `plan_trade` sizing (stops, take-profit, risk_pct).
- `src/hyprl/pipeline.py` – single-run `AnalysisPipeline` combining indicators, sentiment stub, model, risk plan.
- `src/hyprl/backtest/` – walk-forward engine with thresholds, EV filter, trade logging, costs, benchmark.
- `scripts/` – CLI utilities (analysis, backtest, threshold sweep, universe sweep, trade analysis).
- `configs/` – optional per-ticker YAML overrides (e.g., threshold baselines).

## Installation and setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Optional accelerator (Rust + maturin):
```bash
pip install maturin
cd accelerators/rust/hyprl_accel
maturin develop --release
```

Native supercalc (Rust, optionnel pour run_supersearch) :
```bash
pip install maturin
bash scripts/build_supercalc.sh
```

Notes:
- The repository also includes a FastAPI-based API under `api/` with offline tests. Serving details are project-specific and not finalized here. TODO: add API run instructions (uvicorn command, env vars, DB setup) when stabilized.

## Quickstart
```bash
# 1) Single analysis
python scripts/run_analysis.py --ticker AAPL --period 5d

# 2) Backtest with costs + benchmark + optional CSV
python scripts/run_backtest.py \
  --ticker AAPL \
  --period 1y \
  --initial-balance 10000 \
  --seed 42 \
  --long-threshold 0.60 \
  --short-threshold 0.40 \
  --export-trades data/trades_AAPL_1y.csv

# 3) Threshold sweep (long threshold grid, short threshold from config)
python scripts/run_threshold_sweep.py \
  --ticker AAPL \
  --period 1y \
  --initial-balance 10000 \
  --seed 42

# 4) Universe sweep (multiple tickers, CSV summary)
python scripts/run_universe_sweep.py \
  --tickers AAPL,MSFT,NVDA,TSLA \
  --period 1y \
  --initial-balance 10000 \
  --seed 42 \
  --output data/universe_1y.csv

# 5) Supersearch (research-only grid search)
python scripts/run_supersearch.py \
  --ticker AAPL \
  --tickers AAPL,MSFT,GOOGL \
  --period 1y \
  --interval 1h \
  --initial-balance 10000 \
  --seed 42 \
  --long-thresholds "0.50,0.55,0.60" \
  --short-thresholds "0.35,0.40" \
  --risk-pcts "0.010,0.015,0.020" \
  --sentiment-min-values "-0.4,-0.2" \
  --sentiment-max-values "0.2,0.5" \
  --sentiment-regimes "off,neutral_only" \
  --min-trades 50 \
  --min-pf 1.2 \
  --min-sharpe 0.8 \
  --max-dd 0.35 \
  --max-ror 0.1 \
  --min-expectancy 0.0 \
  --min-portfolio-pf 1.2 \
  --min-portfolio-sharpe 0.8 \
  --max-portfolio-dd 0.35 \
  --max-portfolio-ror 0.1 \
  --max-correlation 0.8 \
  --engine auto \
  --output data/supersearch_AAPL_1y.csv

# 6) Trade analysis (after exporting trades)
python scripts/analyze_trades.py --trades data/trades_AAPL_1y.csv

# Optional: alternative model + probability calibration
python scripts/run_backtest.py \
  --ticker NVDA \
  --period 1y \
  --initial-balance 10000 \
  --seed 42 \
  --long-threshold 0.55 \
  --short-threshold 0.40 \
  --model-type random_forest \
  --calibration platt

# 7) GUI (optional, Streamlit)
pip install streamlit
streamlit run scripts/hyprl_gui.py

# 8) Market Replay GUI
streamlit run scripts/hyprl_replay_gui.py
``` 
Replay app:
- Mode simulation (HyprL engine, risk profiles Safe/Normal/Aggressive, metrics + trades preview, snapshot export).
- Mode replay CSV (charger un trades.csv existant, reconstruire equity, slider temporel, snapshot).

### Interactive Launcher
```bash
# Linux/macOS
bash scripts/hyprl_launcher.sh

# Windows (PowerShell/CMD)
scripts\hyprl_launcher.bat
```
Menu options couvrent GUI, replay, analysis/backtest interactifs (prompts pour tickers/périodes/seuils), pytest rapide et commandes personnalisées (toutes exécutées avec l'environnement `.venv` actif).

# 9) Model Comparison (NVDA example)
```bash
python scripts/compare_models.py --ticker NVDA --period 1y --interval 1h --risk-profile normal --output data/nvda_model_compare.csv
```

## Scripts overview
- Analysis and backtests:
  - run_analysis.py, run_backtest.py, analyze_trades.py
- Research sweeps and search:
  - run_threshold_sweep.py, run_universe_sweep.py, run_supersearch.py, compare_models.py
- Paper trading and phase workflows:
  - run_paper_trading.py, run_phase1_experiments.py, run_phase1_from_autorank.py, analyze_phase1_results.py
- Real-time (experimental):
  - run_realtime_paper.py
- GUIs:
  - hyprl_gui.py, hyprl_replay_gui.py, hyprl_dashboard.py (Streamlit)
- Helpers:
  - hyprl_launcher.sh (Unix), hyprl_launcher.bat (Windows), run_tests.sh, build_supercalc.sh, gen_universe_table.py

## Tests
Activez l'environnement virtuel puis lancez pytest :

```bash
source .venv/bin/activate
pytest -q
```

Pour viser un jeu précis (ex. seuils configs) :

```bash
source .venv/bin/activate
pytest tests/configs/test_configs.py -k threshold -q
```

Un wrapper est disponible :

```bash
./scripts/run_tests.sh                     # pytest -q par défaut
./scripts/run_tests.sh tests/configs/test_configs.py -k threshold -q
```
Windows:
```powershell
.\.venv\Scripts\Activate.ps1
pytest -q
```

## Environment variables
These environment variables are read by different modules. Set them as needed.

- Data caching:
  - HYPRL_PRICE_CACHE_DIR — override price cache directory (default under data/)
  - HYPRL_PRICE_CACHE_MAX_AGE — max cache age in seconds (default 24h)
  - HYPRL_YFINANCE_CACHE_DIR — override yfinance cache directory
- API auth and dev:
  - HYPRL_DEV_TOKEN — development token used in offline tests/clients
  - HYPRL_ADMIN_TOKEN — optional admin bootstrap token for API (if unset, a dev fallback is used in dev utilities)
  - HYPRL_PREDICT_IMPL — select predict implementation for API stub (default: "stub")
  - REDIS_URL — optional Redis URL for rate limiting (API)
- Broker integrations (optional):
  - ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY — Alpaca API credentials
  - ALPACA_BASE_URL — Alpaca REST base URL (default paper endpoint)
  - ALPACA_DATA_WS — Alpaca data websocket URL

If you don’t use the API or broker connectors, you can ignore the related variables.

## Project structure
High-level directories in this repo:

```
HyprL/
  src/hyprl/               # core library: data, indicators, model, risk, backtest, portfolio, etc.
  scripts/                 # CLI tools and GUIs (Streamlit)
  configs/                 # optional per-ticker YAML presets
  tests/                   # pytest test suite (unit/integration, backtests, configs)
  api/                     # FastAPI scaffolding, offline API tests, auth and usage models
  accelerators/            # optional Rust accelerators
  docs/                    # experiment outputs and notes (used by phase workflows)
  data/                    # outputs, caches, exported CSVs (gitignored in practice)
  logs/                    # run logs
  pyproject.toml           # package metadata (setuptools), Python 3.11+
  requirements.txt         # runtime and tool dependencies (FastAPI, Streamlit, core libs)
```

## Supercalc natif
`run_search` et `run_supersearch.py` détectent automatiquement le module Rust `hyprl_supercalc` (build via `scripts/build_supercalc.sh`).  
Modes disponibles :

- `--engine auto` : tente le moteur natif puis repasse en Python en cas d’échec.
- `--engine python` : force l’implémentation pure Python (plus lente, utile pour le debug).
- `--engine native` : exige le moteur Rust (erreur si absent).

Le dataset (probabilités, ATR, flags sentiment) est préparé une seule fois puis rejoué pour des milliers de candidats, ce qui accélère `run_supersearch` et tout flux qui fait appel à `hyprl.supercalc.evaluate_candidates`.

## Risk Layer & Robustesse
Supersearch applique désormais une couche de filtrage et de scoring orientée risque :

- **Hard constraints** configurables (`--min-trades`, `--min-pf`, `--min-sharpe`, `--max-dd`, `--max-ror`, `--min-expectancy`). Toute config qui ne respecte pas ces garde-fous est rejetée avant le ranking.
- **Metrics avancées** calculées à partir des `trade_returns` :
  - `expectancy_per_trade`,
  - approximation de Risk-of-Ruin via winrate × ratio gain/perte,
  - Monte Carlo bootstrap (quantiles `maxDD_p95`, `pnl_p05`, etc.).
- Le score multi-objectifs pénalise directement le risque de ruine, les drawdowns stressés et les comportements suspects (winrate irréaliste / PF bancal). L’objectif n’est pas de promettre “100 % de trades gagnants” mais de privilégier les combinaisons à espérance positive et risque maîtrisé.

## Portfolio Layer
- `run_search` peut lancer une même stratégie sur plusieurs tickers via `SearchConfig.tickers` (et `scripts/run_supersearch.py --tickers "AAPL,MSFT,GOOGL"`).
- `src/hyprl/portfolio/core.py` aligne les equities individuelles, applique les pondérations (égal par défaut) et calcule PF/Sharpe/DD/RoR à l’échelle portefeuille, plus les corrélations inter-tickers.
- Nouvelles contraintes : `--min-portfolio-pf`, `--min-portfolio-sharpe`, `--max-portfolio-dd`, `--max-portfolio-ror`, `--max-correlation`. Toute config trop concentrée ou trop corrélée est rejetée avant scoring.
- Les CSV/Top-N affichent désormais les métriques portefeuille (`portfolio_*`) et un bloc `per_ticker_details` pour inspecter contributions individuelles.
- Exemple :
  ```bash
  python scripts/run_supersearch.py \
    --ticker AAPL \
    --tickers AAPL,MSFT,GOOGL \
    --period 1y --interval 1h --initial-balance 10000 --seed 42 \
    --long-thresholds "0.55,0.6" --short-thresholds "0.35,0.4" \
    --risk-pcts "0.01,0.015" \
    --sentiment-min-values "-0.4,-0.2" --sentiment-max-values "0.2,0.5" \
    --sentiment-regimes "off,neutral_only" \
    --min-trades 50 --min-pf 1.2 --min-sharpe 0.8 \
    --max-dd 0.35 --max-ror 0.1 --min-expectancy 0.0 \
    --min-portfolio-pf 1.2 --min-portfolio-sharpe 0.8 \
  --max-portfolio-dd 0.35 --max-portfolio-ror 0.1 --max-correlation 0.8 \
  --engine auto \
  --output data/supersearch_portfolio_AAPL_MSFT_GOOGL_1y.csv
  ```

## Paper Trading
- `scripts/run_paper_trading.py` rejoue une stratégie (issue d'un CSV Supersearch) en mode **paper trading** via un broker mémoire.
- Les journaux sont écrits sous `data/live/sessions/<session_id>/` (trades + equity).
- Commande type :
  ```bash
  python scripts/run_paper_trading.py \
    --tickers "AAPL,MSFT" \
    --period 1y --interval 1h \
    --initial-balance 10000 \
    --config-csv data/supersearch_portfolio_AAPL_MSFT_1y.csv \
    --config-index 0 \
    --engine auto
  ```

- Dashboard Streamlit pour suivre une session (mêmes fichiers live) :
  ```bash
  streamlit run scripts/hyprl_dashboard.py
  ```

## Phase 1 – Validation scientifique
Pipeline complet pour confronter plusieurs portefeuilles au mode paper (replay) avec traces scientifiques :

1. **Filtrer un panel** (hard filters + ranking) :
   ```bash
   source .venv/bin/activate
   python scripts/build_phase1_panel.py \
     --csv-paths "data/supersearch_portfolio_AAPL_MSFT_GOOGL_1y.csv,data/supersearch_portfolio_BTC_ETH_1y.csv" \
     --max-strategies 5
   ```
   → écrit `docs/experiments/PHASE1_PANEL.csv`.

2. **Orchestrer les sessions paper** (une par ligne du panel) :
   ```bash
   source .venv/bin/activate
   python scripts/run_phase1_experiments.py \
     --period 1y --interval 1h --initial-balance 10000
   ```
   → rejoue chaque stratégie via `PaperBroker`, loggue les trades/equity sous `data/live/sessions/<session_id>/` et trace les sessions dans `docs/experiments/PHASE1_SESSIONS.csv`.

3. **Analyser les résultats live vs backtest** :
   ```bash
   source .venv/bin/activate
   python scripts/analyze_phase1_results.py
   ```
   → consolide `data/experiments/phase1_results.csv` avec PF/Sharpe/DD live, ratios backtest/live, et un `robustness_score` borné [0,1].

4. **Journal humain** : compléter `docs/experiments/PHASE1_LOG.md` (format libre) avec les observations hebdo.

Les scripts s'appuient sur `hyprl.analysis.phase1` pour :
- filtrer/score les candidats (`Phase1Filters`),
- recalculer PF/Sharpe/DD/expectancy à partir des logs paper,
- comparer systématiquement les métriques backtest vs live pour décider si une configuration reste en shortlist.

## License
No explicit LICENSE file was found at the repository root. TODO: add a LICENSE (e.g., MIT/Apache-2.0/BSD) or clarify the licensing terms for HyprL.

## Adaptive Mode (v1.2)
HyprL can shift risk settings, thresholds, and even model calibration based on recent performance. Define regimes inside a ticker preset:

```yaml
adaptive:
  enabled: true
  lookback_trades: 25
  default_regime: normal
  regimes:
    safe:
      min_equity_drawdown: 0.08
      max_equity_drawdown: 1.0
      risk_overrides:
        risk_pct: 0.01
        reward_multiple: 1.5
      threshold_overrides:
        long_shift: 0.02
        short_shift: -0.02
    normal:
      min_equity_drawdown: -0.01
      max_equity_drawdown: 0.08
    aggressive:
      min_equity_drawdown: -0.05
      max_equity_drawdown: -0.01
      risk_overrides:
        risk_pct: 0.025
      threshold_overrides:
        long_shift: -0.02
        short_shift: 0.02
```

Enable it from the CLI or GUI:

```bash
python scripts/run_backtest.py \
  --ticker AAPL \
  --period 1y \
  --initial-balance 10000 \
  --seed 42 \
  --adaptive \
  --adaptive-lookback 30
```

Results now include the final regime, number of regime switches, and per-regime usage summaries. Streamlit GUIs expose the same toggle plus lookback sliders.

## Reference Presets (v1.2)
- **AAPL 1h (tradable)** – logistic + platt, `short=0.40`, `long=0.52`, normal risk (risk_pct 1.5%, RR 2.0), EV≥20% risk, rolling-return trend filter. Non-adaptive baseline (seed 42, 1y@1h) → strat ≈1.9%, bench ≈18.0%, alpha ≈-16.1 pts, PF≈1.02, Sharpe≈0.19, max DD≈17.9%, 58 trades. Adaptive mode is opt-in (`--adaptive`) and yields ≈5.9% strat, PF≈1.17, Sharpe≈0.47, DD≈13.6% (29 trades, normal→safe once).
- **MSFT 1h (research-only)** – logistic + platt baseline (seed 42, 1y@1h) loses: strat ≈-5.8%, bench ≈17.2%, alpha ≈-22.9 pts, PF≈0.95, Sharpe≈-0.94, DD≈9.1%, 21 trades.
- **NVDA 1h (calibration-only)** – logistic + platt baseline (seed 42, 1y@1h) sits at strat ≈-1.5%, bench ≈28.0%, alpha ≈-29.5 pts, PF≈0.98, Sharpe≈-0.01, DD≈14.7%, 78 trades; aggressive threshold (0.65) produces ≈0.37% but only 2 trades, so keep for calibration.

Universe v1.2 (non-adaptive, 1y@1h, seed 42) summary:

<!-- BEGIN_UNIVERSE_V1_2 -->
Ticker | Tradable | Short | Best Long | Strat % | Ann % | Bench % | Alpha % | PF | Sharpe | Max DD % | Trades | Win % | Exp | Score | Best Regime
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
AAPL | YES | 0.40 | 0.52 | 1.89 | 1.89 | 17.99 | -16.10 | 1.02 | 0.19 | 17.87 | 58 | 43.10 | 3.26 | -0.09 | normal
MSFT | NO | 0.40 | 0.75 | -5.78 | -5.78 | 17.16 | -22.94 | 0.95 | -0.94 | 9.05 | 21 | 28.57 | -27.53 | -1.52 | normal
NVDA | NO | 0.40 | 0.65 | 0.37 | 0.37 | 27.97 | -27.60 | 1.22 | 0.11 | 1.68 | 2 | 50.00 | 18.64 | -0.00 | normal
<!-- END_UNIVERSE_V1_2 -->
```

## Current V1 Results (examples)
- **NVDA** (1y, 1h bars, long_th=0.55, short_th=0.40, seed=42):
  - Strategy return ≈ +19.9% vs buy & hold ≈ +28.0%.
  - Sharpe ≈ 1.53, max drawdown ≈ 6.4%, ~45% win rate, positive expectancy with EV filter.
- **AAPL** (1y, 1h, same thresholds): strategy slightly underperforms benchmark (~15% vs ~18%) but delivers Sharpe ~1.15 with realistic costs.
- Similar behavior on MSFT/TSLA (strategy near but below benchmark on strong uptrends). Treat V1 as a research tool to study driver metrics and alpha gaps.

## Limitations & Next Steps
- Single logistic regression baseline (no calibration, no ensemble).
- Backtest handles one instrument at a time (no portfolio interactions yet).
- No live order execution; EV filter is a heuristic gate.
- Use V1 for diagnostics, sweeps, and trade logging; future work should explore probability calibration, richer models (XGBoost/NN), and smarter risk/portfolio overlays.
