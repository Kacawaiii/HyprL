# Architecture – HyprL Ascendant v2

## Vue d’ensemble (flux logique)
- Data market (yfinance/cache, fenêtres start/end strictes) → feature builder v2 → modèle prob XGB → discrétisation (seuils longs/shorts) → risk layer (Kelly + caps + guards) → exécution (market/TWAP) → broker/logs (trades CSV/summary JSON) → monitor (PF/DD/Sharpe) → ops (concat, agrégation, alerts).

## Data / Market
- Backtest: `scripts/run_backtest.py`, prix 1h via `MarketDataFetcher` + cache local.
- Replay: `scripts/run_live_replay.py` (parité BT↔Replay) + orchestrateur `run_portfolio_replay_orch_cli.py` (multi-ticker).
- Live single-ticker: `scripts/run_live_hour.py` (start/end backfill respectés, warmup requis).
- Live-lite multi-ticker: runners single-ticker + agrégation offline (ops scripts), multi-ticker live runner historique laissé expérimental.

## Features & modèles
- Presets v2 (equity_v2, nvda_v2, amd_v2, etc.) incluant retours multi-horizon, ATR (14/72), RSI (7/14/21), vol regimes, volume z-score/surge, range, skew/kurt.
- Modèles XGBoost par ticker (artefacts `artifacts/*.pkl`, depth ~6, n_estimators ~800, eta ~0.03, seed 42), calibration optionnelle.
- Seuils longs/shorts par ticker, filtres éventuels (trend/EV) paramétrables en YAML.

## Risk layer
- Kelly dynamique (base_risk_pct, min/max multipliers, lookback trades) appliqué par ticker, borné par caps.
- Caps: max_total_risk_pct, max_ticker_risk_pct, max_group_risk_pct, max_positions, max_notional_per_trade.
- Guards: max_drawdown_pct, min_pf_live, max_consec_losses; portfolio guards optionnels.
- Portfolio risk manager: applique caps/guards; monitor en lecture pour PF/DD/Sharpe, pas de couplage au moteur de sizing.

## Execution
- Par défaut: exécution directe (market/stop/TP/trailing) via `run_live_hour.py`.
- Option TWAP: algo twap (num_slices=4, total_seconds=3600) validé en smoke NVDA/AMD (W0), OFF par défaut en live minimal.
- Broker paper: `PaperBrokerImpl`; logging trades CSV + summaries JSON; flatten en fin de fenêtre en backfill pour éviter drift.

## Monitoring & outils
- `scripts/monitor/monitor_portfolio_health.py`: PF, MaxDD, Sharpe, trades (rolling windows) sur CSV trades.
- `scripts/aggregate_portfolio_replay.py`: agrégation portefeuille depuis trades par ticker (replay/live-lite).
- Ops wrappers: runner horaire single-ticker (`run_live_single_ticker_hourly.py`), concat générique (`concat_trades_live_generic.py`), monitor portefeuille Asc v2 (`run_portfolio_monitor_live.py`).

## Orchestration
- Replay-orch CLI: exécute `run_live_replay.py` par ticker puis agrège.
- Live-lite: cron/systemd appelle runners single-ticker; concat nightly; agrégation/monitor portefeuille v2; multi-ticker live runner laissé expérimental.
