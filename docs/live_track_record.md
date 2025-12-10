# Live Track Record – HyprL Ascendant v2

## NVDA Palier 1 (live/paper)
- Collecte: `run_live_hour.py` via wrapper hourly, logs journaliers sous `live/logs/live_nvda/YYYY-MM-DD/`.
- Concat: `concat_trades_live_generic.py --symbol NVDA --input-dir live/logs/live_nvda --output live/logs/live_nvda/trades_NVDA_live_all.csv`.
- Monitor: `monitor_portfolio_health.py --trade-logs trades_NVDA_live_all.csv --weights NVDA=1.0 --initial-equity 10000 --annualization 1638`.
- Fenêtres roulantes recommandées: 120–200 trades. Gates: PF ≥1.5, MaxDD ≤15–20%, Sharpe ≥1.5.
- Statut actuel (Palier 1 live-like, ~120 trades): PF≈3.72, MaxDD≈11.84%, Sharpe≈15.33, Trades≈120 (STATUS OK).

## Multi-ticker live-lite (Asc v2, futur/progressif)
- Runners single-ticker (NVDA/MSFT/AMD, option META/QQQ).
- Concat par ticker: `concat_trades_live_generic.py` par symbole → trades_<TICKER>_live_all.csv.
- Agrégation portefeuille: `aggregate_portfolio_replay.py` ou monitor wrapper en mode live-lite avec poids v2.
- Monitor portefeuille: `monitor_portfolio_health.py --trade-logs <live_all csvs> --weights NVDA=0.30,MSFT=0.27,AMD=0.27,META=0.09,QQQ=0.07,SPY=0.00 --annualization 1638`.
- Fenêtres roulantes: 300–400 trades multi-tickers. Gates: PF_portfolio ≥1.5, MaxDD ≤15–20%, Sharpe ≥1.5.
- TODO: insérer premiers résultats live-lite dès qu’ils existent.
- Track-record mensuel (NVDA + Asc v2): `python scripts/ops/run_monthly_track_record.py --live-root live/logs --output-root reports --month YYYY-MM` (génère md/html + JSON).

## Scripts de référence (ops)
- `scripts/ops/run_live_single_ticker_hourly.py`: lance run_live_hour avec chemins datés.
- `scripts/tools/concat_trades_live_generic.py`: concat/dedup/sort trades live par ticker.
- `scripts/ops/run_portfolio_monitor_live.py`: lance le monitor portefeuille Asc v2 (weights fixes) sur logs live_all.
- `scripts/ops/run_monthly_track_record.py`: génère rapports NVDA et Asc v2 (md/html) + summary JSON par mois.
- `scripts/ops/alert_portfolio_health.py` / `scripts/ops/check_heartbeat.py`: alertes PF/DD/Sharpe/status et heartbeats via webhook.

## Checklist lecteur externe
- Vérifier PF/MaxDD/Sharpe sur fenêtres roulantes (ticker + portefeuille).
- Vérifier le volume de trades (≥150–200 NVDA, ≥300–400 multi-tickers) avant toute montée de risque.
- Vérifier que les ALERT monitor (PF<1.3 ou DD>20%) sont traitées (pause/coupure, revue).
- Vérifier cohérence des poids v2 (SPY=0) et des caps/guards activés.
