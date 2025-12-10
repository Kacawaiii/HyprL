# Risk & Ops – HyprL Ascendant v2

## Paliers de risque
- Palier 1 (NVDA solo): Kelly ON, guards ON, TWAP OFF. Gates (rolling 120–200 trades): PF ≥1.5 (actuel ~3.7), MaxDD ≤15–20% (actuel ~11.8%), Sharpe ≥1.5.
- Palier 2 (multi-ticker live-lite Asc v2): NVDA/MSFT/AMD (+META/QQQ option). Gates (rolling 300–400 trades): PF_portfolio ≥1.5, MaxDD ≤15–20%, Sharpe ≥1.5.
- Ramp: augmenter caps uniquement après respect durable des gates (ex. cap total 0.5% → 0.75–1%).

## Kelly dynamique
- Base_risk_pct défini par ticker (config risk.risk_pct ou dynamic base, ex. NVDA base 0.5%).
- Multipliers min/max, lookback trades (ex. lookback 50, min_mult 0.4, max_mult 4.0).
- Appliqué par ticker, borné par caps/guards (override risk_pct mais jamais au-delà des plafonds).

## Caps & guards
- Caps: max_total_risk_pct, max_ticker_risk_pct, max_group_risk_pct, max_positions, max_notional_per_trade (ports YAML). Ex. NVDA config: max_drawdown_pct 15%, min_pf 1.2, max_consec_losses 8; portfolio v2 caps total 5%, ticker 3%, group 4%, positions 5.
- Guards: max_drawdown_pct, min_pf_live, max_consec_losses; appliqués par ticker et/ou portefeuille (bloque nouvelles entrées, log reason).
- Playbook ALERT: si PF<1.3 ou MaxDD>20% (ticker ou portefeuille) → réduire ou couper tickers drag (META/QQQ), revenir NVDA-centric, suspendre si nécessaire.

## Règles d’arrêt / Playbook
- Ticker ALERT: PF_ticker<1.0 ou DD_ticker>20% → pause ticker, analyse logs, re-run replay si besoin.
- Portefeuille ALERT: PF_portfolio<1.0 ou DD_portfolio>20% → couper META/QQQ, réduire MSFT/AMD, revenir NVDA-only jusqu’à retour dans les gates.
- Reprise: uniquement après retour dans les seuils sur fenêtre roulante et compréhension de la dérive.

## Runbook OPS (haute vue)
- Runners horaires (cron/systemd) par ticker via `run_live_hour.py` (wrappé par `run_live_single_ticker_hourly.py`) ou multi-ticker séquentiel via `run_live_multi_ticker_hourly.py`.
- Logs journaliers par ticker sous `live/logs/live_<ticker>/YYYY-MM-DD/`.
- Concat nightly par ticker (`concat_trades_live_generic.py`) → trades_<TICKER>_live_all.csv.
- Agrégation portefeuille (Asc v2) via `aggregate_portfolio_replay.py` ou wrapper monitor portefeuille live.
- Monitor nightly (NVDA + portefeuille) via `monitor_portfolio_health.py` ou `run_portfolio_monitor_live.py`.
- Rotation/archivage logs et health JSON/CSV; snapshots Git pour états stables (tag ascendant_v2_baseline).

## Alerting & heartbeat (Palier 2)
- Santé PF/DD/Sharpe : `scripts/ops/alert_portfolio_health.py --health live/logs/portfolio_live/health_asc_v2.json --pf-alert 1.3 --dd-alert 20 --sharpe-alert 1.5 --webhook $WEBHOOK`.
- Heartbeat runners : `scripts/ops/check_heartbeat.py --root live/logs --max-age-min 90 --symbols NVDA,MSFT,AMD,META,QQQ --webhook $WEBHOOK`.
- Exécution typique (cron 15–30 min, heures US) :
  - `run_portfolio_monitor_live.py` → produit health_asc_v2.json
  - `alert_portfolio_health.py` → alerte si PF<1.3 ou DD>20 % (exit non-zero)
  - `check_heartbeat.py` → alerte si heartbeat manquant/stale

## Palier 2 live-lite (per-ticker isolation, exemples cron)
- Runners horaires par ticker (exemple) :
  - `0 15-21 * * 1-5 cd /home/kyo/HyprL && . .venv/bin/activate && python scripts/ops/run_live_single_ticker_hourly.py --config configs/NVDA-1h_v2.yaml --log-root live/logs`
  - idem pour MSFT/AMD (option META/QQQ).
- Post-run monitor + alerting (toutes les 30 min) :
  - `python scripts/ops/run_portfolio_monitor_live.py --log-root live/logs --summary-out live/logs/portfolio_live/health_asc_v2.json`
  - `python scripts/ops/alert_portfolio_health.py --health live/logs/portfolio_live/health_asc_v2.json --pf-alert 1.3 --dd-alert 20 --sharpe-alert 1.5 --webhook $WEBHOOK`
  - `python scripts/ops/check_heartbeat.py --root live/logs --max-age-min 90 --symbols NVDA,MSFT,AMD,META,QQQ --webhook $WEBHOOK`
- Script helper : `scripts/ops/palier2_hourly.sh` enchaîne monitor + alert + heartbeat (variables ROOT/WEBHOOK/LOG_ROOT configurables).

## Track-record mensuel
- Génération des rapports mensuels :
  - `python scripts/ops/run_monthly_track_record.py --live-root live/logs --output-root reports --month 2025-01`
  - Produit md/html + JSON pour Palier1 NVDA (weights NVDA=1) et Palier2 Asc v2 (weights officiels).
- Cron exemple (dernier jour ouvré 23h) :
  - `0 23 28-31 * 1-5 cd /home/kyo/HyprL && . .venv/bin/activate && python scripts/ops/run_monthly_track_record.py --live-root live/logs --output-root reports`

## Runbook quotidien (palier + santé)
- Commande unique :  
  `python scripts/ops/run_daily_ops_checks.py --log-root live/logs --health live/logs/portfolio_live/health_asc_v2.json --symbols NVDA,MSFT,AMD,META,QQQ`
- Chaîne exécutée : `check_palier_status` → `alert_portfolio_health` (PF<1.3, DD>20, Sharpe<1.5) → `check_heartbeat` (max-age 90 min).
- Cron exemple (15–23 CET, lun–ven) :  
  `*/15 15-23 * * 1-5 cd /home/kyo/HyprL && set -a && . ./.env.ops && set +a && python scripts/ops/run_daily_ops_checks.py --log-root live/logs --health live/logs/portfolio_live/health_asc_v2.json --symbols NVDA,MSFT,AMD,META,QQQ >> live/logs/cron_daily_ops.log 2>&1`

## Broker dry-run (bridge /v2)
- Lancer l’API : `uvicorn api.app:app --host 0.0.0.0 --port 8000`
- Dry-run broker :  
  `python scripts/ops/run_broker_dryrun.py --tickers NVDA,MSFT,AMD,META,QQQ --api-base http://127.0.0.1:8000 --state-file live/logs/broker_state_dryrun.json --audit-file live/logs/audit_trades_dryrun.jsonl`
- Cron exemple (toutes les heures, 15–23 CET) :  
  `0 15-23 * * 1-5 cd /home/kyo/HyprL && set -a && . ./.env.ops && set +a && python scripts/ops/run_broker_dryrun.py --tickers NVDA,MSFT,AMD,META,QQQ --api-base http://127.0.0.1:8000 --state-file live/logs/broker_state_dryrun.json --audit-file live/logs/audit_trades_dryrun.jsonl >> live/logs/cron_broker_dryrun.log 2>&1`
