# HyprL – Ascendant v2 (1h signal engine + risk layer)

### Product snapshot
- Moteur 1h multi-tickers: NVDA, META, MSFT, AMD, QQQ (SPY=0 dans le profil Ascendant v2 officiel).
- NVDA 1h v2 (1 an BT/replay): PF ≈ 3.2–3.3, MaxDD ≈ 8%, ~330–340 trades, win ≈ 75% (historique/simulé, pas une garantie).
- Portefeuille Ascendant v2 (W0/W1/W2 replay-orch): PF > 1.3, MaxDD ≈ 2–4% (voir docs pour le détail).
- Risk intégré: Kelly dynamique borné, caps, guards; monitoring PF/MaxDD/Sharpe; TWAP optionnel (off par défaut).
- Live minimal NVDA (Palier 1) validé en paper: PF ~3.7, DD ~11.8%, Sharpe > 15 sur ~120 trades.
*Les chiffres sont historiques/simulés; aucune promesse de rendement réel.*

## Quickstart (dev)

### Installation (résumé)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Commandes clés (NVDA 1h v2, 1 an)
```bash
# Backtest 1 an
.venv/bin/python scripts/run_backtest.py \
  --config configs/NVDA-1h_v2.yaml \
  --start 2024-11-27 --end 2025-11-26 \
  --signal-log data/signals/nvda_bt_v2_repro.csv

# Replay 1 an (parity-mode)
.venv/bin/python scripts/run_live_replay.py \
  --config configs/NVDA-1h_v2.yaml \
  --start 2024-11-27 --end 2025-11-26 \
  --parity-mode \
  --signal-log data/signals/nvda_replay_v2_repro.csv

# Diff des signaux BT vs Replay
.venv/bin/python scripts/diff_signal_logs.py \
  --backtest-log data/signals/nvda_bt_v2_repro.csv \
  --replay-log data/signals/nvda_replay_v2_repro.csv \
  --output data/signals/nvda_diff_v2_repro.csv
```

## Reproduire les principaux résultats
- NVDA 1y BT/replay + diff (parité, PF~3.2–3.3, DD~8%, ~337 trades, 0 mismatch).
- Fenêtre W0 (2025-01-01 → 2025-03-01) par ticker via orchestrateur.
- Portefeuille Ascendant v2 W0/W1/W2 via agrégateur (PF>1.3, DD~2–4%).
- NVDA live-like paper (PF~1.55–1.60, DD~10–11%, Sharpe~7, ~170–190 trades).
- Troubleshooting (tolérances, caches, warmup).

[Guide complet de reproduction](docs/reproduce_core_results.md)

## Docs détaillées
- [Product overview](docs/product_overview.md)
- [Architecture](docs/architecture.md)
- [Risk & Ops](docs/risk_and_ops.md)
- [Performance](docs/performance.md)
- [Live track record](docs/live_track_record.md)
- [Reproduce core results](docs/reproduce_core_results.md)

## Remarques
- Profils officiels Ascendant v2: SPY poids 0, META/QQQ réduits, NVDA/MSFT/AMD dominants.
- Les runners/ops live-lite (hourly wrappers, concat, monitor) sont fournis; multi-ticker live runner historique reste expérimental.
- Reporting: `scripts/reporting/export_report.py` (MD/HTML, HTML nécessite `markdown`, warning si la somme des poids s’éloigne de 1.0).

## Ops Palier 2 (live-lite)
- Hourly multi-ticker (NVDA/MSFT/AMD par défaut) : `python scripts/ops/run_live_multi_ticker_hourly.py --log-root live/logs` (ou `--configs configs/NVDA-1h_v2.yaml configs/MSFT-1h_v2.yaml ...`)
- Concat par ticker : `python scripts/tools/concat_trades_live_generic.py --symbol NVDA --input-dir live/logs/live_nvda --output live/logs/live_nvda/trades_NVDA_live_all.csv`
- Monitor portefeuille Asc v2 : `python scripts/ops/run_portfolio_monitor_live.py --log-root live/logs --summary-out live/logs/portfolio_live/health_asc_v2.json`
- Rapport : `python scripts/reporting/export_report.py --trade-logs live/logs/live_nvda/trades_NVDA_live_all.csv --weights NVDA=1.0 --initial-equity 10000 --annualization 1638 --output reports/nvda_live_report.md --format md`
- Rapport portefeuille : `python scripts/reporting/export_report.py --trade-logs live/logs/portfolio/orch_cli_W0_repro/trades_NVDA_orch.csv ... --weights NVDA=0.30,MSFT=0.27,AMD=0.27,META=0.09,QQQ=0.07,SPY=0.00 --initial-equity 10000 --annualization 1638 --output reports/asc_v2_W0_report.md --format md`
