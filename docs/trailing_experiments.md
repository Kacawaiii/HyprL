# Trailing Stop Experiments — NVDA Protocol v1

## Setup commun
- Ticker: `NVDA`
- Period: `1y`
- Interval: `1h`
- Initial balance: `10_000`
- Seed: `42`
- Stops de base: `stop_loss = 1.0R`, `take_profit = 2.0R`
- Export: `--export-trades data/trades_NVDA_*.csv`

## Modes testés
| Alias | Trailing activation (R) | Trailing distance (R) | Notes |
|-------|------------------------|-----------------------|-------|
| `nvda_trailing_off_baseline` | off | off | baseline sans trailing |
| `nvda_trailing_safe_1_0_0_5` | 1.0 | 0.5 | défensif (lock-in après 1R) |
| `nvda_trailing_aggr_0_8_0_4` | 0.8 | 0.4 | agressif (proche, capture rapide) |
| `nvda_trailing_let_run_1_5_0_75` | 1.5 | 0.75 | laisse courir jusqu'à 1.5R |

## Metrics à logger
- Profit Factor (PF)
- Sharpe Ratio
- Max Drawdown %
- Risk of Ruin (RoR)
- Expectancy (en R)
- Mix des exits (% stop_loss / take_profit / trailing_stop / close_end)

## Interprétation des résultats
- **Trailing défensif utile** : PF ≈ baseline ou ↑, Sharpe ↑, MaxDD ou RoR ↓, %trailing ≥ 10–20 % → bon preset "safe".
- **Trailing offensif utile** : PF ↑ (ou Expectancy ↑), Sharpe ↑, MaxDD ≈ identique ou hausse acceptable → preset "agressif".
- **Trailing inutile** : PF et Sharpe ≈ identiques, %trailing < 5–10 % → feature optionnelle seulement.
- **Trailing toxique** : PF ↓, Expectancy ↓, %stop_loss ↑ fort, %take_profit ↓ → marquer comme preset déconseillé.

## Commands (exemples)
```bash
python scripts/run_backtest.py \
  --ticker NVDA \
  --period 1y \
  --interval 1h \
  --initial-balance 10000 \
  --seed 42 \
  --export-trades data/trades_NVDA_1y_trailing_off.csv

python scripts/run_backtest.py \
  --ticker NVDA \
  --period 1y \
  --interval 1h \
  --initial-balance 10000 \
  --seed 42 \
  --trailing-stop-activation 1.0 \
  --trailing-stop-distance 0.5 \
  --export-trades data/trades_NVDA_1y_trailing_1_0_0_5.csv

python scripts/analyze_trades.py \
  --trades data/trades_NVDA_1y_trailing_1_0_0_5.csv
```

## Exploitation
- Presets prêts: `nvda_trailing_off_baseline`, `nvda_trailing_safe_1_0_0_5`, `nvda_trailing_aggr_0_8_0_4`, `nvda_trailing_let_run_1_5_0_75`.
- API/Discord: retourner `exit_reason_counts` + pourcentages pour illustrer l'usage du trailing (`/preset-info nvda_trailing_safe_1_0_0_5`).
- Option bonus: script `run_trailing_sweep.py` pour automatiser la grille et produire une table PF/Sharpe/DD + mix d'exits.
