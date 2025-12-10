# HyprL Ascendant v2 – Rapport de performance
_Généré le 2025-11-30T23:03:05.748867_
Fenêtre: 2025-01-21 → 2025-02-28

## Résumé global
- PF_portfolio: 2.586
- MaxDD_portfolio: 2.02%
- Sharpe_portfolio: 12.17
- Trades_total: 224
- Equity initiale: 10000.00 | Equity finale: 12665.04

## Détail par ticker
| Ticker | Weight | PF | MaxDD% | Trades | Win% | Sharpe | Equity_end |
|--------|--------|----|--------|--------|------|--------|------------|
| NVDA | 0.3000 | 2.516 | 10.78 | 43 | 72.09% | 18.46 | 7384.78 |
| META | 0.0900 | 2.674 | 11.59 | 27 | 74.07% | 21.42 | 2242.87 |
| MSFT | 0.2700 | 4.122 | 7.75 | 37 | 81.08% | 23.99 | 5182.91 |
| AMD | 0.2700 | 2.172 | 11.85 | 36 | 63.89% | 14.11 | 4552.36 |
| QQQ | 0.0700 | 1.590 | 28.43 | 42 | 69.05% | 7.90 | 1531.76 |
| SPY | 0.0000 | 0.878 | 289.59 | 39 | 61.54% | 6.22 | -191.41 |

## Section métriques de risque
- Kelly dynamique (borne par caps).
- Caps typiques: max_total_risk_pct ~5%, max_ticker_risk_pct ~3%, max_group_risk_pct ~4%, max_positions ~5.
- Guards: max_drawdown_pct, min_pf_live, max_consecutive_losses (voir configs Ascendant v2).
- Gates visées: PF ≥ 1.5, MaxDD ≤ 15–20%, Sharpe ≥ 1.5 sur fenêtre roulante.

## Notes / Limitations
- Résultats issus de backtests/replays/live-paper selon la source des logs.
- Les performances passées ne préjugent pas des performances futures.