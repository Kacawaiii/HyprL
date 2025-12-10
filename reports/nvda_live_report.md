# HyprL Ascendant v2 – Rapport de performance
_Généré le 2025-12-01T10:30:27.171997+00:00_
Fenêtre: 2024-11-11 → 2025-11-28

## Résumé global
- PF_portfolio: 3.117
- MaxDD_portfolio: 24.48%
- Sharpe_portfolio: 12.39
- Trades_total: 234
- Equity initiale: 10000.00 | Equity finale: 85390.01

## Détail par ticker
| Ticker | Weight | PF | MaxDD% | Trades | Win% | Sharpe | Equity_end |
|--------|--------|----|--------|--------|------|--------|------------|
| NVDA | 1.0000 | 3.117 | 43.37 | 234 | 69.23% | 13.62 | 85390.01 |

## Section métriques de risque
- Kelly dynamique (borne par caps).
- Caps typiques: max_total_risk_pct ~5%, max_ticker_risk_pct ~3%, max_group_risk_pct ~4%, max_positions ~5.
- Guards: max_drawdown_pct, min_pf_live, max_consecutive_losses (voir configs Ascendant v2).
- Gates visées: PF ≥ 1.5, MaxDD ≤ 15–20%, Sharpe ≥ 1.5 sur fenêtre roulante.

## Notes / Limitations
- Résultats issus de backtests/replays/live-paper selon la source des logs.
- Les performances passées ne préjugent pas des performances futures.