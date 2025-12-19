# Portfolio core 1h v3 — Factsheet (portfolio_core_1h_v3_gate2_oos_v1)

## Résumé exécutif
- PF_net full: 3.01 | Sharpe_net full: 7.32 | MaxDD_net: 4.53% | Trades: 1582
- Portefeuille 1h multi-titres (NVDA/MSFT/QQQ), coûts inclus (commission=slippage=0.0005/side).

## Setup technique
- Configs: configs/NVDA-1h_v3.yaml, configs/MSFT-1h_v3.yaml, configs/QQQ-1h_v3.yaml, configs/portfolio_core_1h_v3.yaml
- Annualisation Sharpe: 1638 (1h bars/an)
- Tag interne: portfolio_core_1h_v3_gate2_oos_v1

## Performance full période (net coûts base)
- PF_net: 3.01, Sharpe_net: 7.32, MaxDD_net: 4.53%, Trades: 1582
- Période: alignée aux configs (interval 1h, period ~730d)

## Validation OOS
- Fenêtre OOS: 2024-03-01 -> 2024-12-01
- OOS: PF_net 7.79, Sharpe_net 17.91, MaxDD_net 0.82%, Trades: 390

## Stabilité par sous-périodes (extrait)
| Segment | PF_net | Sharpe_net | MaxDD_net (%) | Trades |
|---|---|---|---|---|
| T1 | 3.18 | 17.46 | 0.60 | 29 |
| T2 | 8.20 | 20.64 | 0.67 | 155 |
| T3 | 7.70 | 21.81 | 0.58 | 155 |
| T4 | 6.49 | 21.73 | 0.96 | 162 |

## Sensibilité aux coûts (base)
| comm/slip per side | PF_net | Sharpe_net | MaxDD_net (%) | Trades |
|---|---|---|---|---|
| 0.0005 / 0.0005 | 3.01 | 7.32 | 4.53 | 1582 |
| 0.0010 / 0.0010 | 3.01 | 7.32 | 4.54 | 1582 |

### Revision 2 (2025-12-19)
- **Fix** : Replay feature contract aligned with v3 models
- **Validation** : Full portfolio + NVDA/QQQ/MSFT W0/W1/W2 reproduced
- **Metrics** : Identical to v1 (PF_net 3.01, Sharpe 7.32, MaxDD 4.53%)


### Revision 2 (2025-12-19)
- **Fix** : Replay feature contract aligned with v3 models
- **Validation** : Full portfolio + NVDA/QQQ/MSFT W0/W1/W2 reproduced
- **Metrics** : Identical to v1 (PF_net 3.01, Sharpe 7.32, MaxDD 4.53%)

