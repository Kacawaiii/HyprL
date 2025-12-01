# Performance – HyprL Ascendant v2

## Backtests/Replay v2 (par ticker, 1h)
- NVDA, MSFT, AMD, META, QQQ, SPY (v2 configs).
- Métriques clés: Profit Factor (PF), Max Drawdown (MaxDD), Sharpe/Sortino (ann=1638 pour 1h).
- Artefacts: XGB v2 sous `artifacts/*.pkl`; features presets equity_v2.

| Ticker | PF (W0 replay) | MaxDD % | Trades | Sharpe (si dispo) |
|--------|----------------|---------|--------|-------------------|
| NVDA   | ~2.52          | ~4.58   | 86     | TODO (non loggé)  |
| META   | ~2.67          | ~2.27   | 54     | TODO              |
| MSFT   | ~4.12          | ~2.36   | 74     | TODO              |
| AMD    | ~2.17          | ~4.77   | 72     | TODO              |
| QQQ    | ~1.59          | ~3.85   | 42     | TODO              |
| SPY    | ~0.88          | ~6.56   | 39     | TODO (drag → poids 0 v2) |

## Portefeuille Ascendant v2 (replay-orch)
- Poids v2: NVDA 0.30, MSFT 0.27, AMD 0.27, META 0.09, QQQ 0.07, SPY 0.00.
- Fenêtres testées :
  - W0 (2025-01-01 → 2025-03-01)
  - W1 (2024-09-01 → 2024-11-30)
  - W2 (2025-03-01 → 2025-05-31)

| Window | PF  | MaxDD % | Trades | Sharpe (si dispo) |
|--------|-----|---------|--------|-------------------|
| W0     | 6.8560 | 3.21  | 367    | TODO              |
| W1     | 1.3031 | 2.01  | 356    | TODO              |
| W2     | 3.7319 | 3.34  | 375    | TODO              |

## Lecture des métriques
- PF: somme gains / somme pertes absolues.
- MaxDD: drawdown max sur equity cumulée.
- Sharpe: moyenne des retours / std * sqrt(annualization) (1h → 1638).
- Sortino: optionnel si besoin focalisé downside.

## Robustesse / Sensibilité
- W1 plus faible (PF proche de 1.3) → SPY coupé (poids 0), META/QQQ réduits dans v2.
- TWAP smoke NVDA/AMD (W0) neutre: PF/DD identiques baseline vs TWAP (4 slices/3600s).
- Guards + Kelly: visent à limiter DD et bloquer séquences de pertes prolongées.

## Points d’attention
- Les perfs BT/Replay ne garantissent pas le live; nécessité de track-record réel.
- Fenêtres adverses (range, faible tendance) peuvent réduire PF; d’où monitor et playbook ALERT.
- Commission/slippage en live peuvent dégrader PF; calibrer `commission_pct`/`slippage_pct` selon broker.
