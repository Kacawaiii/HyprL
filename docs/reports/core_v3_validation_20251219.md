# core_v3 validation — 2025-12-19

- Source replay: live/exp/portfolio_repro_run (OOS 2024-03-01 → 2024-12-01)
- Golden hashes: docs/reports/core_v3_repro_golden.sha256
- Models: models/*_1h_xgb_v3.joblib (19-feature contract)

## Per-ticker performance
| Ticker | Trades | PF | Win% | PnL | Final Balance | Trailing% |
|--------|--------|-----|------|-----|---------------|-----------|
| NVDA | 118 | 5.93 | 79.7% | +22,179.57 | 32,179.57 | 79.7% |
| MSFT | 182 | 2.95 | 74.7% | +20,532.88 | 30,532.88 | 74.2% |
| QQQ | 168 | 2.77 | 74.4% | +16,612.67 | 26,612.67 | 74.4% |

## Portfolio aggregated (weights 0.4/0.3/0.3)
```json
{
  "pf": 3.5733,
  "maxdd": 2.6209,
  "trades": 468,
  "equity_end": 30015.49
}
```

## Quality gates
- PF > 1.5: 3.57 ✅
- MaxDD < 5%: 2.62% ✅
- Win% > 60%: 76.3% ✅
- Trailing dominant: ~75% ✅
- All tickers PF > 1.0: min 2.77 ✅

## Notes
- Costs: Paper broker defaults (commission_pct=0.0005, slippage_pct=0.0005 per side).
- Engine: replay aggregator (no native backtest runner present).
- Factsheet v1r2 remains reference; PF delta likely due to engine/window/cost differences vs. original factsheet.
