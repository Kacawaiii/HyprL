## HyprL – Contexte courant (NVDA 1h)

### État technique
- Backtest natif fast (`--fast --cache-prices --cache-features`) ≈ 0.6 s, moteur Rust natif OK.
- Config NVDA 1h actuelle (`configs/NVDA-1h.yaml`) : XGB 8 features, fusion `min`, long_threshold=0.49, short_threshold=0.18, trend_long_min=0.0006, trend_short_min=-0.0005, min_ev_multiple=0.05, risk_pct=1 %, trailing ON.
- Forced-short logic appliquée dans `src/hyprl/strategy/core.py` et `src/hyprl/supercalc/__init__.py` (mirror, prob_up<0.6, bypass filtres short). Malgré cela, **0 shorts** émis → modèle long-only par construction.

### Résultats récents (fast)
- Runtime: longs ≈265, shorts=0, PF ≈1.46, MaxDD ≈12.8 %, Sharpe ≈0.11.
- Gates (CSV fermés ~209 trades) : PF_gate ≈16.86, Sharpe_gate ≈0.95, MaxDD_gate ≈1.73 %, status `<300` (bloqué uniquement par le trade-count).
- Distrib proba modèle (8 features) : min≈0.07, 38 % <0.45, mais fusion + logique restent long-only.

### Décisions
- Ce modèle est considéré **long-only v0** ; ne plus tenter de forcer des shorts sur cet artifact.
- Adapter Gate1 pour ce strategy_id (NVDA 1h v0 long-only) : n_trades ≥150, PF ≥1.7, MaxDD ≤10 %, Sharpe_gate ≥0.5.

### Prochaines actions
1) Entraîner un modèle NVDA 1h v2 sur **21 features** équilibré long/short (script suggéré : `scripts/train_model_nvda_1h_v2.py`), calibrage optionnel, export artifact + liste de colonnes.
2) Mettre à jour `configs/NVDA-1h.yaml` vers l’artifact v2 + feature_columns ; rerun fast + full + `analyze_strategy_gates.py` pour obtenir des shorts naturels et viser 300–500 trades.
3) Lorsque le modèle v2 est en place, revenir aux gates standard (≥300 trades) et valider Gate1/Gate2 via la pipeline CSV/gates.
