# HyprL Ascendant v2 – Product Overview

## Introduction
- Problème: la plupart des bots retail sont soit naïfs (aucun contrôle DD), soit opaques (impossible à auditer / reproduire).
- Besoin: un moteur quant reproductible (BT/Replay/Live) avec un layer risque (Kelly, caps, guards) et des outils d’ops/monitoring.
- Réponse: HyprL Ascendant v2 = pipeline multi-ticker 1h avec parité BT↔Replay, risk intégré, monitor PF/DD/Sharpe, runbook live minimal déjà validé (NVDA Palier 1).

## HyprL Ascendant v2 en 1 page
- Signal engine: features v2 (equity presets), modèles XGBoost calibrés, seuils longs/shorts par ticker.
- Risk layer: Kelly dynamique borné, caps notional/positions, guards (PF min, DD max, consec losses).
- Portfolio: Ascendant v2 weights (NVDA 0.30, MSFT 0.27, AMD 0.27, META 0.09, QQQ 0.07, SPY 0.00).
- Execution: market orders par défaut; TWAP optionnel (4 slices/3600s) validé en smoke NVDA/AMD.
- Monitoring: health checker PF/MaxDD/Sharpe; aggregation replay-orch; live-lite ops scripts.

## Avantages / différenciateurs
- Parité BT↔Replay validée par ticker v2.
- Portfolio backtesté multi-fenêtres (W0/W1/W2) avec PF>1.3 et MaxDD<5% (replay-orch).
- Risk intégré (Kelly + guards + caps) sans recoder la stack.
- Runbook live minimal NVDA 1h v2 opérationnel.
- Outils CLI: runners horaires, concat, monitor PF/DD/Sharpe, aggregation portfolio.

## Scope actuel
- Tickers v2: NVDA, MSFT, AMD, META, QQQ (SPY présent mais poids 0 en v2).
- Horizon/TF: 1h US.
- Modes: backtest, replay-orch, live-like (single ticker), live-lite multi-ticker (ops prêts).
- Risk: Kelly dynamique, guards (DD/PF/consec losses), caps (total/ticker/groupe), monitor PF/DD/Sharpe.

## Limitations / disclaimers
- Pas une promesse de rendement; track-record live encore court (NVDA ~120 trades Palier1).
- Multi-ticker live runner historique expérimental; live-lite = runners single-ticker + aggregation offline.
- Données marché non incluses; dépendances Python/Rust à installer.
- TWAP validé en smoke NVDA/AMD uniquement sur W0; optionnel/off par défaut en live minimal.
