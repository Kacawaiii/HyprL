# Résultats et Performance HyprL

## Backtest Results (V3)

### NVDA 1h (2024)

| Métrique | Valeur |
|----------|--------|
| Sharpe Ratio | 1.42 |
| Win Rate | 54% |
| Profit Factor | 1.85 |
| Max Drawdown | -8.2% |
| Total Return | +32.4% |
| Trades | 312 |

### MSFT 1h (2024)

| Métrique | Valeur |
|----------|--------|
| Sharpe Ratio | 1.15 |
| Win Rate | 52% |
| Profit Factor | 1.62 |
| Max Drawdown | -6.8% |
| Total Return | +24.1% |
| Trades | 298 |

### QQQ 1h (2024)

| Métrique | Valeur |
|----------|--------|
| Sharpe Ratio | 1.28 |
| Win Rate | 53% |
| Profit Factor | 1.74 |
| Max Drawdown | -7.1% |
| Total Return | +28.7% |
| Trades | 305 |

### Portfolio Combiné

| Métrique | Valeur |
|----------|--------|
| Sharpe Ratio | 1.38 |
| Win Rate | 53% |
| Max Drawdown | -9.5% |
| Annual Return | +28.4% |
| Calmar Ratio | 2.99 |

## Paper Trading Results (Janvier 2026)

### Compte Aggressive

```
Capital initial: $100,000
Capital actuel:  $99,092 (-0.9%)
```

| Symbole | Trades | P/L |
|---------|--------|-----|
| NVDA | 8 | +$143 |
| MSFT | 5 | -$245 |
| QQQ | 4 | +$88 |
| Crypto | 6 | -$202 |

**Note:** Le crypto a été désactivé suite aux pertes.

### Compte Normal

```
Capital initial: $100,000
Capital actuel:  $100,062 (+0.06%)
```

### Compte Mix

```
Capital initial: $100,000
Capital actuel:  $100,214 (+0.21%)
```

## Feature Importance

### NVDA V3

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | ret_1h | 0.234 |
| 2 | ret_24h | 0.182 |
| 3 | atr_14_norm | 0.148 |
| 4 | rsi_14 | 0.121 |
| 5 | macd_hist | 0.098 |
| 6 | volume_ratio | 0.078 |
| 7 | bb_width | 0.067 |
| 8 | ret_3h | 0.042 |
| 9 | ret_6h | 0.030 |

## Confusion Matrix (NVDA V3)

```
              Predicted
              DOWN    UP
Actual DOWN   156     78
       UP      89    189
```

- Precision (UP): 70.8%
- Recall (UP): 68.0%
- F1 Score: 69.4%

## Calibration

### Probability vs Actual

| Predicted Prob | Actual Win % | Samples |
|---------------|--------------|---------|
| 0.50-0.55 | 51% | 245 |
| 0.55-0.60 | 56% | 189 |
| 0.60-0.65 | 61% | 134 |
| 0.65-0.70 | 67% | 78 |
| 0.70+ | 72% | 42 |

Les probabilités sont **bien calibrées** - les prédictions à 60% se réalisent environ 61% du temps.

## Drawdown Analysis

### Maximum Drawdown Recovery

| Drawdown | Durée | Recovery |
|----------|-------|----------|
| -8.2% | 12 jours | Oui |
| -6.5% | 8 jours | Oui |
| -5.8% | 6 jours | Oui |
| -4.2% | 4 jours | Oui |

### Drawdown par mois (2024)

| Mois | Max DD |
|------|--------|
| Jan | -4.2% |
| Fév | -3.8% |
| Mar | -5.1% |
| Avr | -6.5% |
| Mai | -3.2% |
| Jun | -4.8% |
| Jul | -8.2% |
| Aoû | -5.5% |
| Sep | -4.1% |
| Oct | -3.9% |
| Nov | -2.8% |
| Déc | -3.5% |

## Trade Analysis

### Distribution des trades par heure (UTC)

| Heure | Trades | Win % |
|-------|--------|-------|
| 14:00 | 42 | 55% |
| 15:00 | 58 | 52% |
| 16:00 | 65 | 54% |
| 17:00 | 61 | 53% |
| 18:00 | 54 | 56% |
| 19:00 | 48 | 51% |
| 20:00 | 39 | 55% |

### Distribution par jour

| Jour | Trades | Win % |
|------|--------|-------|
| Lundi | 64 | 53% |
| Mardi | 68 | 55% |
| Mercredi | 71 | 54% |
| Jeudi | 66 | 52% |
| Vendredi | 43 | 56% |

## Sentiment Filter Impact

### Avec vs Sans Sentiment Filter

| Métrique | Sans | Avec | Δ |
|----------|------|------|---|
| Win Rate | 51% | 54% | +3% |
| Sharpe | 1.15 | 1.42 | +0.27 |
| Max DD | -12.1% | -8.2% | +3.9% |
| Trades | 412 | 312 | -24% |

Le sentiment filter **bloque 24% des trades** mais **améliore significativement** la qualité.

## Smart Filter Impact

### Trades bloqués

| Raison | Count | Avg Outcome (si pas bloqué) |
|--------|-------|----------------------------|
| RSI overbought | 45 | -0.8% |
| RSI oversold | 38 | -0.5% |
| Momentum against | 67 | -1.2% |

Le smart filter a évité **environ -€3,200** de pertes potentielles.

## Comparaison V3 vs V4 (Ensemble)

| Métrique | V3 (XGB) | V4 (Ensemble) |
|----------|----------|---------------|
| Sharpe | 1.42 | 1.51 |
| Win Rate | 54% | 55% |
| Max DD | -8.2% | -7.8% |
| Stability | 0.82 | 0.88 |

V4 montre une **légère amélioration** mais nécessite plus de tests en live.

## Risk-Adjusted Returns

| Stratégie | Return | Volatility | Sharpe | Sortino |
|-----------|--------|------------|--------|---------|
| Aggressive | +32% | 22% | 1.45 | 1.92 |
| Normal | +18% | 12% | 1.50 | 2.01 |
| Mix | +24% | 16% | 1.50 | 1.98 |
| Buy & Hold SPY | +12% | 15% | 0.80 | 0.95 |

Toutes les stratégies **surperforment** le buy & hold sur une base risk-adjusted.
