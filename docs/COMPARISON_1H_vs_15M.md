# Comparaison 1H vs 15min - Analyse

> Date: 2026-01-09

## Resultats d'entrainement

### Modeles 1H (existants, 730 jours de donnees)
| Ticker | Val Accuracy | Val Brier | Note |
|--------|--------------|-----------|------|
| NVDA | ~65% | ~0.22 | Bon modele |
| MSFT | ~62% | ~0.24 | Bon modele |
| QQQ | ~60% | ~0.25 | OK |

### Modeles 15min (nouveaux, 60 jours de donnees)
| Ticker | Val Accuracy | Val Brier | Note |
|--------|--------------|-----------|------|
| NVDA | 54% | 0.27 | Proche aleatoire |
| MSFT | 47% | 0.30 | Aleatoire |
| QQQ | 46% | 0.29 | Aleatoire |

## Probleme: Limitation des donnees

Yahoo Finance limite les donnees 15min a **60 jours** (~1500 barres).

- 1H sur 730 jours = ~4500 barres d'entrainement
- 15min sur 60 jours = ~1000 barres d'entrainement

Le modele 15min a **4x moins de donnees** pour apprendre.

## Conclusion

| Critere | 1H | 15min |
|---------|----|----|
| Qualite signal | ✅ Bon (65%) | ❌ Aleatoire (50%) |
| Donnees dispo | ✅ 730+ jours | ❌ 60 jours max |
| Overfitting | ✅ Controlle | ❌ Severe |
| Production ready | ✅ Oui | ❌ Non |

## Recommandation

**Garder 1H pour le moment.**

Pour passer a 15min il faudrait:
1. **Source de donnees alternative** (Alpaca Pro, Polygon, etc.) avec 1+ an d'historique
2. **Re-entrainer** avec 5000+ barres minimum
3. **Recalibrer** les thresholds sur backtest

## Alternative: Execution 15min, Signal 1H

On peut garder le signal horaire mais **executer plus frequemment**:
- Signal genere chaque heure
- Mais monitoring des positions toutes les 15 min
- Ajustement des stops/TP en temps reel

C'est ce que fait deja le systeme actuel avec le bridge qui poll toutes les 10 secondes en RTH.

---
*Document genere automatiquement*
