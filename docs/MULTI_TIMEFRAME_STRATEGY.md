# Multi-Timeframe Strategy (MTF)

## Overview

La stratégie MTF combine les signaux des modèles 1h et 15min pour:
- **Meilleur timing d'entrée** - Plus besoin d'attendre l'heure complète
- **Confirmation des signaux** - Deux timeframes qui s'accordent = signal plus fort
- **Sizing dynamique** - Position plus grosse quand les deux modèles sont d'accord

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Modèle 1h      │     │  Modèle 15min   │
│  (direction)    │     │  (timing)       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  Évaluateur │
              │     MTF     │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Signal    │
              │  Combiné    │
              └─────────────┘
```

## Modèles

### Modèle 1h (existant)
- **Source**: Yahoo Finance / Alpaca
- **Features**: RSI, MACD, Bollinger, SMA, Volume, ATR
- **Rôle**: Direction principale du trade
- **Thresholds**: Long > 0.60, Short < 0.40

### Modèle 15min (nouveau)
- **Source**: Alpaca IEX (2 ans de données)
- **Features**: Même base + time_of_day, intraday patterns
- **Rôle**: Confirmation et timing d'entrée
- **Thresholds**: Long > 0.55, Short < 0.45

### Accuracy des modèles 15min

| Symbol | Train Acc | Val Acc | Status |
|--------|-----------|---------|--------|
| NVDA   | 80.6%     | 75.1%   | ✓ Production |
| MSFT   | 92.0%     | 89.4%   | ✓ Production |
| QQQ    | 94.1%     | 92.2%   | ✓ Production |

## Logique de Combinaison

### Matrice des Signaux

| 1h Signal | 15m Signal | Résultat | Size Mult | Emoji |
|-----------|------------|----------|-----------|-------|
| LONG      | LONG       | STRONG_LONG | 1.5x | 💪 |
| LONG      | FLAT       | NORMAL_LONG | 1.0x | → |
| LONG      | SHORT      | WEAK_LONG | 0.5x | ⚠️ |
| SHORT     | SHORT      | STRONG_SHORT | 1.5x | 💪 |
| SHORT     | FLAT       | NORMAL_SHORT | 1.0x | → |
| SHORT     | LONG       | WEAK_SHORT | 0.5x | ⚠️ |
| FLAT      | LONG       | OPPORTUNISTIC_LONG | 0.7x | 🎯 |
| FLAT      | SHORT      | OPPORTUNISTIC_SHORT | 0.7x | 🎯 |
| FLAT      | FLAT       | NO_SIGNAL | 0.0x | — |

### Exemples

**Cas 1: Signal Fort**
```
1h prob: 0.72 (LONG)
15m prob: 0.68 (LONG)
→ STRONG_LONG, size x1.5
```

**Cas 2: Signal Opportuniste**
```
1h prob: 0.52 (FLAT)
15m prob: 0.62 (LONG)
→ OPPORTUNISTIC_LONG, size x0.7
```

**Cas 3: Conflit**
```
1h prob: 0.65 (LONG)
15m prob: 0.38 (SHORT)
→ WEAK_LONG, size x0.5 (ou skip si configuré)
```

## Configuration

### MTFConfig

```python
@dataclass
class MTFConfig:
    # Thresholds 1h
    threshold_1h_long: float = 0.60
    threshold_1h_short: float = 0.40

    # Thresholds 15m (plus serrés car meilleure accuracy)
    threshold_15m_long: float = 0.55
    threshold_15m_short: float = 0.45

    # Multiplicateurs de taille
    strong_mult: float = 1.5       # Deux TF d'accord
    normal_mult: float = 1.0       # 1h signal, 15m neutre
    weak_mult: float = 0.5         # 1h signal, 15m désaccord
    opportunistic_mult: float = 0.7  # 15m seul

    # Options
    skip_conflicting: bool = False  # Skip les signaux conflictuels?
    allow_15m_only: bool = True     # Autoriser trades 15m seul?
```

## Services Systemd

### Signal Generator 15min

```bash
# Timer - toutes les 15 minutes pendant les heures de marché
hyprl-mtf-15m.timer

# Service
hyprl-mtf-15m.service
```

### Horaires d'exécution
- **Lundi-Vendredi**: 9:31, 9:46, puis :01, :16, :31, :46 de 10h à 16h ET
- **Weekend**: Pas d'exécution

### Commandes

```bash
# Status du timer
sudo systemctl status hyprl-mtf-15m.timer

# Logs
journalctl -u hyprl-mtf-15m.service -f

# Exécution manuelle
sudo systemctl start hyprl-mtf-15m.service

# Désactiver
sudo systemctl stop hyprl-mtf-15m.timer
sudo systemctl disable hyprl-mtf-15m.timer
```

## Format des Signaux

Les signaux MTF sont écrits dans le même fichier que les signaux 1h:
`live/logs/live_signals.jsonl`

```json
{
  "timestamp": "2026-01-10T14:31:00+00:00",
  "signal_id": "NVDA:2026-01-10T14:31:00+00:00",
  "timeframe": "mtf",
  "data_source": "alpaca:iex:mtf",
  "symbol": "NVDA",
  "decision": "long",
  "probability_1h": 0.68,
  "probability_15m": 0.72,
  "decision_1h": "long",
  "decision_15m": "long",
  "strength": "strong",
  "size_multiplier": 1.5,
  "size": 450.5,
  "entry_price": 185.50,
  "stop_price": 183.20,
  "take_profit_price": 190.40,
  "atr": 1.15,
  "reason": "1h_long_15m_confirm",
  "equity": 100000.0
}
```

## Early Exit (Sortie Anticipée)

Le modèle 15min peut aussi déclencher des sorties anticipées:

```python
def should_exit_early(current_position, prob_15m):
    """
    Position LONG + 15m SHORT → Exit signal
    Position SHORT + 15m LONG → Exit signal
    """
```

Cette fonctionnalité est intégrée dans Strategy V3 via les position updates.

## Fichiers Clés

| Fichier | Description |
|---------|-------------|
| `src/hyprl/strategy/multi_timeframe.py` | Logique MTF |
| `scripts/run_mtf_signal_generator.py` | Générateur de signaux |
| `scripts/ops/run_mtf_signals.sh` | Wrapper pour tous les symboles |
| `scripts/train_15m_alpaca.py` | Script d'entraînement 15min |
| `models/*_15m_alpaca.joblib` | Modèles 15min entraînés |
| `deploy/systemd/hyprl-mtf-15m.*` | Services systemd |

## Entraînement des Modèles

### Réentraîner un modèle 15min

```bash
# Sur le VPS
source ~/.env.broker.alpaca
export APCA_API_KEY_ID APCA_API_SECRET_KEY

cd ~/HyprL
python scripts/train_15m_alpaca.py \
    --symbol NVDA \
    --days 700 \
    --forward-bars 4 \
    --output-dir models
```

### Paramètres

- `--days`: Jours de données historiques (max ~700 pour Alpaca free)
- `--forward-bars`: Barres à prédire (4 bars = 1h pour 15min data)
- `--output-dir`: Répertoire de sortie

## Avantages vs 1h Seul

| Aspect | 1h Seul | MTF (1h + 15m) |
|--------|---------|----------------|
| Fréquence signaux | 1/heure | 4/heure |
| Timing entrée | Peut être tardif | Plus précis |
| Confirmation | Aucune | Double validation |
| Sizing | Fixe | Dynamique selon force |
| Early exit | Non | Oui via 15m |

## Risques et Limitations

1. **Overtrading**: Plus de signaux = plus de commissions potentielles
2. **Conflits**: Les modèles peuvent se contredire
3. **Latence data**: IEX a ~15min de délai vs SIP
4. **Weekend/Holidays**: Pas de signaux hors marché

## Monitoring

### Métriques à suivre

```bash
# Signaux par jour
grep "mtf" live/logs/live_signals.jsonl | grep "$(date +%Y-%m-%d)" | wc -l

# Distribution des forces
grep "mtf" live/logs/live_signals.jsonl | jq -r '.strength' | sort | uniq -c

# Win rate par force
# (à implémenter dans le dashboard)
```

### Alertes Discord

Les signaux MTF sont envoyés via le même webhook Discord que les signaux 1h.
Le champ `strength` indique la force: strong/normal/weak/opportunistic.
