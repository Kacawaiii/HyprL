# HyprL Strategy Modes

Ce document décrit les trois modes de trading disponibles et comment les déployer.

## Vue d'ensemble

| Mode | Risque | Retour (backtest) | Drawdown Max | Filtres |
|------|--------|-------------------|--------------|---------|
| **Normal** | Modéré | +225% | 2.2% | Tous activés |
| **Aggressive** | Élevé | +12,764% | 14.9% | Désactivés |
| **Mix** | Hybride | ~400-500% | 5-8% | 70% filtré |

## 1. Strategy Normal (Recommandé)

Configuration équilibrée avec tous les filtres activés.

### Caractéristiques
- Position size: 2% du capital par trade
- Maximum 10% par symbole
- Stop loss: 1.5 ATR
- Take profit: 3.0 ATR (ratio 2:1)
- Smart Filter: activé (bloque falling knives, capitulation)
- Sentiment Filter: activé
- Quality Filter: activé
- Multi-timeframe: activé

### Résultats Backtest (2023-2025)
- **Retour**: +225%
- **Win Rate**: 71.9%
- **Max Drawdown**: 2.2%
- **Sharpe Ratio**: 9.07
- **Profit Factor**: 4.24

### Quand l'utiliser
- Capital que vous ne voulez pas perdre
- Trading à long terme
- Première mise en production

---

## 2. Strategy Aggressive (Haut Risque)

Configuration maximale sans filtres. **ATTENTION: Risque élevé.**

### Caractéristiques
- Position size: 30% du capital par trade
- Maximum 50% par symbole
- Exposition totale: jusqu'à 150% (leverage)
- Stop loss: 3.0 ATR (large)
- Take profit: 6.0 ATR
- **Tous les filtres désactivés**
- Seuil de probabilité: 50% (accepte tout signal > 50%)

### Résultats Backtest (2023-2025)
- **Retour**: +12,764%
- **Win Rate**: 76.8%
- **Max Drawdown**: 14.9%
- **Sharpe Ratio**: 11.47
- **Profit Factor**: 2.21

### Quand l'utiliser
- Capital que vous pouvez vous permettre de perdre
- Compte paper pour tester
- Petits montants pour maximiser les gains

### Avertissements
- Peut subir des pertes importantes rapidement
- Non recommandé pour les débutants
- Volatilité élevée du PnL quotidien

---

## 3. Strategy Mix (70/30)

Approche hybride combinant les deux stratégies.

### Caractéristiques
- **70% du capital**: Strategy Normal (filtré, conservateur)
- **30% du capital**: Strategy Aggressive (non filtré, risqué)
- Limite de perte journalière: 8%
- Drawdown max: 15%
- Rebalancement si drift > 10%

### Résultats Attendus
- **Retour**: ~400-500%
- **Win Rate**: ~74%
- **Max Drawdown**: ~5-8%
- **Sharpe**: Meilleur que aggressive pur

### Quand l'utiliser
- Équilibre entre croissance et protection
- Diversification du risque
- Testing de production

---

## Fichiers de Configuration

Les configs sont dans `configs/runtime/`:

```
configs/runtime/
├── strategy_normal.yaml      # Config Normal
├── strategy_aggressive.yaml  # Config Aggressive
└── strategy_mix.yaml         # Config Mix (70/30)
```

### Exemple: Modifier un seuil

```yaml
# Dans strategy_normal.yaml
symbols:
  NVDA:
    long_threshold: 0.53    # Changer ici
    short_threshold: 0.45
```

---

## Déploiement

### Option 1: Lancement Manuel

```bash
# Normal (recommandé)
python scripts/execution/run_strategy_bridge.py --strategy normal --paper

# Aggressive (haut risque)
python scripts/execution/run_strategy_bridge.py --strategy aggressive --paper

# Mix (hybride)
python scripts/execution/run_strategy_bridge.py --strategy mix --paper

# Mode dry-run (pas d'ordres)
python scripts/execution/run_strategy_bridge.py --strategy normal --dry-run

# Mode live (ATTENTION!)
python scripts/execution/run_strategy_bridge.py --strategy normal --live --i-understand
```

### Option 2: Services Systemd

```bash
# Copier les services
sudo cp deploy/systemd/hyprl-bridge-*.service /etc/systemd/system/

# Recharger systemd
sudo systemctl daemon-reload

# Démarrer un service
sudo systemctl start hyprl-bridge-normal

# Activer au démarrage
sudo systemctl enable hyprl-bridge-normal

# Voir les logs
journalctl -u hyprl-bridge-normal -f
```

### Fichiers de sortie

Chaque stratégie écrit ses propres logs:

```
live/
├── logs/
│   ├── orders_normal.jsonl      # Ordres Normal
│   ├── orders_aggressive.jsonl  # Ordres Aggressive
│   └── orders_mix.jsonl         # Ordres Mix
└── state/
    ├── bridge_normal.json       # État Normal
    ├── bridge_aggressive.json   # État Aggressive
    └── bridge_mix.json          # État Mix
```

---

## Variables d'Environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `HYPRL_BASE_EQUITY` | Capital de base pour sizing | 100000 |
| `HYPRL_STRATEGY` | Stratégie par défaut | normal |
| `APCA_API_KEY_ID` | Clé API Alpaca | (requis) |
| `APCA_API_SECRET_KEY` | Secret API Alpaca | (requis) |
| `APCA_API_BASE_URL` | URL API (paper/live) | paper |

### Fichier .env Exemple

```bash
# Alpaca Paper Trading
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets

# HyprL Config
HYPRL_BASE_EQUITY=100000
HYPRL_STRATEGY=normal
```

---

## Sécurité

### Protection Live Trading

Le mode live nécessite `--i-understand` pour éviter les erreurs:

```bash
# Ne fonctionne PAS
python run_strategy_bridge.py --strategy normal --live

# Fonctionne
python run_strategy_bridge.py --strategy normal --live --i-understand
```

### Kill Switch

Créer un fichier pour arrêter d'urgence:

```bash
# Arrêter tous les trades
touch /opt/hyprl/live/KILL_SWITCH

# Reprendre
rm /opt/hyprl/live/KILL_SWITCH
```

---

## Monitoring

### Dashboard Track Record

```bash
cd apps/track_record
streamlit run streamlit_app.py
```

### Logs en temps réel

```bash
# Voir les signaux
tail -f live/logs/live_signals.jsonl | jq

# Voir les ordres
tail -f live/logs/orders_normal.jsonl | jq
```

### Alertes Discord

Configurer via `--discord-webhook`:

```bash
python run_strategy_bridge.py \
    --strategy normal \
    --paper \
    --discord-webhook "https://discord.com/api/webhooks/..."
```

---

## FAQ

### Puis-je exécuter plusieurs stratégies en parallèle?

Oui, chaque stratégie a ses propres fichiers d'état. Cependant, elles partageront le même compte Alpaca, donc attention aux conflits de position.

### Comment changer la taille des positions?

Modifier `position_size_pct` dans le fichier YAML correspondant:

```yaml
capital:
  position_size_pct: 0.05  # 5% au lieu de 2%
```

### Comment désactiver un filtre?

Dans le fichier YAML:

```yaml
filters:
  enable_smart_filter: false
  enable_sentiment: false
```

### Comment ajouter un nouveau symbole?

Ajouter dans la section `symbols`:

```yaml
symbols:
  NVDA:
    long_threshold: 0.53
    short_threshold: 0.45
  AAPL:  # Nouveau
    long_threshold: 0.55
    short_threshold: 0.48
```

**Note**: Le modèle ML doit avoir été entraîné sur ce symbole.
