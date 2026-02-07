# Strategy 2.0 - Guide de Deploiement

> Date: 2026-01-09 | Status: Implemente localement, en attente de deploiement serveur

## Modules Crees

### 1. Calendar Events (`src/hyprl/calendar/events.py`)
- Bloque trades 3 jours avant earnings
- Bloque trades jour FOMC
- Bloque trades jour OpEx (3eme vendredi)

### 2. Correlation Tracker (`src/hyprl/risk/correlation.py`)
- Limite exposition correlee a $15k
- Matrice de correlation NVDA/MSFT/QQQ
- Ajuste taille selon correlation portfolio

### 3. Macro Regime (`src/hyprl/regime/macro.py`)
- Detecte regime via VIX (risk_on, neutral, cautious, risk_off, crisis)
- Ajuste thresholds selon regime
- Bloque shorts en risk_off/crisis
- Reduit taille en volatilite elevee

### 4. Liquidity Manager (`src/hyprl/risk/liquidity.py`)
- Reduit taille first/last 30 min
- Limite a 1% du volume quotidien
- Ajuste selon spread actuel vs moyen

### 5. Options Flow (`src/hyprl/options/flow.py`)
- Analyse put/call ratio
- Detecte unusual activity
- Ajuste probabilite signal

### 6. Guards V2 Integration (`src/hyprl/strategy/guards_v2.py`)
- Combine tous les modules
- Interface unifiee pour le bridge

## Deploiement Serveur

### Etape 1: Copier les nouveaux fichiers
```bash
# Depuis la machine locale
scp -r /home/kyo/HyprL/src/hyprl/calendar ubuntu@51.81.167.177:~/HyprL/src/hyprl/
scp -r /home/kyo/HyprL/src/hyprl/options ubuntu@51.81.167.177:~/HyprL/src/hyprl/
scp /home/kyo/HyprL/src/hyprl/risk/correlation.py ubuntu@51.81.167.177:~/HyprL/src/hyprl/risk/
scp /home/kyo/HyprL/src/hyprl/risk/liquidity.py ubuntu@51.81.167.177:~/HyprL/src/hyprl/risk/
scp /home/kyo/HyprL/src/hyprl/regime/macro.py ubuntu@51.81.167.177:~/HyprL/src/hyprl/regime/
scp /home/kyo/HyprL/src/hyprl/strategy/guards_v2.py ubuntu@51.81.167.177:~/HyprL/src/hyprl/strategy/
scp /home/kyo/HyprL/scripts/execution/run_alpaca_bridge.py ubuntu@51.81.167.177:~/HyprL/scripts/execution/
```

### Etape 2: Creer les __init__.py
```bash
ssh ubuntu@51.81.167.177 "
touch ~/HyprL/src/hyprl/calendar/__init__.py
touch ~/HyprL/src/hyprl/options/__init__.py
"
```

### Etape 3: Mettre a jour le service systemd
Ajouter au fichier override.conf:
```
--enable-guards-v2 \
--earnings-blackout-days 3 \
--max-correlated-notional 15000 \
```

### Etape 4: Redemarrer le service
```bash
sudo systemctl daemon-reload
sudo systemctl restart hyprl_alpaca_bridge
sudo journalctl -u hyprl_alpaca_bridge -f
```

## Verification

### Events attendus
| Event | Description |
|-------|-------------|
| guards_v2_init | Initialisation reussie |
| guards_v2_check | Check execute (allowed=true/false) |
| guards_v2_size_adj | Taille ajustee par regime/liquidity |

## Desactivation Selective

```bash
--disable-calendar-guard     # Desactive earnings/FOMC/OpEx
--disable-correlation-guard  # Desactive limites correlation
--disable-regime-guard       # Desactive VIX/macro
--disable-liquidity-guard    # Desactive sizing liquidite
--disable-options-guard      # Desactive options flow
```

---

*Guide cree le 2026-01-09*
