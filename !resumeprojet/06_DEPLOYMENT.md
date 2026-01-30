# Déploiement HyprL

## Infrastructure

### VPS Oracle Cloud

| Paramètre | Valeur |
|-----------|--------|
| IP | 89.168.48.147 |
| OS | Ubuntu 22.04 LTS |
| CPU | 4 vCPU |
| RAM | 24 GB |
| Disk | 200 GB |

### Accès SSH

```bash
ssh -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147
```

### Structure sur le VPS

```
/opt/hyprl/
├── venv/                          # Environnement Python
├── src/hyprl/                     # Code source
├── scripts/                       # Scripts d'exécution
├── models/                        # Modèles ML
├── configs/
│   └── runtime/
│       ├── .env.aggressive        # Clés API Alpaca Aggressive
│       ├── .env.normal            # Clés API Alpaca Normal
│       ├── .env.mix               # Clés API Alpaca Mix
│       ├── strategy_aggressive.yaml
│       ├── strategy_normal.yaml
│       └── strategy_mix.yaml
├── live/logs/
│   ├── live_signals.jsonl         # Signaux générés
│   └── exit_signals.jsonl         # Signaux de sortie
└── data/
    └── position_state.json        # État des positions
```

## Services systemd

### Liste des services

```bash
systemctl list-units --type=service | grep hyprl
```

| Service | Description | Port |
|---------|-------------|------|
| hyprl-aggressive | Bridge Aggressive | - |
| hyprl-normal | Bridge Normal | - |
| hyprl-mix | Bridge Mix | - |
| hyprl-mt5-api | API pour MT5 | 5050 |

### Fichiers de service

#### hyprl-aggressive.service

```ini
[Unit]
Description=HyprL Trading Bridge - Aggressive Strategy
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/hyprl
EnvironmentFile=/opt/hyprl/configs/runtime/.env.aggressive
ExecStart=/opt/hyprl/venv/bin/python scripts/execution/run_strategy_bridge.py --strategy aggressive --paper
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

#### hyprl-mt5-api.service

```ini
[Unit]
Description=HyprL MT5 Signal API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/hyprl
Environment="MT5_API_KEY=hyprl_mt5_ftmo_2026"
ExecStart=/opt/hyprl/venv/bin/python scripts/execution/mt5_signal_api.py --port 5050 --signals /opt/hyprl/live/logs/live_signals.jsonl
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Commandes utiles

```bash
# Voir le status
sudo systemctl status hyprl-aggressive

# Voir les logs
sudo journalctl -u hyprl-aggressive -f

# Redémarrer
sudo systemctl restart hyprl-aggressive

# Activer au boot
sudo systemctl enable hyprl-aggressive
```

## Web Server (Nginx)

### Configuration

```nginx
# /etc/nginx/sites-available/hyprlcore.com

server {
    server_name hyprlcore.com www.hyprlcore.com;
    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # MT5 Signal API
    location /mt5-api/ {
        proxy_pass http://127.0.0.1:5050/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/hyprlcore.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hyprlcore.com/privkey.pem;
}
```

### Fichiers web

```
/var/www/html/
├── index.html              # Landing page
├── portfolio.html          # Page portfolio détaillée
├── checkout.html           # Page de paiement Stripe
├── css/
├── js/
└── assets/
```

## API Endpoints

### MT5 Signal API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Status de l'API |
| `/signals?key=<KEY>` | GET | Tous les signaux actifs |
| `/signals/<SYMBOL>?key=<KEY>` | GET | Signal pour un symbole |
| `/mt5/format?key=<KEY>` | GET | Format MT5 compatible |

### Exemple

```bash
# Health check
curl https://hyprlcore.com/mt5-api/health

# Get signals
curl "https://hyprlcore.com/mt5-api/signals?key=hyprl_mt5_ftmo_2026"
```

## Déploiement du code

### Script de déploiement

```bash
#!/bin/bash
# deploy.sh

VPS="ubuntu@89.168.48.147"
KEY="~/.ssh/oracle_hyprl"

# Copier les fichiers
scp -i $KEY src/hyprl/**/*.py $VPS:/tmp/

# Installer
ssh -i $KEY $VPS "sudo cp /tmp/*.py /opt/hyprl/src/hyprl/"

# Redémarrer les services
ssh -i $KEY $VPS "sudo systemctl restart hyprl-aggressive hyprl-normal hyprl-mix"
```

### Mise à jour des modèles

```bash
# Copier un nouveau modèle
scp -i ~/.ssh/oracle_hyprl models/nvda_1h_xgb_v4.joblib ubuntu@89.168.48.147:/tmp/
ssh -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147 "sudo cp /tmp/*.joblib /opt/hyprl/models/"
```

## Monitoring

### Logs en temps réel

```bash
# Signaux générés
ssh -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147 "tail -f /opt/hyprl/live/logs/live_signals.jsonl"

# Logs d'un service
ssh -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147 "sudo journalctl -u hyprl-aggressive -f"
```

### Vérifier les positions

```bash
ssh -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147 "cat /opt/hyprl/data/position_state.json | jq ."
```

## Sécurité

### Firewall (iptables)

```bash
# Ports ouverts
22    # SSH
80    # HTTP
443   # HTTPS
5050  # MT5 API (via nginx proxy)
```

### Clés API

Les clés API Alpaca sont stockées dans les fichiers `.env.*`:

```bash
# .env.aggressive
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
```

**Ne jamais commiter ces fichiers!**

## Backup

### Sauvegarde des données

```bash
# Backup des logs
ssh -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147 "tar -czf /tmp/hyprl_backup.tar.gz /opt/hyprl/live/logs /opt/hyprl/data"
scp -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147:/tmp/hyprl_backup.tar.gz ./backups/
```

### Sauvegarde des configs

```bash
scp -i ~/.ssh/oracle_hyprl ubuntu@89.168.48.147:/opt/hyprl/configs/runtime/*.yaml ./backups/configs/
```
