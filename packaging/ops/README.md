# HyprL Ops Toolkit

This is an operations bundle: systemd templates + crypto bridge monitoring.
It assumes the HyprL core repo is already installed on the target host.

## Contents
- systemd templates for crypto signals + bridge + monitor
- `scripts/ops/run_crypto_signals.sh`
- `scripts/ops/monitor_crypto_bridge.py`

## Requirements
- HyprL repo layout (same as this package)
- Python venv at `.venv/`
- Alpaca paper keys

## Setup
1) Copy templates
```
sudo cp -r systemd/* /etc/systemd/system/
```

2) Edit placeholders in the unit files
- Replace `{{HYPRL_ROOT}}` with your repo path
- Replace `{{HYPRL_USER}}` with your Linux user

3) Create env files
```
cp .env.bridge.example /path/to/HyprL/.env.bridge
# Optional
# /path/to/HyprL/.env.discord
```

4) Enable services
```
sudo systemctl daemon-reload
sudo systemctl enable --now hyprl-crypto.timer
sudo systemctl enable --now hyprl-crypto-bridge.service
sudo systemctl enable --now hyprl-crypto-monitor.service
```

## Validation
- `systemctl status hyprl-crypto-bridge`
- Check `live/execution/crypto/orders.jsonl`

