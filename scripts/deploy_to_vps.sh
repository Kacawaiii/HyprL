#!/bin/bash
# =============================================================================
# HyprL VPS Deployment Script
# =============================================================================
# Déploie HyprL sur le VPS avec systemd timer (toutes les 15 min)
#
# Usage:
#   ./scripts/deploy_to_vps.sh [VPS_HOST]
#
# Example:
#   ./scripts/deploy_to_vps.sh root@vps.example.com
#   ./scripts/deploy_to_vps.sh kyo@192.168.1.100
# =============================================================================

set -e

# Configuration
VPS_HOST="${1:-root@vps}"
REMOTE_DIR="/opt/hyprl"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=============================================="
echo "HYPRL VPS DEPLOYMENT"
echo "=============================================="
echo "Local:  $LOCAL_DIR"
echo "Remote: $VPS_HOST:$REMOTE_DIR"
echo ""

# Check SSH connection
echo "[1/6] Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 "$VPS_HOST" "echo 'SSH OK'" 2>/dev/null; then
    echo "❌ Cannot connect to $VPS_HOST"
    echo "   Usage: $0 user@host"
    exit 1
fi
echo "  ✓ SSH connection OK"

# Create remote directory
echo ""
echo "[2/6] Creating remote directory..."
ssh "$VPS_HOST" "mkdir -p $REMOTE_DIR/{scripts,configs/runtime,deploy/systemd,src/hyprl/monitoring,logs}"
echo "  ✓ Directory structure created"

# Sync files
echo ""
echo "[3/6] Syncing files..."

# Core scripts
rsync -avz --progress \
    "$LOCAL_DIR/scripts/run_multi_account_v2.py" \
    "$LOCAL_DIR/scripts/install_systemd.sh" \
    "$VPS_HOST:$REMOTE_DIR/scripts/"

# Configs
rsync -avz --progress \
    "$LOCAL_DIR/configs/runtime/.env.normal" \
    "$LOCAL_DIR/configs/runtime/.env.aggressive" \
    "$LOCAL_DIR/configs/runtime/.env.mix" \
    "$LOCAL_DIR/configs/runtime/.env.discord" \
    "$VPS_HOST:$REMOTE_DIR/configs/runtime/" 2>/dev/null || echo "  (some env files may not exist)"

# Systemd
rsync -avz --progress \
    "$LOCAL_DIR/deploy/systemd/hyprl-multi-account.service" \
    "$LOCAL_DIR/deploy/systemd/hyprl-multi-account.timer" \
    "$VPS_HOST:$REMOTE_DIR/deploy/systemd/"

# Source code (monitoring module)
rsync -avz --progress \
    "$LOCAL_DIR/src/hyprl/monitoring/" \
    "$VPS_HOST:$REMOTE_DIR/src/hyprl/monitoring/"

# .env.ops if exists
if [ -f "$LOCAL_DIR/.env.ops" ]; then
    rsync -avz "$LOCAL_DIR/.env.ops" "$VPS_HOST:$REMOTE_DIR/"
fi

echo "  ✓ Files synced"

# Setup Python environment on VPS
echo ""
echo "[4/6] Setting up Python environment on VPS..."
ssh "$VPS_HOST" << 'REMOTE_SCRIPT'
cd /opt/hyprl

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    apt-get update && apt-get install -y python3 python3-pip python3-venv
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Install dependencies
echo "Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet \
    yfinance \
    pandas \
    numpy \
    alpaca-trade-api \
    python-dotenv \
    pytz \
    aiohttp \
    discord.py

echo "  ✓ Python environment ready"
REMOTE_SCRIPT

# Install systemd services
echo ""
echo "[5/6] Installing systemd services..."
ssh "$VPS_HOST" << 'REMOTE_SCRIPT'
cd /opt/hyprl

# Fix paths in service file
sed -i 's|/home/kyo/HyprL|/opt/hyprl|g' deploy/systemd/hyprl-multi-account.service
sed -i 's|User=kyo|User=root|g' deploy/systemd/hyprl-multi-account.service

# Copy to systemd
cp deploy/systemd/hyprl-multi-account.service /etc/systemd/system/
cp deploy/systemd/hyprl-multi-account.timer /etc/systemd/system/

# Reload and enable
systemctl daemon-reload
systemctl enable hyprl-multi-account.timer
systemctl start hyprl-multi-account.timer

echo "  ✓ Systemd services installed"
REMOTE_SCRIPT

# Verify
echo ""
echo "[6/6] Verifying deployment..."
ssh "$VPS_HOST" << 'REMOTE_SCRIPT'
echo ""
echo "=== TIMER STATUS ==="
systemctl status hyprl-multi-account.timer --no-pager | head -10

echo ""
echo "=== NEXT EXECUTIONS ==="
systemctl list-timers hyprl-multi-account.timer --no-pager

echo ""
echo "=== TEST RUN ==="
cd /opt/hyprl
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')
from scripts.run_multi_account_v2 import is_market_open
print(f'Market open: {is_market_open()}')
print('✓ Script loads OK')
"
REMOTE_SCRIPT

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""
echo "📋 Commandes utiles sur le VPS:"
echo ""
echo "  # Voir le status"
echo "  ssh $VPS_HOST 'systemctl status hyprl-multi-account.timer'"
echo ""
echo "  # Voir les logs"
echo "  ssh $VPS_HOST 'journalctl -u hyprl-multi-account -f'"
echo ""
echo "  # Exécuter manuellement"
echo "  ssh $VPS_HOST 'systemctl start hyprl-multi-account.service'"
echo ""
echo "  # Arrêter"
echo "  ssh $VPS_HOST 'systemctl stop hyprl-multi-account.timer'"
echo ""
echo "=============================================="
