#!/bin/bash
# ==============================================================
# HyprL MVP - Setup Live Trading
# ==============================================================
# Usage: ./scripts/setup_live.sh
# ==============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "HyprL MVP - SETUP LIVE TRADING"
echo "============================================================"

# 1. Check Python environment
echo -e "\n[1/5] Checking Python environment..."
if [ -f ".venv/bin/python" ]; then
    echo "  ✓ Virtual environment found"
else
    echo "  ✗ Virtual environment not found. Run: python -m venv .venv"
    exit 1
fi

# 2. Check dependencies
echo -e "\n[2/5] Checking dependencies..."
.venv/bin/pip install -q alpaca-trade-api websockets yfinance pandas numpy pyyaml
echo "  ✓ Dependencies installed"

# 3. Check Alpaca credentials
echo -e "\n[3/5] Checking Alpaca credentials..."
if [ -f ".env.broker.alpaca" ]; then
    source .env.broker.alpaca
    if [ -n "$ALPACA_API_KEY" ] && [ -n "$ALPACA_SECRET_KEY" ]; then
        echo "  ✓ API keys found"
        echo "  Testing connection..."

        .venv/bin/python -c "
import os
os.environ['ALPACA_API_KEY'] = '$ALPACA_API_KEY'
os.environ['ALPACA_SECRET_KEY'] = '$ALPACA_SECRET_KEY'
import alpaca_trade_api as tradeapi
api = tradeapi.REST('$ALPACA_API_KEY', '$ALPACA_SECRET_KEY', 'https://paper-api.alpaca.markets', api_version='v2')
try:
    account = api.get_account()
    print(f'  ✓ Connected! Equity: \${float(account.equity):,.2f}')
except Exception as e:
    print(f'  ✗ Connection failed: {e}')
    print('    → Regenerate keys at: https://app.alpaca.markets/paper/dashboard/overview')
    exit(1)
" || {
            echo "  ✗ Alpaca connection failed"
            echo "    → Regenerate keys at: https://app.alpaca.markets/paper/dashboard/overview"
            echo "    → Update .env.broker.alpaca with new keys"
            exit 1
        }
    else
        echo "  ✗ API keys missing in .env.broker.alpaca"
        exit 1
    fi
else
    echo "  ✗ .env.broker.alpaca not found"
    echo "    → Copy example: cp .env.broker.alpaca.example .env.broker.alpaca"
    echo "    → Add your API keys"
    exit 1
fi

# 4. Remove kill switch if present
echo -e "\n[4/5] Checking kill switch..."
if [ -f ".kill_switch" ]; then
    rm .kill_switch
    echo "  ✓ Kill switch removed"
else
    echo "  ✓ No kill switch active"
fi

# 5. Run validation check
echo -e "\n[5/5] Running pre-launch checks..."
source .env.broker.alpaca
.venv/bin/python scripts/go_live.py --check

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "To start trading:"
echo ""
echo "  # Dry run (recommended first):"
echo "  source .env.broker.alpaca && python scripts/go_live.py --start --capital 2000 --dry-run"
echo ""
echo "  # Live trading:"
echo "  source .env.broker.alpaca && python scripts/go_live.py --start --capital 2000"
echo ""
echo "  # Or install systemd service:"
echo "  sudo cp deploy/systemd/hyprl-mvp-live.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable hyprl-mvp-live"
echo "  sudo systemctl start hyprl-mvp-live"
echo ""
echo "============================================================"
