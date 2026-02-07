#!/bin/bash
# Deploy Aggressive V2 to VPS
# Run this locally: ./deploy_aggressive_v2.sh

VPS="ubuntu@89.168.48.147"
REMOTE_DIR="/opt/hyprl"

echo "=== DEPLOYING AGGRESSIVE V2 TO VPS ==="

# 1. Copy new config
echo "1. Uploading new config..."
scp configs/runtime/strategy_aggressive.yaml $VPS:$REMOTE_DIR/configs/runtime/

# 2. Copy new safety modules
echo "2. Uploading safety modules..."
scp src/hyprl/risk/circuit_breakers.py $VPS:$REMOTE_DIR/src/hyprl/risk/
scp src/hyprl/risk/market_hours.py $VPS:$REMOTE_DIR/src/hyprl/risk/
scp src/hyprl/risk/safety_wrapper.py $VPS:$REMOTE_DIR/src/hyprl/risk/

# 3. Copy monitoring modules
echo "3. Uploading monitoring modules..."
ssh $VPS "mkdir -p $REMOTE_DIR/src/hyprl/monitoring"
scp src/hyprl/monitoring/*.py $VPS:$REMOTE_DIR/src/hyprl/monitoring/

# 4. Copy emergency scripts
echo "4. Uploading emergency scripts..."
scp scripts/ops/stop_all_trading.py $VPS:$REMOTE_DIR/scripts/ops/
scp scripts/ops/emergency_recovery.sh $VPS:$REMOTE_DIR/scripts/ops/

# 5. Activate kill switch on VPS
echo "5. Activating kill switch on VPS..."
ssh $VPS "echo '1' > $REMOTE_DIR/.kill_switch && echo 'Deploy in progress' >> $REMOTE_DIR/.kill_switch"

# 6. Restart services
echo "6. Restarting aggressive service..."
ssh $VPS "sudo systemctl restart hyprl-bridge-aggressive || sudo systemctl restart hyprl-aggressive || echo 'Service restart skipped'"

# 7. Deactivate kill switch
echo "7. Deactivating kill switch..."
ssh $VPS "echo '0' > $REMOTE_DIR/.kill_switch"

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "Check status: ssh $VPS 'sudo systemctl status hyprl-bridge-aggressive'"
