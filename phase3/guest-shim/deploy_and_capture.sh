#!/bin/bash
#
# Deploy changes and trigger error capture on VM
#
# This script:
# 1. Rebuilds shim libraries with enhanced error capture
# 2. Deploys to VM
# 3. Triggers discovery and captures errors
#
# Usage: ./deploy_and_capture.sh [vm_user@vm_host]
#

set -e

VM=${1:-"test-10@10.25.33.110"}
PASSWORD="Calvin@123"

echo "=========================================="
echo "Deploy and Capture Script"
echo "=========================================="
echo "VM: $VM"
echo ""

# Step 1: Rebuild libraries
echo "[1/4] Rebuilding shim libraries..."
cd "$(dirname "$0")"
if [ -f install.sh ]; then
    ./install.sh
    echo "  ✓ Libraries rebuilt"
else
    echo "  ✗ install.sh not found"
    exit 1
fi

# Step 2: Deploy to VM
echo "[2/4] Deploying to VM..."
echo "  Copying files..."

# Copy libraries
scp -o StrictHostKeyChecking=no /usr/lib64/libvgpu-*.so "$VM:/tmp/" 2>/dev/null || {
    echo "  Note: Libraries may need to be copied manually or already exist on VM"
}

# Copy scripts
scp -o StrictHostKeyChecking=no capture_errors.sh analyze_errors.sh verify_symbols.sh "$VM:~/phase3/guest-shim/" 2>/dev/null || {
    echo "  Note: Scripts may need to be copied manually"
}

# Step 3: Install on VM
echo "[3/4] Installing on VM..."
ssh -o StrictHostKeyChecking=no "$VM" <<'ENDSSH'
    cd ~/phase3/guest-shim
    if [ -f install.sh ]; then
        sudo ./install.sh
        echo "  ✓ Installation complete"
    else
        echo "  ✗ install.sh not found on VM"
        exit 1
    fi
    
    # Make scripts executable
    chmod +x capture_errors.sh analyze_errors.sh verify_symbols.sh 2>/dev/null || true
ENDSSH

# Step 4: Trigger error capture
echo "[4/4] Triggering error capture on VM..."
echo "  This will take 60 seconds..."
ssh -o StrictHostKeyChecking=no "$VM" <<'ENDSSH'
    cd ~/phase3/guest-shim
    if [ -f capture_errors.sh ]; then
        ./capture_errors.sh 60
    else
        echo "  ✗ capture_errors.sh not found"
        exit 1
    fi
ENDSSH

echo ""
echo "=========================================="
echo "Deploy and capture complete!"
echo "=========================================="
echo ""
echo "To retrieve captured errors, run:"
echo "  scp $VM:/tmp/ollama_error_capture_* ./"
echo ""
echo "To analyze errors on VM, run:"
echo "  ssh $VM 'cd ~/phase3/guest-shim && ./analyze_errors.sh'"
