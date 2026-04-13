#!/bin/bash
# Deploy dlsym interception to VM
# Run this script on the VM: bash deploy_dlsym_interception.sh

set -e

echo "======================================================================"
echo "DEPLOYING dlsym INTERCEPTION"
echo "======================================================================"

cd ~/phase3/guest-shim || exit 1

echo ""
echo "[1/5] Building shims with dlsym interception..."
sudo ./install.sh 2>&1 | tail -40

echo ""
echo "[2/5] Verifying dlsym symbol in libvgpu-cuda.so..."
if nm -D ~/phase3/guest-shim/libvgpu-cuda.so 2>/dev/null | grep -q " dlsym"; then
    echo "  ✓ dlsym symbol found"
    nm -D ~/phase3/guest-shim/libvgpu-cuda.so 2>/dev/null | grep " dlsym" | head -3
else
    echo "  ⚠ dlsym symbol not found - checking build..."
    nm -D ~/phase3/guest-shim/libvgpu-cuda.so 2>/dev/null | head -5
fi

echo ""
echo "[3/5] Restarting Ollama service..."
sudo systemctl restart ollama
echo "  ✓ Ollama restarted"

echo ""
echo "[4/5] Waiting for discovery (8 seconds)..."
sleep 8

echo ""
echo "[5/5] Checking logs for dlsym interception..."
echo ""
echo "--- dlsym/libggml-cuda messages ---"
sudo journalctl -u ollama -n 300 --no-pager 2>/dev/null | grep -E "(dlsym|libggml-cuda)" | head -20 || echo "  (No dlsym messages found)"

echo ""
echo "--- compute capability ---"
sudo journalctl -u ollama -n 300 --no-pager 2>/dev/null | grep -i "compute" | head -10 || echo "  (No compute messages found)"

echo ""
echo "--- GPU detection status ---"
sudo journalctl -u ollama -n 300 --no-pager 2>/dev/null | grep -iE "(gpu|device|didn't fully)" | tail -10 || echo "  (No GPU messages found)"

echo ""
echo "======================================================================"
echo "DEPLOYMENT COMPLETE"
echo "======================================================================"
echo ""
echo "Check the logs above for:"
echo "  1. dlsym interception messages"
echo "  2. compute=9.0 (instead of compute=0.0)"
echo "  3. GPU not filtered as 'didn't fully initialize'"
echo ""
