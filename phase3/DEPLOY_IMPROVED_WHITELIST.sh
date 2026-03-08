#!/bin/bash

set -e

echo "=== DEPLOYING IMPROVED WHITELIST FIX ==="
echo ""

echo "Step 1: Copy improved libvgpu_cuda.c to VM"
echo "Run from local machine:"
echo "  scp /home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c test-7@10.25.33.17:~/phase3/guest-shim/libvgpu_cuda.c"
echo ""

echo "Step 2: Building shim..."
cd ~/phase3/guest-shim
sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2

if [ ! -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "✗ Build failed"
    exit 1
fi

echo "Build successful"
ls -lh /usr/lib64/libvgpu-cuda.so

echo ""
echo "Step 3: Verifying preload configuration..."
if [ "$(cat /etc/ld.so.preload)" = "/usr/lib64/libvgpu-cuda.so" ]; then
    echo "Preload configured correctly"
else
    echo "Configuring preload..."
    echo /usr/lib64/libvgpu-cuda.so | sudo tee /etc/ld.so.preload
fi

echo ""
echo "Step 4: Testing lspci (critical test)..."
lspci | grep -i nvidia
if [ $? -eq 0 ]; then
    echo "lspci works (no crash)"
else
    echo "✗ lspci issue"
fi

echo ""
echo "Step 5: Restarting Ollama..."
sudo systemctl restart ollama
sleep 30

echo ""
echo "Step 6: Checking Ollama status..."
systemctl is-active ollama

echo ""
echo "Step 7: Checking GPU mode..."
journalctl -u ollama -n 300 --no-pager | grep -iE "library=" | tail -5

echo ""
echo "Step 8: Checking Ollama process info..."
OLLAMA_PID=$(pgrep -f ollama | head -1)
if [ -n "$OLLAMA_PID" ]; then
    echo "Ollama PID: $OLLAMA_PID"
    echo "Process comm: $(cat /proc/$OLLAMA_PID/comm)"
    echo "Process cmdline: $(cat /proc/$OLLAMA_PID/cmdline | tr '\0' ' ' | head -c 100)"
    echo ""
    if cat /proc/$OLLAMA_PID/maps | grep -q libvgpu-cuda; then
        echo "Shim is loaded in Ollama"
    else
        echo "✗ Shim NOT loaded in Ollama"
    fi
fi

echo ""
echo "Step 9: Final GPU mode verification..."
journalctl -u ollama -n 500 --no-pager | grep -iE "library=|gpu|device|pci_id" | tail -10

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo ""
echo "Expected result:"
echo "  - library=gpu or library=cuda (NOT library=cpu)"
echo "  - pci_id=\"0000:00:05.0\" (NOT empty)"
echo "  - lspci works without crashes"
