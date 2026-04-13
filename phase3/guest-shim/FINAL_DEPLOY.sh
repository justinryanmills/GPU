#!/bin/bash
# Final deployment script for cuInit fix

set -e

echo "=========================================="
echo "VGPU CUDA Shim Deployment - cuInit Fix"
echo "=========================================="
echo ""

cd ~/phase3/guest-shim || {
    echo "ERROR: ~/phase3/guest-shim not found"
    exit 1
}

echo "[1/4] Rebuilding libvgpu-cuda.so with cuInit fix..."
sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1 | grep -E "(error|warning|Building)" || true

if [ ! -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo "✓ Build complete: $(ls -lh /usr/lib64/libvgpu-cuda.so | awk '{print $5}')"
echo ""

echo "[2/4] Ensuring /etc/ld.so.preload is configured..."
if ! grep -q "libvgpu-cuda.so" /etc/ld.so.preload 2>/dev/null; then
    echo "/usr/lib64/libvgpu-cuda.so" | sudo tee -a /etc/ld.so.preload > /dev/null
    echo "✓ Added to /etc/ld.so.preload"
else
    echo "✓ Already in /etc/ld.so.preload"
fi
echo ""

echo "[3/4] Restarting Ollama service..."
sudo systemctl stop ollama
sleep 2
sudo systemctl start ollama
sleep 8

if systemctl is-active --quiet ollama; then
    echo "✓ Ollama is running"
else
    echo "WARNING: Ollama may not be running"
    sudo systemctl status ollama --no-pager -l | head -10
fi
echo ""

echo "[4/4] Checking for shim loading messages..."
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$OLLAMA_PID" ]; then
    echo "Ollama PID: $OLLAMA_PID"
    if [ -f "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log" ]; then
        echo "✓ Shim log file found:"
        cat "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log"
    else
        echo "? No shim log file found yet"
    fi
    
    echo ""
    echo "Checking loaded libraries:"
    sudo cat /proc/$OLLAMA_PID/maps | grep -E "cuda|vgpu" | head -5 || echo "No CUDA/vGPU libraries in process maps"
fi

echo ""
echo "=========================================="
echo "Checking Ollama logs for library mode..."
echo "=========================================="
sudo journalctl -u ollama -n 500 --no-pager 2>&1 | grep -E "library=|Pre-initializ|libvgpu" | tail -20 || echo "No relevant log entries found"

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: ollama run llama3.2:1b 'test'"
echo "2. Check logs: sudo journalctl -u ollama --since '1 minute ago' | grep library="
echo "3. Look for 'library=cuda' in the output"
