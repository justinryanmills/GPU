#!/bin/bash

set -e
exec > /tmp/final_deploy.log 2>&1

echo "================================================================"
echo "FINAL DEPLOYMENT SCRIPT"
echo "================================================================"
date

PASSWORD="Calvin@123"

echo ""
echo "[1] Verifying files..."
cd ~/phase3/guest-shim
for f in libvgpu_cuda.c cuda_transport.c cuda_transport.h gpu_properties.h; do
    if [ -f "$f" ]; then
        echo "  $f"
    else
        echo "  ✗ MISSING: $f"
        exit 1
    fi
done

if [ -f ../include/cuda_protocol.h ]; then
    echo "  ../include/cuda_protocol.h"
else
    echo "  ✗ MISSING: ../include/cuda_protocol.h"
    exit 1
fi

echo ""
echo "[2] Building shim..."
echo "$PASSWORD" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1

if [ -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "  BUILD SUCCESS"
    ls -lh /usr/lib64/libvgpu-cuda.so
else
    echo "  ✗ BUILD FAILED"
    exit 1
fi

echo ""
echo "[3] Configuring..."
echo /usr/lib64/libvgpu-cuda.so | echo "$PASSWORD" | sudo -S tee /etc/ld.so.preload > /dev/null
cat /etc/ld.so.preload

echo "$PASSWORD" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d
echo -e "[Service]\nType=simple" | echo "$PASSWORD" | sudo -S tee /etc/systemd/system/ollama.service.d/override.conf
echo "$PASSWORD" | sudo -S systemctl daemon-reload

echo ""
echo "[4] Starting Ollama..."
echo "$PASSWORD" | sudo -S systemctl start ollama
sleep 25

if systemctl is-active --quiet ollama; then
    echo "  OLLAMA ACTIVE"
    OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
    echo "  PID: $OLLAMA_PID"
    
    if echo "$PASSWORD" | sudo -S cat /proc/$OLLAMA_PID/maps 2>&1 | grep -q "libvgpu-cuda"; then
        echo "  SHIM LOADED"
    fi
else
    echo "  ✗ OLLAMA FAILED"
    echo "$PASSWORD" | sudo -S journalctl -u ollama -n 30 --no-pager | tail -20
    exit 1
fi

echo ""
echo "[5] Testing inference..."
timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -5

echo ""
echo "[6] Checking logs..."
echo "Library mode:"
echo "$PASSWORD" | sudo -S journalctl -u ollama --since "3 minutes ago" --no-pager 2>&1 | grep "library=" | tail -5

echo ""
echo "Shim initialization:"
echo "$PASSWORD" | sudo -S journalctl -u ollama --since "3 minutes ago" --no-pager 2>&1 | grep -E "libvgpu-cuda|cuInit|System process" | tail -10

echo ""
echo "================================================================"
echo "DEPLOYMENT COMPLETE"
echo "================================================================"
date
