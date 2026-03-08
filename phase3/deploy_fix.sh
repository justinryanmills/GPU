#!/bin/bash

VM="test-5@10.25.33.15"
PASSWORD="Calvin@123"

echo "======================================================================"
echo "DEPLOYING FIXED SHIM"
echo "======================================================================"

echo ""
echo "[1] Copying fixed libvgpu_cuda.c..."
sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
    /home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c \
    $VM:~/phase3/guest-shim/libvgpu_cuda.c

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $VM << 'ENDSSH'
set -e

PASSWORD="Calvin@123"

echo ""
echo "[2] Rebuilding shim..."
cd ~/phase3/guest-shim
echo "$PASSWORD" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1

if [ -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "  Build successful"
    ls -lh /usr/lib64/libvgpu-cuda.so
    
    echo ""
    echo "[3] Checking if vGPU device appears..."
    lspci | grep -i "2331\|3d controller" || echo "  Device not yet visible"
    lspci -nn | grep -E "10de:2331|0302.*10de" || echo "  Device details not found"
    
    echo ""
    echo "[4] Restarting Ollama..."
    echo "$PASSWORD" | sudo -S systemctl restart ollama
    sleep 25
    
    if systemctl is-active --quiet ollama; then
        echo "  Ollama running"
        
        echo ""
        echo "[5] Running test inference..."
        timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -5
        
        echo ""
        echo "[6] Checking GPU mode..."
        echo "$PASSWORD" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep "library=" | tail -5
        
        echo ""
        echo "[7] Checking shim logs..."
        echo "$PASSWORD" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep -E "libvgpu-cuda|cuInit|VGPU-STUB|Found" | tail -10
    else
        echo "  ✗ Ollama failed to start"
        echo "$PASSWORD" | sudo -S journalctl -u ollama -n 20 --no-pager | tail -15
    fi
else
    echo "  ✗ Build failed"
fi

echo ""
echo "======================================================================"
echo "DEPLOYMENT COMPLETE"
echo "======================================================================"
ENDSSH

echo ""
echo "Done!"
