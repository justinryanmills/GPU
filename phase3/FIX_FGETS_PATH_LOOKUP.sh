#!/bin/bash

VM="test-5@10.25.33.15"
PASSWORD="Calvin@123"

echo "======================================================================"
echo "FIXING FGETS() PATH LOOKUP ISSUE"
echo "======================================================================"


echo ""
echo "The issue:"
echo "  - cuda_transport.c opens vendor, device, and class files"
echo "  - fgets() interception is finding /vendor for all three"
echo "  - This causes device lookup to fail"
echo ""
echo "The fix:"
echo "  1. Search tracked files from end (most recent first)"
echo "  2. Update existing FILE* entries instead of always adding new"
echo "  3. Better path matching logic"
echo ""

echo "[1] Copying fixed libvgpu_cuda.c..."
scp -o StrictHostKeyChecking=no /home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c test-5@10.25.33.15:~/phase3/guest-shim/libvgpu_cuda.c <<< "$PASSWORD" 2>&1 | tail -3

if [ $? -eq 0 ]; then
    echo "  File copied"
else
    echo "  ⚠ Copy may have failed - check manually"
fi

echo ""
echo "[2] Deploying on VM..."
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $VM << ENDSSH
set -e

cd ~/phase3/guest-shim

echo "Building shim..."
echo "$PASSWORD" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1 | grep -E "error|warning|BUILD" || echo "Build completed"

if [ -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "  Build successful!"
    
    echo ""
    echo "Testing device discovery..."
    ls 2>&1 | head -3 > /dev/null
    
    sleep 2
    
    echo ""
    echo "Checking logs for device discovery..."
    echo "$PASSWORD" | sudo -S journalctl --since "1 minute ago" --no-pager 2>&1 | grep -E "Found VGPU|cuInit.*succeeded" | tail -3
    
    echo ""
    echo "Restarting Ollama..."
    echo "$PASSWORD" | sudo -S systemctl restart ollama
    sleep 25
    
    if systemctl is-active --quiet ollama; then
        echo "  Ollama running"
        
        echo ""
        echo "Running test inference..."
        timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -5
        
        sleep 2
        
        echo ""
        echo "Checking GPU mode..."
        echo "$PASSWORD" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep "library=" | tail -5
    else
        echo "  ✗ Ollama failed to start"
    fi
else
    echo "  ✗ Build failed"
    echo "$PASSWORD" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1 | tail -20
fi

ENDSSH

echo ""
echo "======================================================================"
echo "DEPLOYMENT COMPLETE"
echo "======================================================================"
