#!/bin/bash
set -e
exec > /tmp/complete_deploy_results.txt 2>&1

echo "=== COMPLETE DEPLOYMENT ==="
date

PASSWORD="Calvin@123"

cd ~/phase3/guest-shim
echo "$PASSWORD" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1

if [ -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "BUILD_SUCCESS"
    ls -lh /usr/lib64/libvgpu-cuda.so
else
    echo "BUILD_FAILED"
    exit 1
fi

echo /usr/lib64/libvgpu-cuda.so | echo "$PASSWORD" | sudo -S tee /etc/ld.so.preload
echo "$PASSWORD" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d
echo -e "[Service]\nType=simple" | echo "$PASSWORD" | sudo -S tee /etc/systemd/system/ollama.service.d/override.conf
echo "$PASSWORD" | sudo -S systemctl daemon-reload

echo "$PASSWORD" | sudo -S systemctl start ollama
sleep 25

if systemctl is-active --quiet ollama; then
    echo "OLLAMA_ACTIVE"
    
    timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -5
    
    echo ""
    echo "LIBRARY_MODE:"
    echo "$PASSWORD" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep "library=" | tail -5
else
    echo "OLLAMA_FAILED"
fi

echo ""
echo "=== COMPLETE ==="
date
