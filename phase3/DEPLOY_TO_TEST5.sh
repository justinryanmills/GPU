#!/bin/bash

set -e
VM="test-5@10.25.33.15"
PASSWORD="Calvin@123"
LOG="/tmp/deploy_test5.log"

exec > "$LOG" 2>&1

echo "=== DEPLOYMENT TO test-5 ==="
date

echo ""
echo "[1] Copying files..."
scp -o StrictHostKeyChecking=no \
    phase3/guest-shim/libvgpu_cuda.c \
    phase3/guest-shim/cuda_transport.c \
    phase3/guest-shim/cuda_transport.h \
    phase3/guest-shim/gpu_properties.h \
    phase3/include/cuda_protocol.h \
    ${VM}:~/phase3/guest-shim/ <<< "${PASSWORD}"

ssh -o StrictHostKeyChecking=no ${VM} "mkdir -p ~/phase3/include && cp ~/phase3/guest-shim/cuda_protocol.h ~/phase3/include/" <<< "${PASSWORD}"

echo ""
echo "[2] Installing and building..."
ssh -o StrictHostKeyChecking=no ${VM} << 'DEPLOY_SCRIPT' <<< "${PASSWORD}"
set -e
PASSWORD="${PASSWORD}"

echo "${PASSWORD}" | sudo -S apt-get update -qq
echo "${PASSWORD}" | sudo -S apt-get install -y curl build-essential 2>&1 | tail -3

if ! command -v ollama > /dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
    sleep 5
fi

cd ~/phase3/guest-shim
echo "${PASSWORD}" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1

if [ -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "BUILD_SUCCESS"
    ls -lh /usr/lib64/libvgpu-cuda.so
else
    echo "BUILD_FAILED"
    exit 1
fi

echo "${PASSWORD}" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload"
echo "${PASSWORD}" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d
echo -e "[Service]\nType=simple" | echo "${PASSWORD}" | sudo -S tee /etc/systemd/system/ollama.service.d/override.conf
echo "${PASSWORD}" | sudo -S systemctl daemon-reload
echo "${PASSWORD}" | sudo -S systemctl start ollama
sleep 25

if systemctl is-active --quiet ollama; then
    echo "OLLAMA_ACTIVE"
    
    timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -5
    echo "${PASSWORD}" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep "library=" | tail -3
else
    echo "OLLAMA_FAILED"
fi
DEPLOY_SCRIPT

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
date
