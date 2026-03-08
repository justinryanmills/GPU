#!/bin/bash
set -e
PASSWORD="Calvin@123"
VM="test-5@10.25.33.15"
RESULTS="/tmp/deployment_results.txt"

echo "================================================================"
echo "COMPLETE DEPLOYMENT TO test-5 VM"
echo "================================================================"
date

echo ""
echo "[1] Copying files to VM..."
scp -o StrictHostKeyChecking=no \
    phase3/guest-shim/libvgpu_cuda.c \
    phase3/guest-shim/cuda_transport.c \
    phase3/guest-shim/cuda_transport.h \
    phase3/guest-shim/gpu_properties.h \
    phase3/include/cuda_protocol.h \
    ${VM}:~/phase3/guest-shim/ 2>&1 | head -10

ssh -o StrictHostKeyChecking=no ${VM} "mkdir -p ~/phase3/include && cp ~/phase3/guest-shim/cuda_protocol.h ~/phase3/include/ 2>&1 || true" <<< "${PASSWORD}"

echo "  Files copied"

echo ""
echo "[2] Installing dependencies..."
ssh -o StrictHostKeyChecking=no ${VM} << EOF
echo "${PASSWORD}" | sudo -S apt-get update -qq
echo "${PASSWORD}" | sudo -S apt-get install -y curl build-essential 2>&1 | tail -5
EOF

echo ""
echo "[3] Installing Ollama..."
ssh -o StrictHostKeyChecking=no ${VM} << EOF
if ! command -v ollama > /dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
    sleep 5
fi
/usr/local/bin/ollama --version 2>&1 || echo "Version check failed"
EOF

echo ""
echo "[4] Building shim library..."
ssh -o StrictHostKeyChecking=no ${VM} << EOF
cd ~/phase3/guest-shim
echo "${PASSWORD}" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1 | tail -10

if [ -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "BUILD_SUCCESS"
    ls -lh /usr/lib64/libvgpu-cuda.so
else
    echo "BUILD_FAILED"
fi
EOF

echo ""
echo "[5] Configuring /etc/ld.so.preload..."
ssh -o StrictHostKeyChecking=no ${VM} << EOF
echo "${PASSWORD}" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload"
cat /etc/ld.so.preload
EOF

echo ""
echo "[6] Configuring Ollama service..."
ssh -o StrictHostKeyChecking=no ${VM} << EOF
echo "${PASSWORD}" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d
cat << 'SVC_EOF' | echo "${PASSWORD}" | sudo -S tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Type=simple
Restart=always
RestartSec=5
SVC_EOF
echo "${PASSWORD}" | sudo -S systemctl daemon-reload
EOF

echo ""
echo "[7] Starting Ollama..."
ssh -o StrictHostKeyChecking=no ${VM} << EOF
echo "${PASSWORD}" | sudo -S systemctl start ollama
sleep 25
systemctl is-active ollama && echo "OLLAMA_ACTIVE" || echo "OLLAMA_INACTIVE"
EOF

echo ""
echo "[8] Testing and verifying GPU mode..."
ssh -o StrictHostKeyChecking=no ${VM} << EOF
timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -8
echo "${PASSWORD}" | sudo -S journalctl -u ollama --since "3 minutes ago" --no-pager 2>&1 | grep "library=" | tail -5
EOF

echo ""
echo "================================================================"
echo "DEPLOYMENT COMPLETE"
echo "================================================================"
date
