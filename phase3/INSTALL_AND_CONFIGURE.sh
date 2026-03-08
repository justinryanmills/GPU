#!/bin/bash
set -e
PASSWORD="Calvin@123"
RESULTS="/tmp/install_results.txt"
exec > "$RESULTS" 2>&1

echo "=== INSTALLATION AND CONFIGURATION START ==="
date

sudo_cmd() {
    echo "$PASSWORD" | sudo -S "$@"
}

echo ""
echo "STEP 1: Installing curl..."
if ! command -v curl > /dev/null 2>&1; then
    sudo_cmd apt-get update -qq
    sudo_cmd apt-get install -y curl
fi
curl --version | head -1

echo ""
echo "STEP 2: Installing Ollama..."
if [ ! -f /usr/local/bin/ollama ]; then
    curl -fsSL https://ollama.com/install.sh | sh
    sleep 5
fi
/usr/local/bin/ollama --version || echo "Version check failed"

echo ""
echo "STEP 3: Making binary executable..."
sudo_cmd chmod +x /usr/local/bin/ollama

echo ""
echo "STEP 4: Configuring systemd service..."
sudo_cmd mkdir -p /etc/systemd/system/ollama.service.d
cat << EOF | sudo_cmd tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Type=simple
Restart=always
RestartSec=5
EOF
sudo_cmd mkdir -p /usr/share/ollama /var/lib/ollama
sudo_cmd chmod 755 /var/lib/ollama
sudo_cmd systemctl daemon-reload

echo ""
echo "STEP 5: Starting Ollama..."
sudo_cmd systemctl start ollama
sleep 25

if systemctl is-active --quiet ollama; then
    echo "OLLAMA_STARTED"
    OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
    echo "PID:$OLLAMA_PID"
else
    echo "OLLAMA_START_FAILED"
    journalctl -u ollama -n 100 --no-pager | tail -50
    exit 1
fi

echo ""
echo "STEP 6: Building shim library..."
cd ~/phase3/guest-shim
sudo_cmd gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1 | tail -5
if [ -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "SHIM_BUILT"
else
    echo "SHIM_BUILD_FAILED"
    exit 1
fi

echo ""
echo "STEP 7: Configuring preload..."
echo "/usr/lib64/libvgpu-cuda.so" | sudo_cmd tee /etc/ld.so.preload > /dev/null
cat /etc/ld.so.preload

echo ""
echo "STEP 8: Restarting with shim..."
sudo_cmd systemctl restart ollama
sleep 15

if systemctl is-active --quiet ollama; then
    echo "WITH_SHIM_STARTED"
    OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
    echo "PID:$OLLAMA_PID"
    
    if sudo_cmd cat /proc/$OLLAMA_PID/maps 2>&1 | grep -q "libvgpu-cuda"; then
        echo "SHIM_LOADED"
    else
        echo "SHIM_NOT_LOADED"
    fi
else
    echo "WITH_SHIM_FAILED"
    journalctl -u ollama -n 50 --no-pager | tail -20
    exit 1
fi

echo ""
echo "STEP 9: Running test inference..."
timeout 50 ollama run llama3.2:1b "hello" 2>&1 | head -10

echo ""
echo "STEP 10: Checking library mode..."
LIBRARY_MODE=$(journalctl -u ollama --since "4 minutes ago" --no-pager 2>&1 | grep -E "library=" | tail -5)
echo "LIBRARY_MODE_START"
echo "$LIBRARY_MODE"
echo "LIBRARY_MODE_END"

if echo "$LIBRARY_MODE" | grep -qi "library=cuda"; then
    echo "GPU_MODE_CONFIRMED"
else
    echo "GPU_MODE_NOT_CONFIRMED"
fi

echo ""
echo "=== INSTALLATION COMPLETE ==="
date
