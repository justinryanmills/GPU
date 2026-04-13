#!/bin/bash
# Comprehensive diagnostic and deployment script

exec > /tmp/vgpu_full_diagnostic.log 2>&1

echo "=========================================="
echo "VGPU FULL DIAGNOSTIC AND DEPLOYMENT"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Step 1: Check current state
echo "=== STEP 1: Current State ==="
echo ""

echo "[1.1] Source file check:"
if [ -f ~/phase3/guest-shim/libvgpu_cuda.c ]; then
    echo "  ✓ Source file exists"
    grep -n "Pre-initializing CUDA at load time" ~/phase3/guest-shim/libvgpu_cuda.c && echo "  ✓ cuInit fix is in source" || echo "  ✗ cuInit fix NOT in source"
else
    echo "  ✗ Source file NOT found"
fi
echo ""

echo "[1.2] Current shim library:"
ls -lh /usr/lib64/libvgpu-cuda.so 2>&1
echo ""

echo "[1.3] Ollama service status:"
systemctl is-active ollama 2>&1
systemctl status ollama --no-pager -l | head -10
echo ""

echo "[1.4] Current library mode (last 10 entries):"
sudo journalctl -u ollama -n 1000 --no-pager 2>&1 | grep -E "library=" | tail -10 || echo "  No library mode entries"
echo ""

echo "[1.5] Shim loading check:"
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$OLLAMA_PID" ]; then
    echo "  Ollama PID: $OLLAMA_PID"
    if [ -f "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log" ]; then
        echo "  Shim log file exists:"
        cat "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log" | tail -20
    else
        echo "  ✗ No shim log file found"
    fi
    
    echo ""
    echo "  Loaded libraries (CUDA/vGPU related):"
    sudo cat /proc/$OLLAMA_PID/maps 2>&1 | grep -E "cuda|vgpu" | head -10 || echo "    None found"
else
    echo "  ✗ Ollama not running"
fi
echo ""

# Step 2: Deploy fix
echo "=== STEP 2: Deploying Fix ==="
echo ""

cd ~/phase3/guest-shim 2>/dev/null || {
    echo "ERROR: Cannot cd to ~/phase3/guest-shim"
    exit 1
}

echo "[2.1] Rebuilding shim with cuInit fix..."
sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Build successful"
    ls -lh /usr/lib64/libvgpu-cuda.so
else
    echo "  ✗ Build failed"
    exit 1
fi
echo ""

echo "[2.2] Ensuring /etc/ld.so.preload:"
if ! grep -q "libvgpu-cuda.so" /etc/ld.so.preload 2>/dev/null; then
    echo "/usr/lib64/libvgpu-cuda.so" | sudo tee -a /etc/ld.so.preload
    echo "  ✓ Added to /etc/ld.so.preload"
else
    echo "  ✓ Already in /etc/ld.so.preload"
fi
cat /etc/ld.so.preload
echo ""

echo "[2.3] Restarting Ollama:"
sudo systemctl stop ollama
sleep 2
sudo systemctl start ollama
sleep 10

if systemctl is-active --quiet ollama; then
    echo "  ✓ Ollama is running"
else
    echo "  ✗ Ollama failed to start"
    sudo systemctl status ollama --no-pager -l | head -20
fi
echo ""

# Step 3: Verify
echo "=== STEP 3: Verification ==="
echo ""

echo "[3.1] Checking for pre-initialization:"
sleep 2
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$OLLAMA_PID" ]; then
    echo "  Ollama PID: $OLLAMA_PID"
    if [ -f "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log" ]; then
        echo "  Shim log entries:"
        cat "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log"
        echo ""
        if grep -q "Pre-initialization succeeded" "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log"; then
            echo "  ✓ Pre-initialization SUCCESS"
        else
            echo "  ✗ Pre-initialization not found in log"
        fi
    else
        echo "  ✗ No shim log file yet"
    fi
fi
echo ""

echo "[3.2] Running test inference:"
timeout 30 ollama run llama3.2:1b "test" 2>&1 | head -20
echo ""

echo "[3.3] Final library mode check:"
sudo journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep -E "library=" | tail -10 || echo "  No library mode entries"
echo ""

echo "[3.4] All recent Ollama logs (last 50 lines):"
sudo journalctl -u ollama -n 50 --no-pager 2>&1 | tail -50
echo ""

echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "Log saved to: /tmp/vgpu_full_diagnostic.log"
echo "=========================================="
