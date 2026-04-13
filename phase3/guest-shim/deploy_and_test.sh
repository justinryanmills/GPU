#!/bin/bash
set -e

LOG_FILE="/tmp/vgpu_deploy_$(date +%s).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== VGPU CUDA Shim Deployment and Test ==="
echo "Timestamp: $(date)"
echo ""

# Step 1: Ensure we have the source files
cd ~/phase3/guest-shim || {
    echo "ERROR: ~/phase3/guest-shim not found"
    echo "Creating directory structure..."
    mkdir -p ~/phase3/guest-shim
    cd ~/phase3/guest-shim
}

# Check if libvgpu_cuda.c exists
if [ ! -f libvgpu_cuda.c ]; then
    echo "ERROR: libvgpu_cuda.c not found in ~/phase3/guest-shim"
    exit 1
fi

# Step 2: Rebuild the CUDA shim
echo "=== Step 1: Rebuilding libvgpu-cuda.so ==="
sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1

if [ ! -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "ERROR: Failed to build libvgpu-cuda.so"
    exit 1
fi

echo "✓ Library built: $(ls -lh /usr/lib64/libvgpu-cuda.so | awk '{print $5, $9}')"

# Step 3: Verify symlinks
echo ""
echo "=== Step 2: Checking symlinks ==="
if [ -L /usr/lib64/libcuda.so.1 ]; then
    echo "✓ /usr/lib64/libcuda.so.1 -> $(readlink /usr/lib64/libcuda.so.1)"
else
    echo "Creating symlink..."
    sudo ln -sf /usr/lib64/libvgpu-cuda.so /usr/lib64/libcuda.so.1
fi

# Step 4: Check /etc/ld.so.preload
echo ""
echo "=== Step 3: Checking /etc/ld.so.preload ==="
if grep -q "libvgpu-cuda.so" /etc/ld.so.preload 2>/dev/null; then
    echo "✓ libvgpu-cuda.so is in /etc/ld.so.preload"
    cat /etc/ld.so.preload
else
    echo "Adding libvgpu-cuda.so to /etc/ld.so.preload..."
    echo "/usr/lib64/libvgpu-cuda.so" | sudo tee -a /etc/ld.so.preload
fi

# Step 5: Restart Ollama
echo ""
echo "=== Step 4: Restarting Ollama ==="
sudo systemctl stop ollama
sleep 2
sudo systemctl start ollama
sleep 8

if systemctl is-active --quiet ollama; then
    echo "✓ Ollama service is running"
else
    echo "WARNING: Ollama service may not be running"
    sudo systemctl status ollama --no-pager -l | head -20
fi

# Step 6: Check logs for shim loading
echo ""
echo "=== Step 5: Checking for shim loading ==="
sudo journalctl -u ollama -n 200 --no-pager | grep -iE "libvgpu|LOADED|Pre-initializ|cuInit" | head -20 || echo "No shim loading messages found"

# Step 7: Check library mode
echo ""
echo "=== Step 6: Checking library mode (before test) ==="
sudo journalctl -u ollama -n 300 --no-pager | grep -E "library=" | tail -10 || echo "No library mode entries found"

# Step 8: Run test inference
echo ""
echo "=== Step 7: Running test inference ==="
timeout 30 ollama run llama3.2:1b "test" 2>&1 | head -20 || echo "Inference test completed or timed out"

# Step 9: Check library mode after inference
echo ""
echo "=== Step 8: Checking library mode (after test) ==="
sudo journalctl -u ollama --since "2 minutes ago" --no-pager | grep -E "library=" | tail -10 || echo "No library mode entries found"

# Step 10: Detailed status
echo ""
echo "=== Step 9: Detailed status ==="
echo "Ollama process info:"
ps aux | grep -E "[o]llama" | head -5

echo ""
echo "Loaded libraries in Ollama process:"
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$OLLAMA_PID" ]; then
    sudo cat /proc/$OLLAMA_PID/maps | grep -E "cuda|vgpu" | head -10 || echo "No CUDA/vGPU libraries found in process maps"
else
    echo "Ollama process not found"
fi

echo ""
echo "=== Deployment complete ==="
echo "Log file: $LOG_FILE"
echo ""
echo "To check results manually:"
echo "  sudo journalctl -u ollama --since '5 minutes ago' | grep -E 'library='"
echo "  cat $LOG_FILE"
