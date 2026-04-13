#!/bin/bash
# Complete fix deployment and verification

{
echo "=========================================="
echo "VGPU CUDA SHIM - COMPLETE FIX DEPLOYMENT"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Step 1: Verify source has the fix
echo "STEP 1: Verifying source code fix..."
cd ~/phase3/guest-shim 2>/dev/null || {
    echo "ERROR: ~/phase3/guest-shim not found"
    exit 1
}

if grep -q "Pre-initializing CUDA at load time" libvgpu_cuda.c; then
    echo "✓ cuInit fix found in source code"
    grep -n "Pre-initializing CUDA at load time" libvgpu_cuda.c | head -1
else
    echo "✗ cuInit fix NOT found - need to deploy source"
    exit 1
fi
echo ""

# Step 2: Rebuild
echo "STEP 2: Rebuilding CUDA shim..."
sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall 2>&1

if [ $? -ne 0 ]; then
    echo "✗ Build failed"
    exit 1
fi

echo "✓ Build successful"
ls -lh /usr/lib64/libvgpu-cuda.so
echo ""

# Step 3: Ensure preload
echo "STEP 3: Ensuring /etc/ld.so.preload..."
if ! grep -q "libvgpu-cuda.so" /etc/ld.so.preload 2>/dev/null; then
    echo "/usr/lib64/libvgpu-cuda.so" | sudo tee -a /etc/ld.so.preload > /dev/null
    echo "✓ Added to /etc/ld.so.preload"
fi
echo ""

# Step 4: Restart Ollama
echo "STEP 4: Restarting Ollama..."
sudo systemctl stop ollama
sleep 2
sudo systemctl start ollama
sleep 10

if ! systemctl is-active --quiet ollama; then
    echo "✗ Ollama failed to start"
    sudo systemctl status ollama --no-pager -l | head -10
    exit 1
fi
echo "✓ Ollama is running"
echo ""

# Step 5: Check shim loading
echo "STEP 5: Checking shim loading..."
sleep 3
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -z "$OLLAMA_PID" ]; then
    echo "✗ Ollama process not found"
    exit 1
fi

echo "Ollama PID: $OLLAMA_PID"

if [ -f "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log" ]; then
    echo "Shim log file exists:"
    cat "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log"
    echo ""
    
    if grep -q "Pre-initialization succeeded" "/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log"; then
        echo "✓ cuInit pre-initialization SUCCESS"
    else
        echo "✗ Pre-initialization not found in log"
    fi
else
    echo "✗ No shim log file found"
fi
echo ""

# Step 6: Run test
echo "STEP 6: Running test inference..."
timeout 30 ollama run llama3.2:1b "test" 2>&1 | head -15
echo ""

# Step 7: Check library mode
echo "STEP 7: Checking library mode..."
LIBRARY_MODE=$(sudo journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep -E "library=" | tail -5)
echo "$LIBRARY_MODE"
echo ""

# Step 8: Final status
echo "=========================================="
echo "FINAL STATUS"
echo "=========================================="

if echo "$LIBRARY_MODE" | grep -qi "library=cuda"; then
    echo "✓ SUCCESS! Ollama is using GPU mode (library=cuda)"
    echo "  The cuInit early initialization fix is working!"
    exit 0
elif echo "$LIBRARY_MODE" | grep -qi "library=cpu"; then
    echo "✗ Still using CPU mode (library=cpu)"
    echo "  Further investigation needed"
    exit 1
else
    echo "? Could not determine library mode"
    echo "  Check logs manually: sudo journalctl -u ollama --since '2 minutes ago' | grep library="
    exit 2
fi

} 2>&1 | tee /tmp/vgpu_complete_fix.log
