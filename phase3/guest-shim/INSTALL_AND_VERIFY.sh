#!/bin/bash
# Complete installation and verification script for Ollama GPU detection
# This script does everything needed to make Ollama detect the vGPU

set -e

PASS="Calvin@123"
OUTPUT_FILE="/tmp/install_verify_results.txt"

exec > "$OUTPUT_FILE" 2>&1

echo "======================================================================"
echo "OLLAMA GPU DETECTION - COMPLETE INSTALLATION AND VERIFICATION"
echo "Date: $(date)"
echo "======================================================================"
echo ""

# Step 1: Configure /etc/ld.so.preload (system-wide library preloading)
echo ">>> STEP 1: Configuring /etc/ld.so.preload"
echo "$PASS" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload && echo /usr/lib64/libvgpu-nvml.so >> /etc/ld.so.preload && chmod 644 /etc/ld.so.preload" 2>&1
echo "Contents:"
echo "$PASS" | sudo -S cat /etc/ld.so.preload 2>&1
if [ -f /etc/ld.so.preload ] && grep -q "libvgpu-cuda.so" /etc/ld.so.preload; then
    echo "✓ /etc/ld.so.preload configured correctly"
else
    echo "✗ /etc/ld.so.preload configuration FAILED"
    exit 1
fi
echo ""

# Step 2: Build LD_AUDIT interceptor
echo ">>> STEP 2: Building LD_AUDIT interceptor"
cd ~/phase3/guest-shim 2>&1
if [ -f ld_audit_interceptor.c ]; then
    echo "$PASS" | sudo -S gcc -shared -fPIC -o /usr/lib64/libldaudit_cuda.so ld_audit_interceptor.c -ldl -O2 -Wall 2>&1
    if [ -f /usr/lib64/libldaudit_cuda.so ]; then
        echo "✓ LD_AUDIT interceptor built successfully"
        ls -la /usr/lib64/libldaudit_cuda.so
    else
        echo "✗ LD_AUDIT build FAILED"
    fi
else
    echo "✗ ld_audit_interceptor.c not found"
fi
echo ""

# Step 3: Build force_load_shim
echo ">>> STEP 3: Building force_load_shim"
if [ -f force_load_shim.c ]; then
    echo "$PASS" | sudo -S gcc -o /usr/local/bin/force_load_shim force_load_shim.c -ldl -O2 -Wall 2>&1
    if [ -f /usr/local/bin/force_load_shim ]; then
        echo "✓ force_load_shim built successfully"
        ls -la /usr/local/bin/force_load_shim
    else
        echo "✗ force_load_shim build FAILED"
    fi
else
    echo "✗ force_load_shim.c not found"
fi
echo ""

# Step 4: Create systemd override for Ollama
echo ">>> STEP 4: Creating Ollama systemd override"
echo "$PASS" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d
echo "$PASS" | sudo -S bash -c 'cat > /etc/systemd/system/ollama.service.d/override.conf << "OVR"
[Service]
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so /usr/lib64/libvgpu-nvml.so"
OVR'
if [ -f /etc/systemd/system/ollama.service.d/override.conf ]; then
    echo "✓ Systemd override created"
    echo "$PASS" | sudo -S cat /etc/systemd/system/ollama.service.d/override.conf
else
    echo "✗ Systemd override creation FAILED"
fi
echo ""

# Step 5: Restart Ollama
echo ">>> STEP 5: Restarting Ollama"
echo "$PASS" | sudo -S systemctl daemon-reload 2>&1
echo "$PASS" | sudo -S systemctl restart ollama 2>&1
sleep 5
echo "$PASS" | sudo -S systemctl is-active ollama 2>&1
echo ""

# Step 6: Check Ollama logs for shim loading
echo ">>> STEP 6: Checking Ollama logs for shim loading"
echo "$PASS" | sudo -S journalctl -u ollama -n 200 --no-pager 2>&1 | grep -iE "libvgpu|LOADED|cuInit|cuda|gpu" | head -30
echo ""

# Step 7: Check Ollama process libraries
echo ">>> STEP 7: Checking Ollama process libraries"
OLLAMA_PID=$(echo "$PASS" | sudo -S pidof ollama | head -1)
if [ -n "$OLLAMA_PID" ]; then
    echo "Ollama PID: $OLLAMA_PID"
    echo "Libraries containing 'vgpu' or 'cuda':"
    echo "$PASS" | sudo -S cat /proc/$OLLAMA_PID/maps 2>&1 | grep -E "libvgpu|libcuda" | head -10
    if echo "$PASS" | sudo -S cat /proc/$OLLAMA_PID/maps 2>&1 | grep -q "libvgpu"; then
        echo "✓ Shim libraries are loaded in Ollama process"
    else
        echo "✗ Shim libraries NOT found in Ollama process"
    fi
else
    echo "✗ Ollama process not found"
fi
echo ""

# Step 8: Check shim log files
echo ">>> STEP 8: Checking shim log files"
if ls /tmp/vgpu-shim-cuda-*.log 2>/dev/null; then
    echo "Shim log files found:"
    for logfile in /tmp/vgpu-shim-cuda-*.log; do
        echo "--- $logfile ---"
        cat "$logfile" 2>&1
    done
else
    echo "No shim log files found (shim constructor may not have run)"
fi
echo ""

# Step 9: Test Ollama GPU detection
echo ">>> STEP 9: Testing Ollama GPU detection"
OLLAMA_INFO=$(ollama info 2>&1 | head -50)
echo "$OLLAMA_INFO"
echo ""

# Check for GPU indicators
if echo "$OLLAMA_INFO" | grep -qiE "gpu|cuda|nvidia"; then
    echo "======================================================================"
    echo "✓ SUCCESS: GPU DETECTION INDICATORS FOUND IN OLLAMA OUTPUT"
    echo "======================================================================"
    GPU_DETECTED=1
else
    echo "======================================================================"
    echo "✗ WARNING: NO GPU DETECTION IN OLLAMA OUTPUT"
    echo "======================================================================"
    GPU_DETECTED=0
fi
echo ""

# Step 10: Final status summary
echo ">>> STEP 10: Final status summary"
echo "ld.so.preload:"
echo "$PASS" | sudo -S cat /etc/ld.so.preload 2>&1
echo ""
echo "Installed files:"
ls -la /usr/lib64/libvgpu-cuda.so /usr/lib64/libvgpu-nvml.so /usr/lib64/libldaudit_cuda.so /usr/local/bin/force_load_shim 2>&1
echo ""

echo "======================================================================"
echo "INSTALLATION COMPLETE"
echo "Results saved to: $OUTPUT_FILE"
if [ "$GPU_DETECTED" = "1" ]; then
    echo "STATUS: ✓ GPU DETECTED"
else
    echo "STATUS: ✗ GPU NOT DETECTED - Check logs above for issues"
fi
echo "======================================================================"
