#!/bin/bash

PASS="Calvin@123"
OUTPUT_FILE="/tmp/diagnose_output.txt"

exec > "$OUTPUT_FILE" 2>&1

echo "======================================================================"
echo "COMPREHENSIVE DIAGNOSTIC AND FIX FOR OLLAMA GPU DETECTION"
echo "======================================================================"
date
echo ""

echo ">>> STEP 1: Fixing /etc/ld.so.preload"
echo "$PASS" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload && echo /usr/lib64/libvgpu-nvml.so >> /etc/ld.so.preload && chmod 644 /etc/ld.so.preload" 2>&1
echo "Contents:"
echo "$PASS" | sudo -S cat /etc/ld.so.preload 2>&1
echo ""

echo ">>> STEP 2: Building LD_AUDIT and force_load_shim"
cd ~/phase3/guest-shim 2>&1
echo "$PASS" | sudo -S gcc -shared -fPIC -o /usr/lib64/libldaudit_cuda.so ld_audit_interceptor.c -ldl -O2 -Wall 2>&1
echo "$PASS" | sudo -S gcc -o /usr/local/bin/force_load_shim force_load_shim.c -ldl -O2 -Wall 2>&1
ls -la /usr/lib64/libldaudit_cuda.so /usr/local/bin/force_load_shim 2>&1
echo ""

echo ">>> STEP 3: Checking Ollama process libraries"
OLLAMA_PID=$(echo "$PASS" | sudo -S pidof ollama | head -1)
if [ -n "$OLLAMA_PID" ]; then
    echo "Ollama PID: $OLLAMA_PID"
    echo "Libraries containing 'vgpu' or 'cuda':"
    echo "$PASS" | sudo -S cat /proc/$OLLAMA_PID/maps 2>&1 | grep -E "libvgpu|libcuda|libnvidia" | head -20
else
    echo "Ollama process not found"
fi
echo ""

echo ">>> STEP 4: Testing shim loading"
gcc -o /tmp/test_shim_load ~/phase3/guest-shim/test_shim_load.c -ldl 2>&1
LD_PRELOAD=/usr/lib64/libvgpu-cuda.so /tmp/test_shim_load 2>&1
echo ""

echo ">>> STEP 5: Checking shim library symbols"
nm -D /usr/lib64/libvgpu-cuda.so 2>&1 | grep -E "cuInit|cuGetProcAddress" | head -10
echo ""

echo ">>> STEP 6: Creating systemd override"
echo "$PASS" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d
echo "$PASS" | sudo -S bash -c 'cat > /etc/systemd/system/ollama.service.d/override.conf << "OVR"
[Service]
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so /usr/lib64/libvgpu-nvml.so"
OVR'
echo "$PASS" | sudo -S cat /etc/systemd/system/ollama.service.d/override.conf 2>&1
echo ""

echo ">>> STEP 7: Restarting Ollama"
echo "$PASS" | sudo -S systemctl daemon-reload 2>&1
echo "$PASS" | sudo -S systemctl restart ollama 2>&1
sleep 5
echo "$PASS" | sudo -S systemctl status ollama --no-pager -l 2>&1 | head -20
echo ""

echo ">>> STEP 8: Checking Ollama logs"
echo "$PASS" | sudo -S journalctl -u ollama -n 200 --no-pager 2>&1 | grep -iE "libvgpu|LOADED|cuInit|cuda|gpu" | head -30
echo ""

echo ">>> STEP 9: Testing Ollama GPU detection"
ollama info 2>&1 | head -100
echo ""

echo ">>> STEP 10: Final verification"
echo "ld.so.preload:"
echo "$PASS" | sudo -S cat /etc/ld.so.preload 2>&1
echo ""
echo "Installed files:"
ls -la /usr/lib64/libvgpu-cuda.so /usr/lib64/libvgpu-nvml.so /usr/lib64/libldaudit_cuda.so /usr/local/bin/force_load_shim 2>&1
echo ""

echo "======================================================================"
echo "DIAGNOSTIC COMPLETE - Output saved to $OUTPUT_FILE"
echo "======================================================================"
