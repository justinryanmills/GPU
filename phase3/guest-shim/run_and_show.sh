#!/bin/bash
# Simple script that runs installation and shows all output

PASS="Calvin@123"

exec 2>&1  # Merge stderr to stdout

echo "======================================================================"
echo "SHIM INSTALLATION AND TEST - $(date)"
echo "======================================================================"
echo ""

# Step 1: Install ld.so.preload
echo ">>> STEP 1: Creating /etc/ld.so.preload"
echo "$PASS" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload && echo /usr/lib64/libvgpu-nvml.so >> /etc/ld.so.preload && chmod 644 /etc/ld.so.preload" 2>&1
echo "$PASS" | sudo -S cat /etc/ld.so.preload 2>&1
echo ""

# Step 2: Build LD_AUDIT
echo ">>> STEP 2: Building LD_AUDIT interceptor"
cd ~/phase3/guest-shim 2>&1
echo "$PASS" | sudo -S gcc -shared -fPIC -o /usr/lib64/libldaudit_cuda.so ld_audit_interceptor.c -ldl -O2 -Wall 2>&1
ls -la /usr/lib64/libldaudit_cuda.so 2>&1
echo ""

# Step 3: Build force_load_shim
echo ">>> STEP 3: Building force_load_shim"
echo "$PASS" | sudo -S gcc -o /usr/local/bin/force_load_shim force_load_shim.c -ldl -O2 -Wall 2>&1
ls -la /usr/local/bin/force_load_shim 2>&1
echo ""

# Step 4: Test
echo ">>> STEP 4: Testing shim loading"
gcc -o /tmp/test_shim_load ~/phase3/guest-shim/test_shim_load.c -ldl 2>&1
/tmp/test_shim_load 2>&1
echo ""

# Step 5: Restart Ollama
echo ">>> STEP 5: Restarting Ollama"
echo "$PASS" | sudo -S systemctl restart ollama 2>&1
sleep 5
echo ""

# Step 6: Check logs
echo ">>> STEP 6: Checking Ollama logs for shim"
echo "$PASS" | sudo -S journalctl -u ollama -n 150 --no-pager 2>&1 | grep -iE "libvgpu|LOADED|cuInit|cuda|gpu" | head -25
echo ""

# Step 7: Test Ollama
echo ">>> STEP 7: Testing Ollama GPU detection"
ollama info 2>&1 | head -50
echo ""

# Step 8: Final status
echo ">>> STEP 8: Final status"
echo "ld.so.preload contents:"
echo "$PASS" | sudo -S cat /etc/ld.so.preload 2>&1
echo ""
echo "Installed files:"
ls -la /usr/lib64/libldaudit_cuda.so /usr/local/bin/force_load_shim 2>&1
echo ""

echo "======================================================================"
echo "COMPLETE"
echo "======================================================================"
