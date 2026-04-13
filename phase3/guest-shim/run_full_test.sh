#!/bin/bash
# Comprehensive test script for shim injection

set -euo pipefail

OUTPUT_FILE="/tmp/shim_test_results.txt"
rm -f "$OUTPUT_FILE"

exec > >(tee -a "$OUTPUT_FILE") 2>&1

echo "======================================================================"
echo "COMPREHENSIVE SHIM INJECTION TEST"
echo "======================================================================"
echo ""

echo "=== STEP 1: Verify Installation Files ==="
ls -la /etc/ld.so.preload /usr/lib64/libldaudit_cuda.so /usr/local/bin/force_load_shim 2>&1 || echo "Some files missing"
echo ""

echo "=== STEP 2: Check /etc/ld.so.preload Content ==="
sudo cat /etc/ld.so.preload 2>&1 || echo "No preload file"
echo ""

echo "=== STEP 3: Check Shim Libraries ==="
ls -la /usr/lib64/libvgpu-*.so 2>&1
echo ""

echo "=== STEP 4: Test Shim Loading with Test Program ==="
if [ -f ~/phase3/guest-shim/test_shim_load.c ]; then
    gcc -o /tmp/test_shim_load ~/phase3/guest-shim/test_shim_load.c -ldl 2>&1
    /tmp/test_shim_load 2>&1
else
    echo "Test program not found"
fi
echo ""

echo "=== STEP 5: Check Ollama Service Status ==="
sudo systemctl status ollama --no-pager 2>&1 | head -20
echo ""

echo "=== STEP 6: Check Ollama Logs for Shim Loading ==="
sudo journalctl -u ollama -n 100 --no-pager 2>&1 | grep -iE "libvgpu|LOADED|cuInit|cuda|gpu" | head -30
echo ""

echo "=== STEP 7: Check Shim Log Files ==="
ls -la /tmp/vgpu-shim-*.log 2>&1 | head -10
echo "--- Recent shim logs ---"
sudo cat /tmp/vgpu-shim-cuda-*.log 2>&1 | tail -20 || echo "No shim logs found"
echo ""

echo "=== STEP 8: Test Ollama GPU Detection ==="
ollama info 2>&1 | head -50
echo ""

echo "=== STEP 9: Test Ollama Inference (Quick) ==="
timeout 10 ollama run llama3.2:1b "test" 2>&1 | head -20 || echo "Inference test timed out or failed"
echo ""

echo "======================================================================"
echo "TEST COMPLETE - Results saved to $OUTPUT_FILE"
echo "======================================================================"
