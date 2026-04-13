#!/bin/bash
# Comprehensive VM review script

REVIEW_FILE="/tmp/vm_review_$(date +%s).log"
exec > >(tee "$REVIEW_FILE") 2>&1

echo "=========================================="
echo "COMPREHENSIVE VM REVIEW"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "=========================================="
echo ""

# 1. System Information
echo "=== 1. SYSTEM INFORMATION ==="
echo "OS: $(uname -a)"
echo ""
echo "Disk Space:"
df -h / | tail -1
echo ""
echo "Memory:"
free -h | head -2
echo ""

# 2. Ollama Service Status
echo "=== 2. OLLAMA SERVICE STATUS ==="
systemctl is-active ollama && echo "Status: ACTIVE" || echo "Status: INACTIVE"
echo ""
echo "Service Details:"
systemctl status ollama --no-pager -l | head -15
echo ""

# 3. Ollama Process
echo "=== 3. OLLAMA PROCESS ==="
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$OLLAMA_PID" ]; then
    echo "Ollama PID: $OLLAMA_PID"
    echo ""
    echo "Process Details:"
    ps aux | grep -E "[o]llama" | head -3
    echo ""
    echo "Process Environment (LD_PRELOAD):"
    sudo cat /proc/$OLLAMA_PID/environ 2>/dev/null | tr '\0' '\n' | grep LD_PRELOAD || echo "  No LD_PRELOAD found"
else
    echo "No Ollama process found"
fi
echo ""

# 4. Shim Libraries
echo "=== 4. SHIM LIBRARY STATUS ==="
echo "CUDA Shim:"
ls -lh /usr/lib64/libvgpu-cuda.so 2>&1
file /usr/lib64/libvgpu-cuda.so 2>&1
echo ""
echo "NVML Shim:"
ls -lh /usr/lib64/libvgpu-nvml.so 2>&1
file /usr/lib64/libvgpu-nvml.so 2>&1
echo ""

# 5. ld.so.preload
echo "=== 5. LD.SO.PRELOAD ==="
if [ -f /etc/ld.so.preload ]; then
    echo "Contents:"
    cat /etc/ld.so.preload
else
    echo "File does not exist"
fi
echo ""

# 6. Loaded Libraries in Process
echo "=== 6. LOADED LIBRARIES IN OLLAMA ==="
if [ -n "$OLLAMA_PID" ]; then
    echo "CUDA/vGPU related libraries:"
    sudo cat /proc/$OLLAMA_PID/maps 2>&1 | grep -E "vgpu|cuda|nvidia" | head -15 || echo "  None found"
else
    echo "No Ollama process"
fi
echo ""

# 7. Shim Logs
echo "=== 7. SHIM LOG FILES ==="
if [ -n "$OLLAMA_PID" ]; then
    SHIM_LOG="/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log"
    if [ -f "$SHIM_LOG" ]; then
        echo "CUDA Shim Log ($SHIM_LOG):"
        cat "$SHIM_LOG"
        echo ""
        if grep -q "Pre-initialization succeeded" "$SHIM_LOG"; then
            echo "✓ Pre-initialization SUCCESS found in log"
        else
            echo "✗ Pre-initialization success NOT found"
        fi
    else
        echo "No shim log file found: $SHIM_LOG"
    fi
else
    echo "No Ollama process"
fi
echo ""

# 8. Source Files
echo "=== 8. SOURCE FILES ==="
if [ -d ~/phase3/guest-shim ]; then
    echo "Source directory exists:"
    ls -la ~/phase3/guest-shim/ | head -20
    echo ""
    
    if [ -f ~/phase3/guest-shim/libvgpu_cuda.c ]; then
        echo "Checking for cuInit fix:"
        grep -n "Pre-initializing CUDA at load time" ~/phase3/guest-shim/libvgpu_cuda.c | head -1 || echo "  ✗ Fix NOT found"
        echo ""
        echo "Checking for thread-safe fix:"
        grep -n "__sync_bool_compare_and_swap" ~/phase3/guest-shim/libvgpu_cuda.c | head -1 || echo "  ✗ Thread-safe fix NOT found"
    else
        echo "✗ libvgpu_cuda.c not found"
    fi
else
    echo "✗ Source directory not found: ~/phase3/guest-shim"
fi
echo ""

# 9. Ollama Logs - Library Mode
echo "=== 9. OLLAMA LOGS - LIBRARY MODE ==="
sudo journalctl -u ollama -n 500 --no-pager 2>&1 | grep -E "library=" | tail -10 || echo "No library mode entries found"
echo ""

# 10. Recent Errors
echo "=== 10. RECENT ERRORS ==="
sudo journalctl -u ollama -n 200 --no-pager 2>&1 | grep -iE "error|fail|panic|crash" | tail -10 || echo "No errors found"
echo ""

# 11. Deployment Scripts
echo "=== 11. DEPLOYMENT SCRIPTS ==="
if [ -f ~/safe_deploy.sh ]; then
    echo "✓ ~/safe_deploy.sh exists"
    ls -lh ~/safe_deploy.sh
elif [ -f ~/phase3/guest-shim/safe_deploy.sh ]; then
    echo "✓ ~/phase3/guest-shim/safe_deploy.sh exists"
    ls -lh ~/phase3/guest-shim/safe_deploy.sh
else
    echo "✗ safe_deploy.sh not found"
fi
echo ""

# 12. Available Models
echo "=== 12. AVAILABLE MODELS ==="
timeout 10 ollama list 2>&1 | head -10 || echo "Could not list models"
echo ""

# 13. Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""

# Check key indicators
ISSUES=0
WARNINGS=0

if ! systemctl is-active --quiet ollama; then
    echo "✗ ISSUE: Ollama service is not running"
    ISSUES=$((ISSUES + 1))
fi

if [ ! -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "✗ ISSUE: CUDA shim library not found"
    ISSUES=$((ISSUES + 1))
fi

if [ -z "$OLLAMA_PID" ]; then
    echo "✗ ISSUE: No Ollama process found"
    ISSUES=$((ISSUES + 1))
else
    if ! sudo cat /proc/$OLLAMA_PID/maps 2>&1 | grep -q "libvgpu-cuda"; then
        echo "⚠ WARNING: Shim library not loaded in Ollama process"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

if [ ! -f /etc/ld.so.preload ] || ! grep -q "libvgpu-cuda.so" /etc/ld.so.preload 2>/dev/null; then
    echo "⚠ WARNING: libvgpu-cuda.so not in /etc/ld.so.preload"
    WARNINGS=$((WARNINGS + 1))
fi

if [ ! -d ~/phase3/guest-shim ] || [ ! -f ~/phase3/guest-shim/libvgpu_cuda.c ]; then
    echo "⚠ WARNING: Source files not found"
    WARNINGS=$((WARNINGS + 1))
fi

if [ -n "$OLLAMA_PID" ]; then
    SHIM_LOG="/tmp/vgpu-shim-cuda-${OLLAMA_PID}.log"
    if [ -f "$SHIM_LOG" ]; then
        if ! grep -q "Pre-initialization succeeded" "$SHIM_LOG"; then
            echo "⚠ WARNING: Pre-initialization not found in shim log"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "⚠ WARNING: No shim log file found"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

LIBRARY_MODE=$(sudo journalctl -u ollama -n 500 --no-pager 2>&1 | grep -E "library=" | tail -1)
if echo "$LIBRARY_MODE" | grep -qi "library=cpu"; then
    echo "⚠ WARNING: Ollama is using CPU mode (library=cpu)"
    WARNINGS=$((WARNINGS + 1))
elif echo "$LIBRARY_MODE" | grep -qi "library=cuda"; then
    echo "✓ SUCCESS: Ollama is using GPU mode (library=cuda)"
else
    echo "⚠ WARNING: Could not determine library mode from logs"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "Total Issues: $ISSUES"
echo "Total Warnings: $WARNINGS"
echo ""
echo "Review file saved to: $REVIEW_FILE"
