#!/bin/bash
set -e

echo "=========================================="
echo "VM GPU Attributes Check Script"
echo "=========================================="

echo ""
echo "=== Finding shim source code ==="
SHIM_CODE=$(find ~ -name "libvgpu_cuda.c" -type f 2>/dev/null | head -1)
if [ -z "$SHIM_CODE" ]; then
    SHIM_CODE=$(find /home -name "libvgpu_cuda.c" -type f 2>/dev/null | head -1)
fi
if [ -z "$SHIM_CODE" ]; then
    SHIM_CODE=$(find /opt -name "libvgpu_cuda.c" -type f 2>/dev/null | head -1)
fi

if [ -n "$SHIM_CODE" ]; then
    echo "Found: $SHIM_CODE"
    SHIM_DIR=$(dirname "$SHIM_CODE")
else
    echo "ERROR: Could not find libvgpu_cuda.c"
    exit 1
fi

echo ""
echo "=== Finding gpu_properties.h ==="
PROPS_FILE=$(find "$SHIM_DIR" -name "gpu_properties.h" -type f 2>/dev/null | head -1)
if [ -n "$PROPS_FILE" ]; then
    echo "Found: $PROPS_FILE"
else
    echo "ERROR: Could not find gpu_properties.h"
    exit 1
fi

echo ""
echo "=== Checking MAX_THREADS_PER_BLOCK value ==="
MAX_THREADS=$(grep "GPU_DEFAULT_MAX_THREADS_PER_BLOCK" "$PROPS_FILE" | grep -v "^#" | head -1)
echo "Current definition: $MAX_THREADS"

if echo "$MAX_THREADS" | grep -q "1620000\|132.*12288"; then
    echo "❌ PROBLEM FOUND: MAX_THREADS_PER_BLOCK has invalid value!"
    echo "   Expected: 1024"
    echo "   Found: $MAX_THREADS"
    NEEDS_FIX=1
elif echo "$MAX_THREADS" | grep -q "1024"; then
    echo "MAX_THREADS_PER_BLOCK is correct (1024)"
    NEEDS_FIX=0
else
    echo "⚠️  Could not determine value from: $MAX_THREADS"
    NEEDS_FIX=0
fi

echo ""
echo "=== Checking cuDeviceGetAttribute implementation ==="
ATTR_CODE=$(grep -A 3 "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK" "$SHIM_CODE" | head -5)
echo "Implementation:"
echo "$ATTR_CODE"

echo ""
echo "=== Checking installed library ==="
if [ -f "/usr/lib64/libvgpu-cuda.so" ]; then
    echo "Library exists: /usr/lib64/libvgpu-cuda.so"
    ls -lh /usr/lib64/libvgpu-cuda.so
    echo ""
    echo "Checking if library contains invalid value (this may take a moment)..."
    if strings /usr/lib64/libvgpu-cuda.so | grep -q "1620000"; then
        echo "❌ PROBLEM: Library contains 1620000 (invalid value)"
        NEEDS_REBUILD=1
    else
        echo "Library does not contain invalid value"
        NEEDS_REBUILD=0
    fi
else
    echo "⚠️  Library not found at /usr/lib64/libvgpu-cuda.so"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
if [ "$NEEDS_FIX" = "1" ]; then
    echo "❌ Code needs to be fixed"
    echo "   File: $PROPS_FILE"
    echo "   Change MAX_THREADS_PER_BLOCK to 1024"
elif [ "$NEEDS_REBUILD" = "1" ]; then
    echo "⚠️  Code is correct but library needs rebuild"
else
    echo "Everything looks correct"
fi

echo ""
echo "Next steps:"
echo "1. If code needs fix, edit $PROPS_FILE"
echo "2. Rebuild: cd $SHIM_DIR && sudo ./install.sh"
echo "3. Restart Ollama: sudo systemctl restart ollama"
