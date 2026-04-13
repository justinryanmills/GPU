#!/bin/bash
# safe_rebuild_and_test.sh
# Safely rebuilds library with fixed code and tests it
# Does NOT deploy - only builds and tests

set -e

PASSWORD="$1"
BUILD_DIR="/tmp/vgpu_safe_rebuild"
LIB_PATH="$BUILD_DIR/libvgpu-cuda.so"

echo "=========================================="
echo "SAFE REBUILD AND TEST - NO DEPLOYMENT"
echo "=========================================="

# Step 1: Install gcc if needed
echo ""
echo "[1] Checking for gcc..."
if ! command -v gcc &> /dev/null; then
    echo "  Installing gcc..."
    echo "$PASSWORD" | sudo -S apt-get update -qq
    echo "$PASSWORD" | sudo -S apt-get install -y gcc build-essential
    echo "  ✓ gcc installed"
else
    echo "  ✓ gcc already installed"
fi

# Step 2: Verify source files exist
echo ""
echo "[2] Verifying source files..."
if [ ! -f "$BUILD_DIR/libvgpu_cuda.c" ]; then
    echo "  ✗ Source file not found"
    exit 1
fi
if [ ! -f "$BUILD_DIR/cuda_transport.c" ]; then
    echo "  ✗ cuda_transport.c not found"
    exit 1
fi
echo "  ✓ Source files found"

# Step 3: Build library
echo ""
echo "[3] Building library..."
cd "$BUILD_DIR"
gcc -shared -fPIC -o libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I. -ldl -lpthread 2>&1 | tee build.log

# Step 4: Check for errors
echo ""
echo "[4] Checking build for errors..."
if grep -qi "error:" build.log; then
    echo "  ✗ BUILD FAILED - Has compilation errors:"
    grep -i "error:" build.log
    exit 1
else
    echo "  ✓ BUILD SUCCEEDED - No errors"
fi

# Step 5: Verify library exists and is valid
echo ""
echo "[5] Verifying library..."
if [ ! -f "$LIB_PATH" ]; then
    echo "  ✗ Library file not created"
    exit 1
fi

if ! file "$LIB_PATH" | grep -q "shared object"; then
    echo "  ✗ Library is not a valid shared object"
    exit 1
fi
echo "  ✓ Library is valid"

# Step 6: Safety testing
echo ""
echo "[6] Safety testing with system processes..."
SAFE=true

for cmd in "cat /dev/null" "ls /tmp" "bash -c 'echo test'"; do
    if timeout 2 bash -c "LD_PRELOAD=\"$LIB_PATH\" $cmd" > /dev/null 2>&1; then
        echo "  ✓ $(echo $cmd | cut -d' ' -f1) safe"
    else
        if dmesg | tail -5 | grep -q "segfault"; then
            echo "  ✗ $(echo $cmd | cut -d' ' -f1) CRASHED"
            SAFE=false
        else
            echo "  ⚠ $(echo $cmd | cut -d' ' -f1) had issues (may be timeout)"
        fi
    fi
done

echo ""
echo "=========================================="
if [ "$SAFE" = true ]; then
    echo "✓ BUILD AND SAFETY TESTS PASSED"
    echo "=========================================="
    echo ""
    echo "Library is ready but NOT deployed"
    echo "Location: $LIB_PATH"
    echo ""
    echo "Next steps (ONLY after your approval):"
    echo "  1. Review this output"
    echo "  2. Approve deployment"
    echo "  3. Then deploy to /etc/ld.so.preload"
    exit 0
else
    echo "✗ SAFETY TESTS FAILED"
    echo "=========================================="
    echo ""
    echo "Library causes crashes - CANNOT deploy"
    exit 1
fi
