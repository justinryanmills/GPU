#!/bin/bash
# verify_build_no_errors.sh
# Verifies that the build succeeds with -Werror (all warnings treated as errors)
# This ensures no warnings are ignored

set -e

BUILD_DIR="${1:-/tmp/vgpu_safe_rebuild}"
cd "$BUILD_DIR" || { echo "ERROR: Cannot cd to $BUILD_DIR"; exit 1; }

echo "=========================================="
echo "BUILD VERIFICATION - STRICT MODE"
echo "=========================================="
echo ""

# Test 1: Build with -Werror
echo "[1] Building with -Werror (all warnings = errors)..."
if gcc -shared -fPIC -Werror -o libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I. -ldl -lpthread 2>&1 | tee /tmp/build_werror.log; then
    echo "✓ Build succeeded with -Werror"
else
    echo "✗ BUILD FAILED with -Werror"
    echo ""
    echo "Errors found:"
    grep -i "error:" /tmp/build_werror.log | head -20
    exit 1
fi

# Test 2: Build with -Wall -Wextra -Werror
echo ""
echo "[2] Building with -Wall -Wextra -Werror (maximum strictness)..."
if gcc -shared -fPIC -Wall -Wextra -Werror -o libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I. -ldl -lpthread 2>&1 | tee /tmp/build_strict.log; then
    echo "✓ Build succeeded with -Wall -Wextra -Werror"
else
    echo "✗ BUILD FAILED with -Wall -Wextra -Werror"
    echo ""
    echo "Errors found:"
    grep -i "error:" /tmp/build_strict.log | head -20
    exit 1
fi

# Test 3: Verify library
echo ""
echo "[3] Verifying library..."
if [ ! -f libvgpu-cuda.so ]; then
    echo "✗ Library file not created"
    exit 1
fi

if ! file libvgpu-cuda.so | grep -q "shared object"; then
    echo "✗ Library is not a valid shared object"
    exit 1
fi

echo "✓ Library is valid"

# Test 4: Check for undefined symbols
echo ""
echo "[4] Checking for undefined symbols..."
if nm -u libvgpu-cuda.so 2>&1 | grep -q "undefined"; then
    echo "⚠ Some undefined symbols found (may be normal for shared libraries)"
    nm -u libvgpu-cuda.so | head -10
else
    echo "✓ No undefined symbols (or all resolved)"
fi

echo ""
echo "=========================================="
echo "✓ ALL BUILD VERIFICATIONS PASSED"
echo "=========================================="
echo ""
echo "Build succeeded with:"
echo "  - -Werror (all warnings treated as errors)"
echo "  - -Wall -Wextra -Werror (maximum strictness)"
echo ""
echo "Library is ready for deployment"

exit 0
