#!/bin/bash
#
# Symbol verification script for libvgpu-cudart.so
#
# This script verifies which of the 39 "undefined" symbols actually exist
# in the library and checks if version symbols are exported correctly.
#
# Usage: ./verify_symbols.sh
#

set -e

LIB_PATH="/usr/lib64/libvgpu-cudart.so"
GGML_LIB="/usr/local/lib/ollama/cuda_v12/libggml-cuda.so"

echo "=========================================="
echo "Symbol Verification Script"
echo "=========================================="
echo ""

# Check if library exists
if [ ! -f "$LIB_PATH" ]; then
    echo "ERROR: Library not found: $LIB_PATH"
    echo "Please build and install the library first."
    exit 1
fi

echo "[1/5] Checking library file..."
ls -lh "$LIB_PATH"
echo ""

# List of undefined symbols from ldd -r output
UNDEFINED_SYMBOLS=(
    "cudaEventCreateWithFlags"
    "cudaEventDestroy"
    "cudaEventRecord"
    "cudaEventSynchronize"
    "cudaFuncGetAttributes"
    "cudaFuncSetAttribute"
    "cudaGraphDestroy"
    "cudaGraphExecDestroy"
    "cudaGraphExecUpdate"
    "cudaGraphInstantiate"
    "cudaGraphLaunch"
    "cudaHostRegister"
    "cudaHostUnregister"
    "cudaLaunchKernel"
    "cudaMallocManaged"
    "cudaMemcpy2DAsync"
    "cudaMemcpy3DPeerAsync"
    "cudaMemcpyAsync"
    "cudaMemcpyPeerAsync"
    "cudaMemGetInfo"
    "cudaMemset"
    "cudaMemsetAsync"
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
    "cudaPeekAtLastError"
    "__cudaPopCallConfiguration"
    "__cudaPushCallConfiguration"
    "__cudaRegisterFatBinaryEnd"
    "__cudaRegisterFatBinary"
    "__cudaRegisterFunction"
    "__cudaRegisterVar"
    "cudaSetDeviceFlags"
    "cudaSetDevice"
    "cudaStreamBeginCapture"
    "cudaStreamCreateWithFlags"
    "cudaStreamDestroy"
    "cudaStreamEndCapture"
    "cudaStreamIsCapturing"
    "cudaStreamSynchronize"
    "cudaStreamWaitEvent"
)

echo "[2/5] Checking for symbols in library..."
echo "Using nm -D to list dynamic symbols..."
echo ""

FOUND_SYMBOLS=()
NOT_FOUND_SYMBOLS=()

for symbol in "${UNDEFINED_SYMBOLS[@]}"; do
    # Check if symbol exists (with or without version)
    if nm -D "$LIB_PATH" 2>/dev/null | grep -q " $symbol"; then
        FOUND_SYMBOLS+=("$symbol")
        echo "  ✓ Found: $symbol"
    else
        NOT_FOUND_SYMBOLS+=("$symbol")
        echo "  ✗ Missing: $symbol"
    fi
done

echo ""
echo "[3/5] Checking version symbols..."
echo "Using readelf -V to check version information..."
echo ""

# Check if version script was applied
if readelf -V "$LIB_PATH" 2>/dev/null | grep -q "libcudart.so.12"; then
    echo "  ✓ Version script applied (libcudart.so.12 found)"
    VERSION_APPLIED=1
else
    echo "  ✗ Version script NOT applied"
    VERSION_APPLIED=0
fi

# Check versioned symbols
VERSIONED_COUNT=0
for symbol in "${FOUND_SYMBOLS[@]}"; do
    if readelf -V "$LIB_PATH" 2>/dev/null | grep -q "$symbol.*libcudart.so.12"; then
        VERSIONED_COUNT=$((VERSIONED_COUNT + 1))
    fi
done

echo "  Versioned symbols: $VERSIONED_COUNT / ${#FOUND_SYMBOLS[@]}"
echo ""

echo "[4/5] Checking symbol resolution with ldd -r..."
echo ""

# Run ldd -r on libggml-cuda.so to see what it reports
if [ -f "$GGML_LIB" ]; then
    echo "Checking undefined symbols in $GGML_LIB:"
    ldd -r "$GGML_LIB" 2>&1 | grep -E "undefined.*cuda" | head -20 || echo "  (no undefined CUDA symbols found)"
    echo ""
else
    echo "  WARNING: $GGML_LIB not found, skipping ldd -r check"
    echo ""
fi

echo "[5/5] Generating report..."
echo ""

REPORT_FILE="/tmp/symbol_verification_report.txt"
cat > "$REPORT_FILE" <<EOF
Symbol Verification Report
=========================
Generated: $(date)
Library: $LIB_PATH

SUMMARY
-------
Total undefined symbols checked: ${#UNDEFINED_SYMBOLS[@]}
Symbols found in library: ${#FOUND_SYMBOLS[@]}
Symbols NOT found: ${#NOT_FOUND_SYMBOLS[@]}
Version script applied: $([ $VERSION_APPLIED -eq 1 ] && echo "YES" || echo "NO")
Versioned symbols: $VERSIONED_COUNT / ${#FOUND_SYMBOLS[@]}

FOUND SYMBOLS
-------------
EOF

for symbol in "${FOUND_SYMBOLS[@]}"; do
    echo "  ✓ $symbol" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" <<EOF

MISSING SYMBOLS
---------------
EOF

for symbol in "${NOT_FOUND_SYMBOLS[@]}"; do
    echo "  ✗ $symbol" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" <<EOF

ANALYSIS
--------
EOF

if [ ${#NOT_FOUND_SYMBOLS[@]} -eq 0 ]; then
    echo "All symbols are present in the library." >> "$REPORT_FILE"
    if [ $VERSION_APPLIED -eq 0 ]; then
        echo "WARNING: Version script may not be applied correctly." >> "$REPORT_FILE"
        echo "This could cause 'no version information available' warnings." >> "$REPORT_FILE"
    fi
    if [ $VERSIONED_COUNT -lt ${#FOUND_SYMBOLS[@]} ]; then
        echo "WARNING: Not all symbols have version information." >> "$REPORT_FILE"
        echo "This could cause symbol resolution issues." >> "$REPORT_FILE"
    fi
else
    echo "Missing symbols need to be implemented in libvgpu_cudart.c" >> "$REPORT_FILE"
fi

cat "$REPORT_FILE"
echo ""
echo "Full report saved to: $REPORT_FILE"
