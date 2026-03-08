#!/bin/bash
set -e

CONF_FILE="/etc/ld.so.conf.d/vgpu.conf"
LIB_DIR="/usr/lib64"

echo "Registering system-wide library paths..."
echo "Library directory: $LIB_DIR"
echo ""

if [ ! -d "$LIB_DIR" ]; then
    echo "ERROR: Library directory not found: $LIB_DIR"
    exit 1
fi

if [ ! -f "$LIB_DIR/libvgpu-cuda.so" ]; then
    echo "WARNING: CUDA shim library not found: $LIB_DIR/libvgpu-cuda.so"
    echo "The path will still be registered, but libraries may not be available."
fi

if [ ! -f "$LIB_DIR/libvgpu-nvml.so" ]; then
    echo "WARNING: NVML shim library not found: $LIB_DIR/libvgpu-nvml.so"
fi

echo "Creating $CONF_FILE..."
cat > "$CONF_FILE" <<EOF
$LIB_DIR
EOF

echo "Created: $CONF_FILE"

echo ""
echo "Running ldconfig to rebuild library cache..."
if ldconfig 2>&1 | grep -v "WARNING" || true; then
    echo "Ran ldconfig"
else
    echo "Ran ldconfig (warnings suppressed)"
fi

echo ""
echo "Verifying libraries in cache..."
if ldconfig -p 2>&1 | grep -q "libvgpu-cuda"; then
    echo "CUDA shim found in ldconfig cache:"
    ldconfig -p 2>&1 | grep "libvgpu-cuda" | head -2
else
    echo "⚠ CUDA shim not found in ldconfig cache (may still work via symlinks)"
fi

if ldconfig -p 2>&1 | grep -q "libvgpu-nvml"; then
    echo "NVML shim found in ldconfig cache:"
    ldconfig -p 2>&1 | grep "libvgpu-nvml" | head -2
else
    echo "⚠ NVML shim not found in ldconfig cache (may still work via symlinks)"
fi

echo ""
echo "System-wide path registration complete"
