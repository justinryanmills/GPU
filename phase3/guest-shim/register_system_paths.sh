#!/bin/bash
# register_system_paths.sh - Register shim libraries in system-wide library paths
#
# This script registers /usr/lib64 in the system-wide library configuration
# so that all processes can discover the shim libraries via ldconfig cache.
# This works at the dynamic linker level, independent of process spawning method.
#
# Usage: sudo ./register_system_paths.sh

set -e

CONF_FILE="/etc/ld.so.conf.d/vgpu.conf"
LIB_DIR="/usr/lib64"

echo "Registering system-wide library paths..."
echo "Library directory: $LIB_DIR"
echo ""

# Check if library directory exists
if [ ! -d "$LIB_DIR" ]; then
    echo "ERROR: Library directory not found: $LIB_DIR"
    exit 1
fi

# Check if shim libraries exist
if [ ! -f "$LIB_DIR/libvgpu-cuda.so" ]; then
    echo "WARNING: CUDA shim library not found: $LIB_DIR/libvgpu-cuda.so"
    echo "The path will still be registered, but libraries may not be available."
fi

if [ ! -f "$LIB_DIR/libvgpu-nvml.so" ]; then
    echo "WARNING: NVML shim library not found: $LIB_DIR/libvgpu-nvml.so"
fi

# Create configuration file
echo "Creating $CONF_FILE..."
cat > "$CONF_FILE" <<EOF
# VGPU shim library path - registered system-wide
# This ensures libraries are discoverable by all processes, including
# subprocesses spawned by Go's runtime (which uses direct syscalls)
$LIB_DIR
EOF

echo "✓ Created: $CONF_FILE"

# Run ldconfig to rebuild cache
echo ""
echo "Running ldconfig to rebuild library cache..."
if ldconfig 2>&1 | grep -v "WARNING" || true; then
    echo "✓ Ran ldconfig"
else
    echo "✓ Ran ldconfig (warnings suppressed)"
fi

# Verify libraries are in cache
echo ""
echo "Verifying libraries in cache..."
if ldconfig -p 2>&1 | grep -q "libvgpu-cuda"; then
    echo "✓ CUDA shim found in ldconfig cache:"
    ldconfig -p 2>&1 | grep "libvgpu-cuda" | head -2
else
    echo "⚠ CUDA shim not found in ldconfig cache (may still work via symlinks)"
fi

if ldconfig -p 2>&1 | grep -q "libvgpu-nvml"; then
    echo "✓ NVML shim found in ldconfig cache:"
    ldconfig -p 2>&1 | grep "libvgpu-nvml" | head -2
else
    echo "⚠ NVML shim not found in ldconfig cache (may still work via symlinks)"
fi

echo ""
echo "✓ System-wide path registration complete"
