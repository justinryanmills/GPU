#!/bin/bash
# configure_systemd.sh - Configure systemd service for safe library loading
#
# This script creates a systemd service override that sets LD_LIBRARY_PATH
# to ensure libraries are found. It does NOT use LD_PRELOAD (Go clears it)
# and does NOT use /etc/ld.so.preload (causes VM crashes).
#
# Usage: sudo ./configure_systemd.sh

set -e

SYSTEMD_OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="$SYSTEMD_OVERRIDE_DIR/vgpu.conf"
LIB_DIR="/usr/lib64"

echo "Configuring systemd service for safe library loading..."
echo ""

# Find Ollama binary location
OLLAMA_BIN=$(which ollama 2>/dev/null || echo "/usr/local/bin/ollama")
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama binary not found. Please install Ollama first."
    echo "Searched for: $OLLAMA_BIN"
    exit 1
fi

echo "Found Ollama binary: $OLLAMA_BIN"

# Create systemd override directory
mkdir -p "$SYSTEMD_OVERRIDE_DIR"

# Create override file with LD_LIBRARY_PATH only (NO LD_PRELOAD, NO /etc/ld.so.preload)
echo "Creating systemd override: $OVERRIDE_FILE"
cat > "$OVERRIDE_FILE" <<EOF
[Service]
# Safe library loading via filesystem-level mechanisms:
# 1. Symlinks in standard paths (libcuda.so.1 -> libvgpu-cuda.so)
# 2. System-wide library paths (/etc/ld.so.conf.d/vgpu.conf)
# 3. LD_LIBRARY_PATH as additional backup (inherited by subprocesses)
#
# CRITICAL: We do NOT use:
# - LD_PRELOAD (Go runtime clears it, doesn't work for runner subprocesses)
# - /etc/ld.so.preload (causes VM crashes, loads into ALL processes)
# - force_load_shim wrapper (not needed with symlinks)
#
# Libraries are discovered via symlinks and system-wide paths, which work
# regardless of how processes spawn (even with Go's direct syscalls).
Environment="LD_LIBRARY_PATH=$LIB_DIR:/usr/lib/x86_64-linux-gnu"
EOF

echo "✓ Created: $OVERRIDE_FILE"

# Reload systemd
echo ""
echo "Reloading systemd daemon..."
systemctl daemon-reload
echo "✓ Systemd daemon reloaded"

# Check if Ollama service exists
if systemctl list-unit-files | grep -q "^ollama.service"; then
    echo "✓ Ollama service found"
    
    # Show current status
    if systemctl is-active --quiet ollama; then
        echo "✓ Ollama service is currently running"
        echo ""
        echo "To apply changes, restart Ollama:"
        echo "  sudo systemctl restart ollama"
    else
        echo "⚠ Ollama service is not running"
        echo ""
        echo "To start Ollama:"
        echo "  sudo systemctl start ollama"
    fi
else
    echo "⚠ Ollama service not found in systemd"
    echo "The override will be applied when Ollama service is created."
fi

echo ""
echo "✓ Systemd configuration complete"
echo ""
echo "Configuration summary:"
echo "  Override file: $OVERRIDE_FILE"
echo "  LD_LIBRARY_PATH: $LIB_DIR:/usr/lib/x86_64-linux-gnu"
echo "  LD_PRELOAD: NOT SET (Go clears it)"
echo "  /etc/ld.so.preload: NOT USED (causes crashes)"
