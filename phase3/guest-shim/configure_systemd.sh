#!/bin/bash
set -e

SYSTEMD_OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="$SYSTEMD_OVERRIDE_DIR/vgpu.conf"
LIB_DIR="/usr/lib64"

echo "Configuring systemd service for safe library loading..."
echo ""

OLLAMA_BIN=$(which ollama 2>/dev/null || echo "/usr/local/bin/ollama")
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama binary not found. Please install Ollama first."
    echo "Searched for: $OLLAMA_BIN"
    exit 1
fi

echo "Found Ollama binary: $OLLAMA_BIN"

mkdir -p "$SYSTEMD_OVERRIDE_DIR"

echo "Creating systemd override: $OVERRIDE_FILE"
cat > "$OVERRIDE_FILE" <<EOF
[Service]
Environment="LD_LIBRARY_PATH=$LIB_DIR:/usr/lib/x86_64-linux-gnu"
EOF

echo "Created: $OVERRIDE_FILE"

echo ""
echo "Reloading systemd daemon..."
systemctl daemon-reload
echo "Systemd daemon reloaded"

if systemctl list-unit-files | grep -q "^ollama.service"; then
    echo "Ollama service found"
    
    if systemctl is-active --quiet ollama; then
        echo "Ollama service is currently running"
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
echo "Systemd configuration complete"
echo ""
echo "Configuration summary:"
echo "  Override file: $OVERRIDE_FILE"
echo "  LD_LIBRARY_PATH: $LIB_DIR:/usr/lib/x86_64-linux-gnu"
echo "  LD_PRELOAD: NOT SET (Go clears it)"
echo "  /etc/ld.so.preload: NOT USED (causes crashes)"
