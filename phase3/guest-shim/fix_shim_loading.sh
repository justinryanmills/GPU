#!/bin/bash
# Fix Shim Loading - Reload systemd and restart Ollama

set -e

echo "=== Fixing Ollama Shim Loading ==="
echo ""

# Check current state
echo "[1] Checking current Ollama process..."
MAIN_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$MAIN_PID" ]; then
    echo "  Current PID: $MAIN_PID"
    echo "  LD_PRELOAD in environment:"
    cat /proc/$MAIN_PID/environ 2>/dev/null | tr '\0' '\n' | grep LD_PRELOAD || echo "    ✗ NOT FOUND"
    echo "  Shims loaded:"
    cat /proc/$MAIN_PID/maps 2>/dev/null | grep libvgpu | head -3 || echo "    ✗ NOT FOUND"
else
    echo "  No Ollama process found"
fi

echo ""
echo "[2] Reloading systemd daemon..."
sudo systemctl daemon-reload

echo ""
echo "[3] Restarting Ollama service..."
sudo systemctl restart ollama

echo ""
echo "[4] Waiting for Ollama to start..."
sleep 3

echo ""
echo "[5] Verifying shims are now loaded..."
NEW_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$NEW_PID" ]; then
    echo "  New PID: $NEW_PID"
    echo "  LD_PRELOAD in environment:"
    cat /proc/$NEW_PID/environ 2>/dev/null | tr '\0' '\n' | grep LD_PRELOAD || echo "    ✗ NOT FOUND"
    echo "  Shims loaded:"
    cat /proc/$NEW_PID/maps 2>/dev/null | grep libvgpu | head -5 || echo "    ✗ NOT FOUND"
    
    if cat /proc/$NEW_PID/maps 2>/dev/null | grep -q libvgpu; then
        echo ""
        echo "  ✓ SUCCESS: Shims are loaded!"
    else
        echo ""
        echo "  ✗ FAILED: Shims are still not loaded"
        echo "  Check systemd configuration:"
        echo "    systemctl show ollama | grep Environment"
    fi
else
    echo "  ✗ No Ollama process found after restart"
fi

echo ""
echo "[6] Checking discovery result..."
sleep 2
journalctl -u ollama --since "10 seconds ago" --no-pager | grep -E "inference compute|discovering" | tail -3 || echo "  No discovery logs yet"

echo ""
echo "=== Fix Complete ==="
