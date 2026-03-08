#!/bin/bash

set -e

echo "================================================================"
echo "FINAL FIX - ENABLING GPU MODE"
echo "================================================================"

echo ""
echo "[1] Fixing preload file..."
echo "/usr/lib64/libvgpu-cuda.so" | sudo tee /etc/ld.so.preload > /dev/null
echo "  Preload file configured:"
cat /etc/ld.so.preload

echo ""
echo "[2] Restarting Ollama..."
sudo systemctl restart ollama
sleep 15

if systemctl is-active --quiet ollama; then
    echo "  Ollama is running"
    
    echo ""
    echo "[3] Checking if shim is loaded..."
    OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
    if sudo cat /proc/$OLLAMA_PID/maps 2>&1 | grep -q "libvgpu-cuda"; then
        echo "  Shim library is loaded in Ollama process (PID: $OLLAMA_PID)"
    else
        echo "  ⚠ Shim not found in process maps"
    fi
    
    echo ""
    echo "[4] Running test inference..."
    timeout 50 ollama run llama3.2:1b "hello" 2>&1 | head -8
    
    echo ""
    echo "[5] Checking library mode..."
    echo "================================================================"
    echo "LIBRARY MODE ENTRIES:"
    echo "================================================================"
    sudo journalctl -u ollama --since "3 minutes ago" --no-pager 2>&1 | grep -E "library=" | tail -5
    
    echo ""
    echo "================================================================"
    echo "STATUS:"
    echo "================================================================"
    if sudo journalctl -u ollama --since "3 minutes ago" --no-pager 2>&1 | grep -qi "library=cuda"; then
        echo ""
        echo "SUCCESS! OLLAMA IS USING GPU MODE! "
        echo ""
        echo "Ollama is now operating in GPU mode instead of CPU mode!"
        echo "The vGPU is fully operational."
    else
        echo ""
        echo "⚠ Ollama is running but GPU mode not yet confirmed in logs."
        echo "   Try running another inference: ollama run llama3.2:1b 'test'"
    fi
else
    echo "  ✗ Ollama failed to start"
    sudo journalctl -u ollama -n 50 --no-pager | tail -20
fi

echo ""
echo "================================================================"
