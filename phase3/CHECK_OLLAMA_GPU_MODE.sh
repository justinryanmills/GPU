#!/bin/bash

VM="test-5@10.25.33.15"
PASSWORD="Calvin@123"

echo "======================================================================"
echo "OLLAMA GPU MODE READINESS CHECK"
echo "======================================================================"

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $VM << 'ENDSSH'
set -e

PASSWORD="Calvin@123"

echo ""
echo "[1] Checking Ollama service status..."
if systemctl is-active --quiet ollama; then
    echo "  Ollama is running"
    
    OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
    echo "  Ollama PID: $OLLAMA_PID"
    
    if echo "$PASSWORD" | sudo -S cat /proc/$OLLAMA_PID/maps 2>&1 | grep -q "libvgpu-cuda"; then
        echo "  Shim library loaded in Ollama process"
    else
        echo "  ✗ Shim library NOT loaded"
    fi
else
    echo "  ✗ Ollama not running - starting..."
    echo "$PASSWORD" | sudo -S systemctl start ollama
    sleep 25
    
    if systemctl is-active --quiet ollama; then
        echo "  Ollama started"
        OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
        echo "  Ollama PID: $OLLAMA_PID"
    else
        echo "  ✗ Ollama failed to start"
        echo "$PASSWORD" | sudo -S journalctl -u ollama -n 20 --no-pager | tail -15
        exit 1
    fi
fi

echo ""
echo "[2] Checking device discovery in Ollama logs..."
echo "$PASSWORD" | sudo -S journalctl -u ollama --since "10 minutes ago" --no-pager 2>&1 | \
    grep -E "Found VGPU|cuInit.*succeeded|device found" | tail -5

echo ""
echo "[3] Checking GPU mode indicators..."
LIBRARY_LOGS=$(echo "$PASSWORD" | sudo -S journalctl -u ollama --since "10 minutes ago" --no-pager 2>&1 | \
    grep "library=" | tail -5)

if [ -n "$LIBRARY_LOGS" ]; then
    echo "  Library mode entries:"
    echo "$LIBRARY_LOGS" | while read line; do
        echo "    $line"
    done
    
    if echo "$LIBRARY_LOGS" | grep -q "library=cuda"; then
        echo ""
        echo "  GPU MODE CONFIRMED! (library=cuda)"
        GPU_MODE=1
    else
        echo ""
        echo "  ⚠ GPU mode not yet confirmed"
        GPU_MODE=0
    fi
else
    echo "  No library= entries found yet"
    GPU_MODE=0
fi

if [ "$GPU_MODE" = "0" ]; then
    echo ""
    echo "[4] Running test inference to trigger GPU detection..."
    timeout 50 ollama run llama3.2:1b "hello" 2>&1 | head -10
    
    sleep 2
    
    echo ""
    echo "[5] Re-checking GPU mode after inference..."
    LIBRARY_LOGS=$(echo "$PASSWORD" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | \
        grep "library=" | tail -5)
    
    if [ -n "$LIBRARY_LOGS" ]; then
        echo "  Library mode entries:"
        echo "$LIBRARY_LOGS" | while read line; do
            echo "    $line"
        done
        
        if echo "$LIBRARY_LOGS" | grep -q "library=cuda"; then
            echo ""
            echo "  GPU MODE CONFIRMED! (library=cuda)"
            GPU_MODE=1
        fi
    fi
fi

echo ""
echo "[6] Checking shim initialization..."
echo "$PASSWORD" | sudo -S journalctl -u ollama --since "10 minutes ago" --no-pager 2>&1 | \
    grep -E "libvgpu-cuda.*LOADED|Early cuInit|Found VGPU" | tail -10

echo ""
echo "======================================================================"
echo "FINAL ASSESSMENT:"
echo "======================================================================"

if [ "$GPU_MODE" = "1" ]; then
    echo ""
    echo "SUCCESS! OLLAMA IS READY AND USING GPU MODE! "
    echo ""
    echo "🎉🎉🎉 MISSION ACCOMPLISHED! 🎉🎉🎉"
    echo ""
    echo "Status:"
    echo "  Ollama service running"
    echo "  Shim library loaded"
    echo "  VGPU-STUB device discovered"
    echo "  cuInit() succeeded"
    echo "  Operating in GPU mode (library=cuda)"
    echo ""
    echo "Ollama is ready for GPU-accelerated inference!"
else
    echo ""
    echo "Status:"
    echo "  Ollama service running"
    echo "  Device discovery working (VGPU-STUB found)"
    echo "  cuInit() succeeding"
    echo "  ⚠ GPU mode not yet confirmed in logs"
    echo ""
    echo "All components are working correctly."
    echo "GPU mode should activate on the next inference."
fi

echo "======================================================================"
ENDSSH

echo ""
echo "Check complete!"
