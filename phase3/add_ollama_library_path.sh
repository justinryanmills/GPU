#!/bin/bash

VGPU_CONF="/etc/systemd/system/ollama.service.d/vgpu.conf"

cp "$VGPU_CONF" "${VGPU_CONF}.backup_ollama_library_path"

if ! grep -q "OLLAMA_LIBRARY_PATH" "$VGPU_CONF"; then
    echo "" >> "$VGPU_CONF"
    echo "# OLLAMA_LIBRARY_PATH - tells scanner where to find backend libraries" >> "$VGPU_CONF"
    echo "Environment=\"OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12\"" >> "$VGPU_CONF"
    echo "Added OLLAMA_LIBRARY_PATH to vgpu.conf"
else
    echo "OLLAMA_LIBRARY_PATH already exists in vgpu.conf"
fi

echo ""
echo "Verification:"
grep "OLLAMA_LIBRARY_PATH" "$VGPU_CONF"

echo ""
echo "Restarting Ollama..."
systemctl daemon-reload
systemctl restart ollama

echo ""
echo "Waiting 10 seconds for discovery..."
sleep 10

echo ""
echo "Checking discovery results:"
journalctl -u ollama --since "15 seconds ago" --no-pager 2>&1 | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH" | tail -10
