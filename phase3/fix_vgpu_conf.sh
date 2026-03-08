#!/bin/bash

set -e

VGPU_CONF="/etc/systemd/system/ollama.service.d/vgpu.conf"

sed -i "s|libvgpu-exec.so:||g" "$VGPU_CONF"
sed -i "s|libvgpu-syscall.so:||g" "$VGPU_CONF"

if ! grep -q "OLLAMA_LIBRARY_PATH" "$VGPU_CONF"; then
    echo 'Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"' >> "$VGPU_CONF"
fi

systemctl daemon-reload

systemctl restart ollama

echo "Configuration fixed and Ollama restarted"
