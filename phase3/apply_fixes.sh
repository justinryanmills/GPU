#!/bin/bash

VGPU_CONF="/etc/systemd/system/ollama.service.d/vgpu.conf"

echo "Applying fixes to vgpu.conf..."

sed -i 's|:/usr/lib64/libvgpu-syscall.so||g' "$VGPU_CONF"
sed -i 's|libvgpu-syscall.so:||g' "$VGPU_CONF"

sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"|g' "$VGPU_CONF"

if ! grep -q "OLLAMA_LIBRARY_PATH" "$VGPU_CONF"; then
    echo "" >> "$VGPU_CONF"
    echo "# OLLAMA_LIBRARY_PATH" >> "$VGPU_CONF"
    echo "Environment=\"OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12\"" >> "$VGPU_CONF"
fi

echo "Fixes applied"
echo ""
echo "Verifying..."
grep -E "LD_PRELOAD|OLLAMA_LIBRARY_PATH" "$VGPU_CONF" | head -2
