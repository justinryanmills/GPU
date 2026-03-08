#!/bin/bash
set -e
SUDO="${SUDO:-sudo}"
SNAP_REV="${SNAP_REV:-105}"
OLLAMA_LIB="/snap/ollama/${SNAP_REV}/lib/ollama"
OLLAMA_CUDA="${OLLAMA_LIB}/cuda_v12"
VGPU_LIB="/opt/vgpu/lib"
OVERLAY_DIR="/opt/ollama-cuda"

echo "=== GPU mode fix for snap Ollama ==="
echo "  OLLAMA_LIB=$OLLAMA_LIB"
echo "  VGPU_LIB=$VGPU_LIB"

$SUDO mkdir -p "$OVERLAY_DIR"
if [[ ! -e "$OVERLAY_DIR/libggml-cuda.so" ]]; then
  $SUDO ln -sf "${OLLAMA_CUDA}/libggml-cuda.so" "$OVERLAY_DIR/libggml-cuda.so"
  echo "  Created $OVERLAY_DIR/libggml-cuda.so -> ${OLLAMA_CUDA}/libggml-cuda.so"
fi

for conf in /etc/systemd/system/snap.ollama.listener.service.d/vgpu.conf \
           /etc/systemd/system/snap.ollama.ollama.service.d/vgpu.conf; do
  if [[ -d "$(dirname "$conf")" ]]; then
    $SUDO tee "$conf" > /dev/null << 'EOF'
[Service]
Environment="LD_PRELOAD=/opt/vgpu/lib/libcuda.so.1"
Environment="LD_LIBRARY_PATH=/opt/vgpu/lib:/snap/ollama/105/lib/ollama:/snap/ollama/105/lib/ollama/cuda_v12"
Environment="OLLAMA_LIBRARY_PATH=/opt/ollama-cuda:/snap/ollama/105/lib/ollama:/snap/ollama/105/lib/ollama/cuda_v12"
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"
EOF
    echo "  Updated $conf"
  fi
done

$SUDO systemctl daemon-reload
$SUDO systemctl restart snap.ollama.listener.service 2>/dev/null || true
$SUDO systemctl restart snap.ollama.ollama.service 2>/dev/null || true
echo "  Restarted Ollama services. Wait a few seconds then check: journalctl -u snap.ollama.ollama -n 50 --no-pager | grep -E 'inference compute|library=|total_vram'"
