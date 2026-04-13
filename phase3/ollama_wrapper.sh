#!/bin/bash
# Ollama wrapper script for systemd
# Ensures LD_PRELOAD is set and propagated to all subprocesses

# Set shim libraries
# Do not preload libvgpu-exec / libvgpu-syscall with the Ollama Go binary (exit 126
# under systemd). CUDA + NVML shims are enough for discovery and mediated CUDA.
export LD_PRELOAD="/usr/lib64/libvgpu-cudart.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
export NVIDIA_VISIBLE_DEVICES=all
export OLLAMA_LLM_LIBRARY=cuda_v12
export OLLAMA_NUM_GPU=999
export VGPU_SHMEM_MIN_SPAN_KB=32
export VGPU_ALLOW_MULTI_PROCESS_SHMEM=1
export VGPU_HTOD_BAR1=0
export VGPU_MODULE_BAR1=0
export VGPU_HTOD_BAR1_SHADOW=0
export OLLAMA_NO_MMAP=1

# Log wrapper execution
echo "[ollama-wrapper] Starting Ollama with shim injection (pid=$$, LD_PRELOAD=$LD_PRELOAD)" >&2

# Prefer a patched/rename binary if present; otherwise the installed ollama build.
for _ollama in /usr/local/bin/ollama.bin.new /usr/local/bin/ollama.bin /usr/local/bin/ollama; do
    if [ -x "$_ollama" ]; then
        exec "$_ollama" serve "$@"
    fi
done
echo "[ollama-wrapper] FATAL: no ollama binary found in /usr/local/bin" >&2
exit 127
