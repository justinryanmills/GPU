#!/bin/bash
export LD_PRELOAD="/opt/vgpu/lib/libnvidia-ml.so.1:/opt/vgpu/lib/libcuda.so.1:/opt/vgpu/lib/libcudart.so.12"
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama"
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_NUM_GPU="999"
exec /usr/local/bin/ollama.real.bin "$@"
