#!/bin/bash
export LD_PRELOAD="/usr/lib64/libvgpu-cudart.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
export NVIDIA_VISIBLE_DEVICES=all
export OLLAMA_LLM_LIBRARY=cuda_v12
export OLLAMA_NUM_GPU=999
exec /usr/local/bin/ollama.bin.new serve "$@"
