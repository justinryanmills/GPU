#!/bin/bash
# Install ollama wrapper: LD_LIBRARY_PATH with /opt/vgpu/lib first, no LD_PRELOAD.
set -e
if [ ! -f /usr/local/bin/ollama.real ]; then
  cp -a /usr/local/bin/ollama /usr/local/bin/ollama.real
fi
cat > /tmp/ollama_wrapper.sh << 'WRAP'
#!/bin/bash
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama"
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_NUM_GPU="1"
exec /usr/local/bin/ollama.real "$@"
WRAP
chmod +x /tmp/ollama_wrapper.sh
cp /tmp/ollama_wrapper.sh /usr/local/bin/ollama
echo "Wrapper installed."
