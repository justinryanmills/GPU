#!/bin/bash
set -e

echo "=== Deploying cuInit() early initialization fix ==="

# Ensure we're in the right directory
cd ~/phase3/guest-shim || { echo "Error: ~/phase3/guest-shim not found"; exit 1; }

# Rebuild the CUDA shim with the fix
echo "Rebuilding libvgpu-cuda.so..."
sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so \
    libvgpu_cuda.c cuda_transport.c \
    -I../include -I. -ldl -lpthread -O2 -Wall

# Verify the library was built
if [ ! -f /usr/lib64/libvgpu-cuda.so ]; then
    echo "ERROR: Failed to build libvgpu-cuda.so"
    exit 1
fi

echo "✓ Library built successfully"
ls -lh /usr/lib64/libvgpu-cuda.so

# Restart Ollama
echo "Restarting Ollama service..."
sudo systemctl restart ollama
sleep 8

# Check if Ollama is running
if ! systemctl is-active --quiet ollama; then
    echo "WARNING: Ollama service may not be running"
fi

# Check logs for pre-initialization
echo ""
echo "=== Checking for pre-initialization logs ==="
sudo journalctl -u ollama -n 200 --no-pager | grep -iE "Pre-initializing|Pre-initialization|cuInit.*OK|Constructor.*cuInit" | head -10 || echo "No pre-initialization messages found"

# Check library mode
echo ""
echo "=== Checking library mode ==="
sudo journalctl -u ollama -n 300 --no-pager | grep -E "library=" | tail -10 || echo "No library mode entries found"

echo ""
echo "=== Fix deployed. Testing with inference ==="
echo "Run: ollama run llama3.2:1b 'test'"
echo "Then check logs with: sudo journalctl -u ollama --since '1 minute ago' | grep library="
