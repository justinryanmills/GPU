#!/bin/bash
# Quick deployment and status check

cd ~/phase3/guest-shim 2>/dev/null || { echo "ERROR: ~/phase3/guest-shim not found"; exit 1; }

echo "=== Rebuilding shim ==="
sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1
echo "Build exit code: $?"

echo ""
echo "=== Restarting Ollama ==="
sudo systemctl restart ollama
sleep 8

echo ""
echo "=== Library mode check ==="
sudo journalctl -u ollama -n 200 --no-pager | grep -E "library=" | tail -5

echo ""
echo "=== Running test inference ==="
timeout 20 ollama run llama3.2:1b "test" 2>&1 | head -10

echo ""
echo "=== Final library mode ==="
sudo journalctl -u ollama --since "1 minute ago" --no-pager | grep -E "library=" | tail -3
