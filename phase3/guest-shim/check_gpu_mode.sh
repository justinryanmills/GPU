#!/bin/bash
# Simple script to check if Ollama is using GPU mode

echo "STATUS_CHECK_START"

# Rebuild if source exists
if [ -f ~/phase3/guest-shim/libvgpu_cuda.c ]; then
    echo "REBUILDING_SHIM"
    cd ~/phase3/guest-shim
    sudo gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1 | head -5
    echo "REBUILD_DONE"
fi

# Restart Ollama
echo "RESTARTING_OLLAMA"
sudo systemctl restart ollama 2>&1
sleep 8

# Check library mode
echo "CHECKING_LIBRARY_MODE"
sudo journalctl -u ollama -n 300 --no-pager 2>&1 | grep -E "library=" | tail -10

# Run test
echo "RUNNING_TEST"
timeout 15 ollama run llama3.2:1b "test" 2>&1 | head -5

# Final check
echo "FINAL_CHECK"
sudo journalctl -u ollama --since "1 minute ago" --no-pager 2>&1 | grep -E "library=" | tail -3

echo "STATUS_CHECK_END"
