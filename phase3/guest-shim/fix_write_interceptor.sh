#!/bin/bash
# Fix write interceptor and rebuild on VM

set -e

cd ~/phase3/guest-shim

echo "=" | tee -a /tmp/fix_write.log
echo "FIXING WRITE INTERCEPTOR" | tee -a /tmp/fix_write.log
echo "=" | tee -a /tmp/fix_write.log

echo "[1/4] Rebuilding libvgpu-cuda.so..." | tee -a /tmp/fix_write.log
gcc -shared -fPIC -o libvgpu-cuda.so libvgpu_cuda.c \
    -I. -ldl -lpthread -D_GNU_SOURCE 2>&1 | tee -a /tmp/fix_write.log

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!" | tee -a /tmp/fix_write.log
    exit 1
fi

echo "[2/4] Installing..." | tee -a /tmp/fix_write.log
sudo cp libvgpu-cuda.so /usr/lib64/ 2>&1 | tee -a /tmp/fix_write.log
sudo ldconfig 2>&1 | tee -a /tmp/fix_write.log

echo "[3/4] Clearing old log files..." | tee -a /tmp/fix_write.log
sudo rm -f /tmp/ollama_errors*.log 2>&1 | tee -a /tmp/fix_write.log

echo "[4/4] Restarting Ollama..." | tee -a /tmp/fix_write.log
sudo systemctl restart ollama 2>&1 | tee -a /tmp/fix_write.log

echo "Waiting for discovery..." | tee -a /tmp/fix_write.log
sleep 12
ollama list >/dev/null 2>&1 &
sleep 5

echo "" | tee -a /tmp/fix_write.log
echo "Checking for error log files..." | tee -a /tmp/fix_write.log
ls -la /tmp/ollama_errors*.log 2>&1 | tee -a /tmp/fix_write.log

if [ -f /tmp/ollama_errors_filtered.log ]; then
    echo "" | tee -a /tmp/fix_write.log
    echo "Filtered log (last 30 lines):" | tee -a /tmp/fix_write.log
    tail -30 /tmp/ollama_errors_filtered.log | tee -a /tmp/fix_write.log
    
    echo "" | tee -a /tmp/fix_write.log
    echo "Looking for ggml_backend_cuda_init error..." | tee -a /tmp/fix_write.log
    grep -i "ggml.*failed\|failed.*cuda\|failed.*init" /tmp/ollama_errors_filtered.log | tail -5 | tee -a /tmp/fix_write.log
fi

echo "" | tee -a /tmp/fix_write.log
echo "DONE" | tee -a /tmp/fix_write.log
