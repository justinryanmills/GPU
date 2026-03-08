#!/bin/bash

VM="test-5@10.25.33.15"
PASSWORD="Calvin@123"
LOCAL_FILE="/home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c"
REMOTE_PATH="~/phase3/guest-shim/libvgpu_cuda.c"

echo "======================================================================"
echo "DEPLOYING FGETS() PATH LOOKUP FIX VIA SCP"
echo "======================================================================"

copy_file_with_retry() {
    local file=$1
    local dest=$2
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo ""
        echo "[Attempt $attempt/$max_attempts] Copying file via SCP..."
        
        if command -v sshpass &> /dev/null; then
            sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
                -o ConnectTimeout=30 \
                -o ServerAliveInterval=10 \
                -o ServerAliveCountMax=3 \
                "$file" "${VM}:${dest}" 2>&1
        else
            scp -o StrictHostKeyChecking=no \
                -o ConnectTimeout=30 \
                -o ServerAliveInterval=10 \
                -o ServerAliveCountMax=3 \
                "$file" "${VM}:${dest}" 2>&1
        fi
        
        local scp_exit=$?
        
        if [ $scp_exit -eq 0 ]; then
            echo "  File copied successfully"
            return 0
        else
            echo "  ⚠ Copy failed (exit code: $scp_exit)"
            if [ $attempt -lt $max_attempts ]; then
                echo "  Waiting 5 seconds before retry..."
                sleep 5
            fi
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "  ✗ Copy failed after $max_attempts attempts"
    return 1
}

run_on_vm() {
    local cmd=$1
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if command -v sshpass &> /dev/null; then
            sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
                -o ConnectTimeout=30 \
                -o ServerAliveInterval=10 \
                -o ServerAliveCountMax=3 \
                "$VM" "$cmd" 2>&1
        else
            ssh -o StrictHostKeyChecking=no \
                -o ConnectTimeout=30 \
                -o ServerAliveInterval=10 \
                -o ServerAliveCountMax=3 \
                "$VM" "$cmd" 2>&1
        fi
        
        local ssh_exit=$?
        
        if [ $ssh_exit -eq 0 ]; then
            return 0
        else
            if [ $attempt -lt $max_attempts ]; then
                echo "  Retrying command (attempt $attempt/$max_attempts)..."
                sleep 3
            fi
        fi
        
        attempt=$((attempt + 1))
    done
    
    return 1
}

echo ""
echo "[1] Verifying local file..."
if [ ! -f "$LOCAL_FILE" ]; then
    echo "  ✗ Local file not found: $LOCAL_FILE"
    exit 1
fi

file_size=$(stat -f%z "$LOCAL_FILE" 2>/dev/null || stat -c%s "$LOCAL_FILE" 2>/dev/null)
echo "  Local file found (size: $file_size bytes)"

echo ""
echo "[2] Copying file to VM..."
if copy_file_with_retry "$LOCAL_FILE" "$REMOTE_PATH"; then
    echo "  File transfer successful"
else
    echo "  ✗ File transfer failed"
    exit 1
fi

echo ""
echo "[3] Verifying file on VM..."
verify_cmd="test -f $REMOTE_PATH && ls -lh $REMOTE_PATH && echo 'FILE_VERIFIED'"
if run_on_vm "$verify_cmd" | grep -q "FILE_VERIFIED"; then
    echo "  File verified on VM"
    run_on_vm "$verify_cmd" | grep -v "FILE_VERIFIED"
else
    echo "  ✗ File verification failed"
    exit 1
fi

echo ""
echo "[4] Rebuilding shim library on VM..."
build_cmd="cd ~/phase3/guest-shim && echo '$PASSWORD' | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1"
build_output=$(run_on_vm "$build_cmd")

if echo "$build_output" | grep -q "error"; then
    echo "  ✗ Build failed with errors:"
    echo "$build_output" | grep -i "error" | head -10
    exit 1
fi

verify_build_cmd="test -f /usr/lib64/libvgpu-cuda.so && ls -lh /usr/lib64/libvgpu-cuda.so && echo 'BUILD_OK'"
if run_on_vm "$verify_build_cmd" | grep -q "BUILD_OK"; then
    echo "  Build successful"
    run_on_vm "$verify_build_cmd" | grep -v "BUILD_OK"
else
    echo "  ✗ Build verification failed"
    exit 1
fi

echo ""
echo "[5] Testing device discovery..."
test_cmd="ls 2>&1 | head -2 > /dev/null; sleep 2; echo '$PASSWORD' | sudo -S journalctl --since '1 minute ago' --no-pager 2>&1 | grep -E 'Found VGPU|fgets.*returning.*device=0x2331|fgets.*returning.*class=0x030200' | tail -5"
test_output=$(run_on_vm "$test_cmd")

echo "  Device discovery logs:"
if [ -n "$test_output" ]; then
    echo "$test_output" | grep -v "password" | while read line; do
        if [ -n "$line" ]; then
            echo "    $line"
        fi
    done
    
    if echo "$test_output" | grep -q "Found VGPU"; then
        echo "  Device found! Fix working!"
    elif echo "$test_output" | grep -q "device=0x2331\|class=0x030200"; then
        echo "  Fix working - correct values returned"
    fi
else
    echo "    (No logs yet - may need another trigger)"
fi

echo ""
echo "[6] Restarting Ollama service..."
restart_cmd="echo '$PASSWORD' | sudo -S systemctl restart ollama && sleep 25 && systemctl is-active ollama && echo 'OLLAMA_OK'"
if run_on_vm "$restart_cmd" | grep -q "OLLAMA_OK"; then
    echo "  Ollama restarted and running"
else
    echo "  ⚠ Ollama restart status unclear"
fi

echo ""
echo "[7] Running test inference and checking GPU mode..."
inference_cmd="timeout 50 ollama run llama3.2:1b 'test' 2>&1 | head -5; sleep 2; echo '$PASSWORD' | sudo -S journalctl -u ollama --since '2 minutes ago' --no-pager 2>&1 | grep 'library=' | tail -5"
inference_output=$(run_on_vm "$inference_cmd")

echo "  GPU mode check:"
if echo "$inference_output" | grep -q "library=cuda"; then
    echo "  GPU MODE ACTIVE! (library=cuda)"
    echo "$inference_output" | grep "library=" | grep -v "password"
else
    echo "$inference_output" | grep "library=" | grep -v "password" || echo "    (No library= entries yet)"
    echo "  ⚠ GPU mode not yet confirmed"
fi

echo ""
echo "======================================================================"
echo "DEPLOYMENT COMPLETE"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  File copied via SCP"
echo "  File verified on VM"
echo "  Shim library rebuilt"
echo "  Ollama restarted"
echo ""
echo "Check the logs above for device discovery and GPU mode status."
echo "======================================================================"
