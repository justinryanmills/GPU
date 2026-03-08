#!/bin/bash

set -e

VM="${1:-test-4@10.25.33.12}"
PASSWORD="Calvin@123"

echo "=== Deploying va_start fix to $VM ==="
echo ""

echo "Step 1: Copying files..."
scp -o StrictHostKeyChecking=no phase3/guest-shim/libvgpu_cuda.c $VM:~/phase3/guest-shim/ <<< "$PASSWORD" || {
    echo "SCP failed, trying alternative method..."
    base64 phase3/guest-shim/libvgpu_cuda.c | ssh -o StrictHostKeyChecking=no $VM "base64 -d > ~/phase3/guest-shim/libvgpu_cuda.c" <<< "$PASSWORD"
}

echo ""
echo "Step 2: Building library..."
ssh -o StrictHostKeyChecking=no $VM <<EOF
set -e
cd ~/phase3/guest-shim
gcc -shared -fPIC -o /tmp/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2
ls -lh /tmp/libvgpu-cuda.so
echo "Build successful"
EOF <<< "$PASSWORD"

echo ""
echo "Step 3: Testing lspci BEFORE preload (baseline)..."
ssh -o StrictHostKeyChecking=no $VM "lspci | head -5" <<< "$PASSWORD" || {
    echo "✗ lspci failed before preload - VM may have issues"
    exit 1
}
echo "lspci works (baseline)"

echo ""
echo "Step 4: Installing library..."
ssh -o StrictHostKeyChecking=no $VM <<EOF
set -e
sudo mkdir -p /usr/lib64
sudo cp /tmp/libvgpu-cuda.so /usr/lib64/
echo "Library installed"
EOF <<< "$PASSWORD"

echo ""
echo "Step 5: Configuring LD_PRELOAD..."
ssh -o StrictHostKeyChecking=no $VM "sudo sh -c 'echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload'" <<< "$PASSWORD"
echo "Preload configured"

echo ""
echo "Step 6: CRITICAL TEST - lspci AFTER preload..."
ssh -o StrictHostKeyChecking=no $VM "lspci | head -5" <<< "$PASSWORD" || {
    echo "✗✗✗ FAILURE: lspci crashed after preload! ✗✗✗"
    exit 1
}
echo "SUCCESS: lspci works after preload! "

echo ""
echo "Step 7: Testing SSH connectivity..."
ssh -o StrictHostKeyChecking=no $VM "echo 'SSH_TEST_SUCCESS'" <<< "$PASSWORD" || {
    echo "✗ SSH connectivity test failed"
    exit 1
}
echo "SSH still functional"

echo ""
echo "Step 8: Testing other system commands..."
ssh -o StrictHostKeyChecking=no $VM <<EOF
whoami
pwd
ls /usr/lib64/libvgpu-cuda.so
echo "System commands work"
EOF <<< "$PASSWORD"

echo ""
echo "=== DEPLOYMENT SUCCESSFUL ==="
echo "Library built and installed"
echo "Preload configured"
echo "lspci works (before and after preload)"
echo "SSH functional"
echo "System commands work"
echo ""
echo "Ready for Ollama installation!"
