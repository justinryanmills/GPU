#!/bin/bash

VM="test-6@10.25.33.16"
PASS="Calvin@123"

echo "=== QUICK FIX DEPLOYMENT ==="
echo ""

echo "Step 1: Removing preload (if accessible)..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$VM" "echo $PASS | sudo -S sh -c 'echo > /etc/ld.so.preload'" 2>/dev/null && echo "Preload removed" || echo "⚠ Could not remove preload (VM may be down)"

echo ""
echo "Step 2: Copying fixed libvgpu_cuda.c..."
sshpass -p "$PASS" scp -o StrictHostKeyChecking=no /home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c "$VM:~/phase3/guest-shim/libvgpu_cuda.c" 2>/dev/null && echo "File copied" || echo "✗ Copy failed"

echo ""
echo "Step 3: Building shim..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$VM" "cd ~/phase3/guest-shim && echo $PASS | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 2>&1 | tail -2" && echo "Build complete" || echo "✗ Build failed"

echo ""
echo "Step 4: Testing lspci BEFORE preload..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$VM" "lspci 2>&1 | head -5" && echo "lspci works (baseline)" || echo "✗ lspci broken"

echo ""
echo "Step 5: Configuring preload..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$VM" "echo $PASS | sudo -S sh -c 'echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload'" && echo "Preload configured" || echo "✗ Preload failed"

echo ""
echo "Step 6: CRITICAL TEST - lspci after preload..."
RESULT=$(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$VM" "lspci 2>&1 | head -10" 2>&1)
if echo "$RESULT" | grep -q "Segmentation fault\|core dumped"; then
    echo "✗✗✗ lspci CRASHED - removing preload..."
    sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$VM" "echo $PASS | sudo -S sh -c 'echo > /etc/ld.so.preload'" 2>/dev/null
    exit 1
else
    echo "lspci WORKS!"
    echo "$RESULT" | head -5
fi

echo ""
echo "Step 7: Verifying SSH..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$VM" "echo 'SSH_OK'" && echo "SSH working" || echo "✗ SSH issue"

echo ""
echo "=== FIX DEPLOYED ==="
