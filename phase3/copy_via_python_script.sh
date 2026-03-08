#!/bin/bash

VM_USER="test-7"
VM_HOST="10.25.33.17"

echo "=== Method 2: Copy via Python script ==="
echo ""

cat > /tmp/receive_file.py << 'PYEOF'
#!/usr/bin/env python3
import base64
import sys

encoded = sys.stdin.read().strip()

decoded = base64.b64decode(encoded)
with open(sys.argv[1], 'wb') as f:
    f.write(decoded)

print(f"File written: {sys.argv[1]}")
print(f"Size: {len(decoded):,} bytes")
PYEOF

echo "1. Copying small Python receiver script..."
scp -o StrictHostKeyChecking=no /tmp/receive_file.py ${VM_USER}@${VM_HOST}:~/receive_file.py || {
    echo "   SCP failed, trying SSH heredoc method..."
    ssh -o StrictHostKeyChecking=no ${VM_USER}@${VM_HOST} 'cat > ~/receive_file.py << '\''PYEOF'\''
import base64
import sys
encoded = sys.stdin.read().strip()
decoded = base64.b64decode(encoded)
with open(sys.argv[1], '\''wb'\'') as f:
    f.write(decoded)
print(f"File written: {sys.argv[1]}")
print(f"Size: {len(decoded):,} bytes")
PYEOF
chmod +x ~/receive_file.py'
}

echo "2. Encoding and sending file..."
base64 /home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c | \
ssh -o StrictHostKeyChecking=no ${VM_USER}@${VM_HOST} 'python3 ~/receive_file.py ~/phase3/guest-shim/libvgpu_cuda.c'

echo ""
echo "3. Verifying..."
ssh -o StrictHostKeyChecking=no ${VM_USER}@${VM_HOST} 'ls -lh ~/phase3/guest-shim/libvgpu_cuda.c && wc -l ~/phase3/guest-shim/libvgpu_cuda.c'

echo ""
echo "Done!"
