#!/usr/bin/env python3
"""
Copy file to VM via base64 encoding (no SCP required)
This method works even with unstable SSH connections
"""

import base64
import sys
import os

def split_into_chunks(data, chunk_size=50000):
    """Split data into chunks"""
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

def generate_copy_script(source_file, remote_path, vm_user, vm_host):
    """Generate a script to copy file via base64"""
    
    print(f"Reading {source_file}...")
    with open(source_file, 'rb') as f:
        file_content = f.read()
    
    # Encode to base64
    encoded = base64.b64encode(file_content).decode('ascii')
    
    # Split into chunks
    chunks = split_into_chunks(encoded, chunk_size=50000)
    
    print(f"File size: {len(file_content):,} bytes")
    print(f"Encoded size: {len(encoded):,} characters")
    print(f"Number of chunks: {len(chunks)}")
    
    # Generate script
    script = f"""#!/bin/bash
# Auto-generated script to copy file via base64
# Run this when VM is accessible: bash copy_to_vm.sh

set -e

echo "=== Copying file via base64 ==="
echo ""

# Connect and create directory
ssh -o StrictHostKeyChecking=no {vm_user}@{vm_host} "mkdir -p ~/phase3/guest-shim"

# Write base64 data
echo "Writing base64 data..."
ssh -o StrictHostKeyChecking=no {vm_user}@{vm_host} 'cat > /tmp/libvgpu_cuda.c.b64 << '\''ENDOFFILE'\''
"""
    
    # Add chunks
    for i, chunk in enumerate(chunks):
        script += chunk + "\n"
        if (i + 1) % 10 == 0:
            script += f"# Chunk {i+1}/{len(chunks)}\n"
    
    script += """ENDOFFILE'

# Decode and write
echo "Decoding and writing file..."
ssh -o StrictHostKeyChecking=no """ + f"{vm_user}@{vm_host}" + """ 'base64 -d /tmp/libvgpu_cuda.c.b64 > ~/phase3/guest-shim/libvgpu_cuda.c && rm /tmp/libvgpu_cuda.c.b64'

# Verify
echo "Verifying file..."
ssh -o StrictHostKeyChecking=no """ + f"{vm_user}@{vm_host}" + """ 'wc -l ~/phase3/guest-shim/libvgpu_cuda.c && ls -lh ~/phase3/guest-shim/libvgpu_cuda.c'

echo ""
echo "File copied successfully!"
"""
    
    return script

def main():
    source_file = '/home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c'
    remote_path = '~/phase3/guest-shim/libvgpu_cuda.c'
    vm_user = 'test-7'
    vm_host = '10.25.33.17'
    
    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found")
        sys.exit(1)
    
    script = generate_copy_script(source_file, remote_path, vm_user, vm_host)
    
    output_file = '/home/david/Downloads/gpu/phase3/copy_to_vm.sh'
    with open(output_file, 'w') as f:
        f.write(script)
    
    os.chmod(output_file, 0o755)
    
    print(f"\nScript generated: {output_file}")
    print(f"\nTo copy file when VM is accessible, run:")
    print(f"  bash {output_file}")
    print(f"\nOr run manually:")
    print(f"  ssh {vm_user}@{vm_host}")
    print(f"  # Then paste the commands from the script")

if __name__ == '__main__':
    main()
