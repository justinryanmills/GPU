#!/usr/bin/env python3
"""DEPRECATED: use transfer_libvgpu_cuda.py for CUDA shim transfers."""

import sys
import os
import base64
import subprocess
import tempfile

def copy_file_to_vm(local_path, remote_path, vm_user=None, vm_host=None):
    """Copy a file from local filesystem to VM using chunked base64."""
    
    if not os.path.exists(local_path):
        print(f"Error: Local file not found: {local_path}")
        return False
    
    # Read the file
    with open(local_path, 'rb') as f:
        file_data = f.read()
    
    # Encode to base64
    b64_data = base64.b64encode(file_data).decode('utf-8')
    
    # Split into chunks of 8000 chars (safe for command line)
    chunk_size = 8000
    chunks = [b64_data[i:i+chunk_size] for i in range(0, len(b64_data), chunk_size)]
    
    print(f"File size: {len(file_data)} bytes, Base64: {len(b64_data)} chars, {len(chunks)} chunks")
    
    # Step 1: Write chunks to temp file on VM
    temp_file = "/tmp/file_transfer_base64.txt"
    
    # Clear temp file first
    cmd_clear = [
        sys.executable,
        "connect_vm.py",
        f"rm -f {temp_file} && touch {temp_file}"
    ]
    
    subprocess.run(cmd_clear, cwd=os.path.dirname(os.path.abspath(__file__)), 
                   capture_output=True, timeout=30)
    
    # Write chunks
    for i, chunk in enumerate(chunks):
        cmd_write = [
            sys.executable,
            "connect_vm.py",
            f'echo "{chunk}" >> {temp_file}'
        ]
        
        result = subprocess.run(cmd_write, cwd=os.path.dirname(os.path.abspath(__file__)),
                               capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Error writing chunk {i+1}/{len(chunks)}: {result.stderr}")
            return False
        
        if (i + 1) % 10 == 0:
            print(f"  Written {i+1}/{len(chunks)} chunks...")
    
    print(f"  All {len(chunks)} chunks written to {temp_file}")
    
    # Step 2: Decode and write final file
    python_decode = f'''
import base64
import os

# Read base64 from temp file
with open("{temp_file}", "r") as f:
    b64_data = f.read()

# Decode
data = base64.b64decode(b64_data)

# Create directory if needed
os.makedirs(os.path.dirname("{remote_path}"), exist_ok=True)

# Write file
with open("{remote_path}", "wb") as f:
    f.write(data)

# Verify
file_size = os.path.getsize("{remote_path}")
print(f"File written: {{file_size}} bytes to {remote_path}")

# Clean up
os.unlink("{temp_file}")
'''
    
    cmd_decode = [
        sys.executable,
        "connect_vm.py",
        f'python3 << "PYEOF"\n{python_decode}\nPYEOF'
    ]
    
    result = subprocess.run(cmd_decode, cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, timeout=120)
    
    if result.returncode == 0:
        print(f"Successfully copied {local_path} to {remote_path}")
        print(result.stdout)
        return True
    else:
        print(f"Error decoding file:")
        print(result.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: reliable_file_copy.py <local_path> <remote_path>")
        print("Example: reliable_file_copy.py guest-shim/libvgpu_cuda.c /home/test-3/phase3/guest-shim/libvgpu_cuda.c")
        print("Prefer: python3 deploy_to_test3.py  (SCP-based, no chunked transfer)")
        sys.exit(1)
    
    local_path = sys.argv[1]
    remote_path = sys.argv[2]
    
    # Extract just the path part if user@host:path format is used
    if ':' in remote_path:
        remote_path = remote_path.split(':', 1)[1]
    
    success = copy_file_to_vm(local_path, remote_path)
    sys.exit(0 if success else 1)
