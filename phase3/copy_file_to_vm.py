#!/usr/bin/env python3
"""
Simple script to copy a file from host to VM using the connect_vm.py infrastructure.
Uses base64 encoding for reliable binary/text file transfer.
"""

import sys
import os
import base64
import subprocess

def copy_file_to_vm(local_path, remote_path, vm_user=None, vm_host=None):
    """Copy a file from local filesystem to VM."""
    
    if not os.path.exists(local_path):
        print(f"Error: Local file not found: {local_path}")
        return False
    
    # Read the file
    with open(local_path, 'rb') as f:
        file_data = f.read()
    
    # Write base64 to temp file first (to avoid command line length limits)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.b64', delete=False) as tmp:
        b64_data = base64.b64encode(file_data).decode('utf-8')
        tmp.write(b64_data)
        tmp_path = tmp.name
    
    try:
        # Create Python script to read from stdin and decode
        python_script = f'''
import base64
import os
import sys

# Read base64 from stdin
b64_data = sys.stdin.read()
data = base64.b64decode(b64_data)

# Create directory if needed
os.makedirs(os.path.dirname("{remote_path}"), exist_ok=True)

# Write file
with open("{remote_path}", "wb") as f:
    f.write(data)

print(f"File copied: {{len(data)}} bytes written to {remote_path}")
'''
        
        # Execute on VM using connect_vm.py, piping the base64 data
        cmd = [
            sys.executable,
            "connect_vm.py",
            f'python3 << "PYEOF"\n{python_script}\nPYEOF'
        ]
        
        # Use the temp file as input - read it first to ensure it's valid
        with open(tmp_path, 'r') as tmp_file:
            b64_content = tmp_file.read()
        
        if not b64_content:
            print(f"Error: Base64 file is empty")
            return False
        
        # Create Python script that reads from a file instead of stdin
        python_script = f'''
import base64
import os

# Read base64 from file
with open("/tmp/libvgpu_cuda_base64.txt", "r") as f:
    b64_data = f.read()

data = base64.b64decode(b64_data)

# Create directory if needed
os.makedirs(os.path.dirname("{remote_path}"), exist_ok=True)

# Write file
with open("{remote_path}", "wb") as f:
    f.write(data)

print(f"File copied: {{len(data)}} bytes written to {remote_path}")
'''
        
        # Write base64 to temp file on VM first
        cmd_write = [
            sys.executable,
            "connect_vm.py",
            f'python3 << "PYEOF"\nimport base64\nb64_data = """{b64_data}"""\nwith open("/tmp/libvgpu_cuda_base64.txt", "w") as f:\n    f.write(b64_data)\nprint("Base64 written to /tmp/libvgpu_cuda_base64.txt")\nPYEOF'
        ]
        
        process = subprocess.Popen(
            cmd_write,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=120)
            
            if process.returncode == 0:
                print(f"Successfully copied {local_path} to {remote_path}")
                print(stdout)
                return True
            else:
                print(f"Error copying file:")
                print(stderr)
                return False
    finally:
        # Clean up temp file
        os.unlink(tmp_path)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"Successfully copied {local_path} to {remote_path}")
            print(result.stdout)
            return True
        else:
            print(f"Error copying file:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("Error: Copy operation timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: copy_file_to_vm.py <local_path> <remote_path>")
        print("Example: copy_file_to_vm.py guest-shim/libvgpu_cuda.c /home/test-3/phase3/guest-shim/libvgpu_cuda.c")
        sys.exit(1)
    
    local_path = sys.argv[1]
    remote_path = sys.argv[2]
    
    success = copy_file_to_vm(local_path, remote_path)
    sys.exit(0 if success else 1)
