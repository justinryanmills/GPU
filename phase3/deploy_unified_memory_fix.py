#!/usr/bin/env python3
"""Deploy unified memory API fix to VM: transfer, rebuild, install, restart Ollama."""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

VM_PASS = VM_PASSWORD
VM_PHASE3 = REMOTE_PHASE3

def run_vm_command(cmd):
    """Run a command on the VM via SSH."""
    ssh_cmd = [
        "sshpass", "-p", VM_PASS,
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"{VM_USER}@{VM_HOST}",
        cmd
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def copy_file_to_vm(local_path, remote_path):
    """Copy a file to the VM using scp."""
    scp_cmd = [
        "sshpass", "-p", VM_PASS,
        "scp", "-o", "StrictHostKeyChecking=no",
        local_path,
        f"{VM_USER}@{VM_HOST}:{remote_path}"
    ]
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    return result.returncode == 0

def main():
    print("=== Deploying Unified Memory API Fix to VM ===\n")
    
    # Step 1: Copy libvgpu_cuda.c to VM
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_file = os.path.join(script_dir, "guest-shim/libvgpu_cuda.c")
    remote_file = f"{VM_PHASE3}/guest-shim/libvgpu_cuda.c"
    
    if not os.path.exists(local_file):
        print(f"ERROR: Local file not found: {local_file}")
        return 1
    
    print(f"Step 1: Copying {local_file} to VM...")
    if not copy_file_to_vm(local_file, remote_file):
        print(f"ERROR: Failed to copy {local_file} to VM")
        return 1
    print("File copied successfully\n")
    
    # Step 2: Rebuild libvgpu-cuda.so.1 on VM
    print("Step 2: Rebuilding libvgpu-cuda.so.1 on VM...")
    build_cmd = f"""
        cd {VM_PHASE3}/guest-shim && \
        gcc -shared -fPIC -o libvgpu-cuda.so.1 libvgpu_cuda.c cuda_transport.c \
            -I../include -I. -ldl -Wall -Werror 2>&1
    """
    rc, stdout, stderr = run_vm_command(build_cmd)
    if rc != 0:
        print(f"ERROR: Build failed")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1
    print("Build successful\n")
    
    # Step 3: Install to /usr/lib64/
    print("Step 3: Installing libvgpu-cuda.so.1 to /usr/lib64/...")
    install_cmd = f"""
        sudo cp {VM_PHASE3}/guest-shim/libvgpu-cuda.so.1 /usr/lib64/ && \
        sudo chmod 755 /usr/lib64/libvgpu-cuda.so.1 && \
        sudo ldconfig
    """
    rc, stdout, stderr = run_vm_command(install_cmd)
    if rc != 0:
        print(f"ERROR: Installation failed")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return 1
    print("Installation successful\n")
    
    # Step 4: Restart Ollama service
    print("Step 4: Restarting Ollama service...")
    restart_cmd = "sudo systemctl restart ollama.service"
    rc, stdout, stderr = run_vm_command(restart_cmd)
    if rc != 0:
        print(f"WARNING: Ollama restart may have failed")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
    else:
        print("Ollama service restarted\n")
    
    # Step 5: Wait for service to be ready
    print("Step 5: Waiting for Ollama service to be ready...")
    import time
    time.sleep(3)
    
    # Step 6: Test with a simple request
    print("Step 6: Testing with simple Ollama request...")
    test_cmd = "timeout 30 ollama run llama3.2:1b 'Hello' 2>&1 | head -20"
    rc, stdout, stderr = run_vm_command(test_cmd)
    print(f"Test output:\n{stdout}")
    if stderr:
        print(f"Test errors:\n{stderr}")
    
    print("\n=== Deployment Complete ===")
    print("\nNext steps:")
    print("1. Check Ollama logs: journalctl -u ollama.service -f")
    print("2. Look for cuMemCreate and cuMemMap logs")
    print("3. Verify no crash occurs")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
