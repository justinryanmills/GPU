#!/usr/bin/env python3
"""
Script to check and fix GPU attributes on VM
Connects to VM, compares code, and fixes MAX_THREADS_PER_BLOCK issue
"""

import paramiko
import sys
import os

VM_HOST = "10.25.33.110"
VM_USER = "test-10"
VM_PASSWORD = "Calvin@123"

def connect_to_vm():
    """Connect to VM via SSH"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print(f"Connecting to {VM_USER}@{VM_HOST}...")
        ssh.connect(VM_HOST, username=VM_USER, password=VM_PASSWORD, timeout=10)
        print("Connected successfully!")
        return ssh
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None

def find_shim_code(ssh):
    """Find shim source code location on VM"""
    print("\n=== Finding shim source code on VM ===")
    commands = [
        "find ~ -name 'libvgpu_cuda.c' -type f 2>/dev/null | head -3",
        "find /home -name 'libvgpu_cuda.c' -type f 2>/dev/null | head -3",
        "find /opt -name 'libvgpu_cuda.c' -type f 2>/dev/null | head -3",
        "ls -la /usr/lib64/libvgpu-cuda.so 2>/dev/null",
    ]
    
    for cmd in commands:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()
        if output:
            print(f"Command: {cmd}")
            print(f"Output: {output}")
            if 'libvgpu_cuda.c' in output:
                return output.split('\n')[0]
        if error and 'No such file' not in error:
            print(f"Error: {error}")
    
    return None

def check_gpu_properties(ssh, code_path):
    """Check GPU properties header file"""
    print(f"\n=== Checking GPU properties in {code_path} ===")
    
    # Find gpu_properties.h
    dir_path = os.path.dirname(code_path) if code_path else None
    if dir_path:
        cmd = f"find {dir_path} -name 'gpu_properties.h' -type f 2>/dev/null | head -1"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        props_file = stdout.read().decode('utf-8').strip()
        
        if props_file:
            print(f"Found gpu_properties.h: {props_file}")
            # Check MAX_THREADS_PER_BLOCK value
            cmd = f"grep -n 'MAX_THREADS_PER_BLOCK' {props_file}"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            output = stdout.read().decode('utf-8').strip()
            print(f"MAX_THREADS_PER_BLOCK definition:\n{output}")
            return props_file
    
    return None

def check_cuDeviceGetAttribute(ssh, code_path):
    """Check cuDeviceGetAttribute implementation"""
    print(f"\n=== Checking cuDeviceGetAttribute in {code_path} ===")
    if not code_path:
        return None
    
    cmd = f"grep -A 5 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK' {code_path} | head -10"
    stdin, stdout, stderr = ssh.exec_command(cmd)
    output = stdout.read().decode('utf-8').strip()
    print(f"MAX_THREADS_PER_BLOCK handling:\n{output}")
    
    return output

def main():
    print("=" * 60)
    print("VM GPU Attributes Check and Fix Script")
    print("=" * 60)
    
    ssh = connect_to_vm()
    if not ssh:
        print("Failed to connect to VM. Exiting.")
        return 1
    
    try:
        # Find shim code
        code_path = find_shim_code(ssh)
        print(f"\nShim code location: {code_path}")
        
        # Check GPU properties
        props_file = check_gpu_properties(ssh, code_path)
        
        # Check cuDeviceGetAttribute
        attr_code = check_cuDeviceGetAttribute(ssh, code_path)
        
        print("\n" + "=" * 60)
        print("Analysis complete. Check output above.")
        print("=" * 60)
        
    finally:
        ssh.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
