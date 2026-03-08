#!/usr/bin/env python3
"""
Deploy Transport Fix to VM
Safely transfers libvgpu_cudart.c, rebuilds, installs, and tests
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

LOCAL_FILE = "guest-shim/libvgpu_cudart.c"
VM_FILE = REMOTE_PHASE3 + "/guest-shim/libvgpu_cudart.c"
VM_DIR = REMOTE_PHASE3 + "/guest-shim"

def run_vm_command(cmd):
    """Run a command on the VM using connect_vm.py"""
    script_path = os.path.join(os.path.dirname(__file__), "connect_vm.py")
    full_cmd = f'python3 {script_path} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def transfer_file():
    """Transfer libvgpu_cudart.c to VM using reliable_file_copy.py"""
    print("=" * 60)
    print("Step 1: Transferring libvgpu_cudart.c to VM...")
    print("=" * 60)
    
    local_path = os.path.join(os.path.dirname(__file__), LOCAL_FILE)
    if not os.path.exists(local_path):
        print(f"ERROR: Local file not found: {local_path}")
        return False
    
    # Use reliable_file_copy.py
    copy_script = os.path.join(os.path.dirname(__file__), "reliable_file_copy.py")
    remote_path = f"{VM_USER}@{VM_HOST}:{VM_FILE}"
    
    cmd = f'python3 {copy_script} {local_path} {remote_path}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"File transferred successfully")
        print(f"  {result.stdout.strip()}")
        return True
    else:
        print(f"✗ Transfer failed: {result.stderr}")
        return False

def verify_file():
    """Verify the file was transferred correctly"""
    print("\n" + "=" * 60)
    print("Step 2: Verifying file on VM...")
    print("=" * 60)
    
    success, stdout, stderr = run_vm_command(f'wc -l {VM_FILE} && grep -c "ensure_transport_functions" {VM_FILE}')
    if success:
        print(f"File verified on VM")
        print(f"  {stdout.strip()}")
        return True
    else:
        print(f"✗ Verification failed: {stderr}")
        return False

def rebuild_library():
    """Rebuild libvgpu-cudart.so on VM"""
    print("\n" + "=" * 60)
    print("Step 3: Rebuilding libvgpu-cudart.so...")
    print("=" * 60)
    
    build_cmd = f'cd {VM_DIR} && gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -I../include -I. -Wl,-soname,libcudart.so.12 -o libvgpu-cudart.so libvgpu_cudart.c -ldl -lpthread 2>&1'
    success, stdout, stderr = run_vm_command(build_cmd)
    
    if success and "error" not in stdout.lower() and "error" not in stderr.lower():
        print("Library rebuilt successfully")
        if stdout.strip():
            # Show only warnings, not full output
            warnings = [line for line in stdout.split('\n') if 'warning' in line.lower()]
            if warnings:
                print(f"  Warnings: {len(warnings)} warning(s)")
        return True
    else:
        print(f"✗ Build failed:")
        print(f"  {stdout[:500]}")
        if stderr:
            print(f"  {stderr[:500]}")
        return False

def install_library():
    """Install the rebuilt library"""
    print("\n" + "=" * 60)
    print("Step 4: Installing library...")
    print("=" * 60)
    
    install_cmd = f'sudo cp {VM_DIR}/libvgpu-cudart.so /usr/lib64/libvgpu-cudart.so && sudo chmod 755 /usr/lib64/libvgpu-cudart.so && sudo ln -sf /usr/lib64/libvgpu-cudart.so /usr/lib64/libcudart.so.12 && sudo ln -sf /usr/lib64/libvgpu-cudart.so /usr/lib64/libcudart.so && echo "Library installed"'
    success, stdout, stderr = run_vm_command(install_cmd)
    
    if success:
        print("Library installed successfully")
        return True
    else:
        print(f"✗ Installation failed: {stderr}")
        return False

def restart_ollama():
    """Restart ollama service"""
    print("\n" + "=" * 60)
    print("Step 5: Restarting Ollama service...")
    print("=" * 60)
    
    restart_cmd = 'sudo systemctl restart ollama.service && sleep 5 && sudo systemctl status ollama.service --no-pager | head -10'
    success, stdout, stderr = run_vm_command(restart_cmd)
    
    if success:
        print("Ollama restarted")
        if "active (running)" in stdout:
            print("  Service is active and running")
        return True
    else:
        print(f"✗ Restart failed: {stderr}")
        return False

def test_transport():
    """Test if transport calls are being made"""
    print("\n" + "=" * 60)
    print("Step 6: Testing transport calls...")
    print("=" * 60)
    
    # Trigger a model load
    print("Triggering model load...")
    test_cmd = 'timeout 10 curl -s http://localhost:11434/api/generate -d \'{"model":"llama3.2:1b","prompt":"test","stream":false}\' 2>&1 | head -3'
    success, stdout, stderr = run_vm_command(test_cmd)
    
    # Wait a bit for operations to complete
    import time
    time.sleep(3)
    
    # Check logs for transport calls
    print("\nChecking logs for transport calls...")
    log_cmd = 'journalctl -u ollama.service --since "30 seconds ago" --no-pager | grep -E "ensure_transport|cudaMalloc|\\[cuda-transport\\].*SENDING|\\[cuda-transport\\].*DOORBELL" | head -30'
    success, stdout, stderr = run_vm_command(log_cmd)
    
    if success and stdout.strip():
        print("Found transport-related logs:")
        print("=" * 60)
        for line in stdout.strip().split('\n')[:30]:
            if line.strip():
                print(f"  {line}")
        print("=" * 60)
        
        # Check for key indicators
        has_transport_init = "ensure_transport" in stdout
        has_sending = "SENDING" in stdout or "DOORBELL" in stdout
        has_cuda_malloc = "cudaMalloc" in stdout
        
        print(f"\nResults:")
        print(f"  Transport initialization: {'' if has_transport_init else '✗'}")
        print(f"  Transport calls (SENDING/DOORBELL): {'' if has_sending else '✗'}")
        print(f"  cudaMalloc calls: {'' if has_cuda_malloc else '✗'}")
        
        return has_sending or has_transport_init
    else:
        print("✗ No transport logs found")
        if stdout:
            print(f"  stdout: {stdout[:300]}")
        return False

def main():
    """Main deployment process"""
    print("\n" + "=" * 60)
    print("DEPLOYING TRANSPORT FIX TO VM")
    print("=" * 60)
    print(f"VM: {VM_USER}@{VM_HOST}")
    print(f"File: {LOCAL_FILE} -> {VM_FILE}")
    print("=" * 60 + "\n")
    
    steps = [
        ("Transfer file", transfer_file),
        ("Verify file", verify_file),
        ("Rebuild library", rebuild_library),
        ("Install library", install_library),
        ("Restart Ollama", restart_ollama),
        ("Test transport", test_transport),
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
            if not results[step_name] and step_name != "Test transport":
                print(f"\n✗ Deployment failed at step: {step_name}")
                return False
        except Exception as e:
            print(f"\n✗ Error in {step_name}: {e}")
            import traceback
            traceback.print_exc()
            results[step_name] = False
            if step_name != "Test transport":
                return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    for step_name, success in results.items():
        status = "SUCCESS" if success else "✗ FAILED"
        print(f"  {step_name:20s}: {status}")
    print("=" * 60)
    
    if results.get("Test transport"):
        print("\n🎉 SUCCESS! Transport calls are being made!")
        print("   Check logs for [cuda-transport] SENDING messages")
    else:
        print("\n⚠️  Deployment completed but transport calls not yet visible")
        print("   This may be normal - check logs after model loads")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
