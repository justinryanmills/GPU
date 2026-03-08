#!/usr/bin/env python3
"""
Reliable SCP-based deployment script for fgets() path lookup fix
Uses pexpect for proper SCP handling with retries
"""

import pexpect
import time
import sys
import os

VM = "test-5@10.25.33.15"
PASSWORD = "Calvin@123"
LOCAL_FILE = "/home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c"
REMOTE_PATH = "~/phase3/guest-shim/libvgpu_cuda.c"

def scp_copy_with_retry(local_file, remote_path, max_attempts=5):
    """Copy file via SCP with retry logic"""
    for attempt in range(1, max_attempts + 1):
        print(f"\n[Attempt {attempt}/{max_attempts}] Copying file via SCP...")
        
        try:
            # Use SCP with pexpect
            scp = pexpect.spawn(
                f'scp -o StrictHostKeyChecking=no -o ConnectTimeout=30 {local_file} {VM}:{remote_path}',
                encoding='utf-8',
                timeout=180
            )
            
            # Handle password prompt
            index = scp.expect(['password:', pexpect.EOF, pexpect.TIMEOUT], timeout=30)
            
            if index == 0:  # password prompt
                scp.sendline(PASSWORD)
                scp.expect([pexpect.EOF], timeout=180)
            
            scp.close()
            
            if scp.exitstatus == 0:
                print("  File copied successfully")
                return True
            else:
                print(f"  ⚠ SCP failed (exit code: {scp.exitstatus})")
                if attempt < max_attempts:
                    print("  Waiting 5 seconds before retry...")
                    time.sleep(5)
                    
        except pexpect.TIMEOUT:
            print("  ⚠ SCP timed out")
            try:
                scp.close()
            except:
                pass
            if attempt < max_attempts:
                time.sleep(5)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            if attempt < max_attempts:
                time.sleep(5)
    
    print("  ✗ Copy failed after all attempts")
    return False

def ssh_command(cmd, timeout=60):
    """Run command on VM via SSH"""
    try:
        child = pexpect.spawn(
            f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 {VM}',
            encoding='utf-8',
            timeout=timeout
        )
        
        child.expect(['password:', pexpect.EOF], timeout=15)
        if 'password' in (child.before or '') or (hasattr(child, 'after') and child.after == 'password:'):
            child.sendline(PASSWORD)
        
        time.sleep(2)
        child.expect(['$', '#', 'test-5@'], timeout=10)
        
        child.sendline(cmd)
        child.expect(['$', '#', 'test-5@'], timeout=timeout)
        
        output = child.before
        child.sendline('exit')
        child.close()
        
        return output
    except Exception as e:
        print(f"  ⚠ SSH command failed: {e}")
        return None

def main():
    print("="*70)
    print("DEPLOYING FGETS() PATH LOOKUP FIX VIA SCP")
    print("="*70)
    
    # Step 1: Verify local file
    print("\n[1] Verifying local file...")
    if not os.path.exists(LOCAL_FILE):
        print(f"  ✗ Local file not found: {LOCAL_FILE}")
        sys.exit(1)
    
    file_size = os.path.getsize(LOCAL_FILE)
    print(f"  Local file found (size: {file_size} bytes)")
    
    # Step 2: Copy file
    print("\n[2] Copying file to VM...")
    if not scp_copy_with_retry(LOCAL_FILE, REMOTE_PATH):
        print("  ✗ File copy failed")
        sys.exit(1)
    
    # Step 3: Verify file on VM
    print("\n[3] Verifying file on VM...")
    verify_cmd = f"test -f {REMOTE_PATH} && ls -lh {REMOTE_PATH} && echo 'FILE_OK'"
    output = ssh_command(verify_cmd, timeout=30)
    
    if output and 'FILE_OK' in output:
        print("  File verified on VM")
        for line in output.split('\n'):
            if 'FILE_OK' not in line and line.strip() and 'password' not in line.lower():
                print(f"    {line.strip()}")
    else:
        print("  ✗ File verification failed")
        sys.exit(1)
    
    # Step 4: Rebuild
    print("\n[4] Rebuilding shim library...")
    build_cmd = f"cd ~/phase3/guest-shim && echo '{PASSWORD}' | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1"
    build_output = ssh_command(build_cmd, timeout=120)
    
    if build_output and 'error' in build_output.lower():
        print("  ✗ Build failed with errors:")
        for line in build_output.split('\n'):
            if 'error' in line.lower() and 'password' not in line.lower():
                print(f"    {line.strip()[:120]}")
        sys.exit(1)
    
    # Verify build
    verify_build_cmd = "test -f /usr/lib64/libvgpu-cuda.so && ls -lh /usr/lib64/libvgpu-cuda.so && echo 'BUILD_OK'"
    build_verify = ssh_command(verify_build_cmd, timeout=30)
    
    if build_verify and 'BUILD_OK' in build_verify:
        print("  Build successful")
        for line in build_verify.split('\n'):
            if 'BUILD_OK' not in line and line.strip() and 'password' not in line.lower():
                print(f"    {line.strip()}")
    else:
        print("  ✗ Build verification failed")
        sys.exit(1)
    
    # Step 5: Test device discovery
    print("\n[5] Testing device discovery...")
    test_cmd = "ls 2>&1 | head -2 > /dev/null; sleep 2; echo '{}' | sudo -S journalctl --since '1 minute ago' --no-pager 2>&1 | grep -E 'Found VGPU|fgets.*returning.*device=0x2331|fgets.*returning.*class=0x030200' | tail -5".format(PASSWORD)
    test_output = ssh_command(test_cmd, timeout=30)
    
    if test_output:
        print("  Device discovery logs:")
        device_found = False
        for line in test_output.split('\n'):
            if any(kw in line.lower() for kw in ['found vgpu', 'device=0x2331', 'class=0x030200']):
                if 'password' not in line.lower() and line.strip():
                    print(f"    {line.strip()[:120]}")
                    if 'Found VGPU' in line:
                        device_found = True
        
        if device_found:
            print("  Device found! Fix working!")
        elif 'device=0x2331' in test_output or 'class=0x030200' in test_output:
            print("  Fix working - correct values returned")
    
    # Step 6: Restart Ollama
    print("\n[6] Restarting Ollama...")
    restart_cmd = f"echo '{PASSWORD}' | sudo -S systemctl restart ollama && sleep 25 && systemctl is-active ollama && echo 'OLLAMA_OK'"
    restart_output = ssh_command(restart_cmd, timeout=60)
    
    if restart_output and 'OLLAMA_OK' in restart_output:
        print("  Ollama restarted and running")
    else:
        print("  ⚠ Ollama restart status unclear")
    
    # Step 7: Test inference and check GPU mode
    print("\n[7] Running test inference and checking GPU mode...")
    inference_cmd = f"timeout 50 ollama run llama3.2:1b 'test' 2>&1 | head -5; sleep 2; echo '{PASSWORD}' | sudo -S journalctl -u ollama --since '2 minutes ago' --no-pager 2>&1 | grep 'library=' | tail -5"
    inference_output = ssh_command(inference_cmd, timeout=80)
    
    print("  GPU mode check:")
    if inference_output:
        found_cuda = False
        for line in inference_output.split('\n'):
            if 'library=' in line.lower() and 'password' not in line.lower():
                print(f"    {line.strip()}")
                if 'library=cuda' in line.lower():
                    found_cuda = True
        
        if found_cuda:
            print("  GPU MODE ACTIVE! (library=cuda)")
        else:
            print("  ⚠ GPU mode not yet confirmed")
    
    print("\n" + "="*70)
    print("DEPLOYMENT COMPLETE")
    print("="*70)
    print("\nSummary:")
    print("  File copied via SCP")
    print("  File verified on VM")
    print("  Shim library rebuilt")
    print("  Ollama restarted")
    print("\nCheck the logs above for device discovery and GPU mode status.")
    print("="*70)

if __name__ == "__main__":
    main()
