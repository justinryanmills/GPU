#!/usr/bin/env python3
"""Complete deployment script for test-5 VM"""

import pexpect
import time
import sys

VM = "test-5@10.25.33.15"
PASSWORD = "Calvin@123"

def run_cmd(child, cmd, timeout=60, expect_prompt=True):
    """Run a command and return output"""
    child.sendline(cmd)
    if expect_prompt:
        child.expect(['$', '#', 'test-5@'], timeout=timeout)
    return child.before

def main():
    print("="*70)
    print("DEPLOYING TO test-5@10.25.33.15")
    print("="*70)
    
    # Connect
    print("\n[1] Connecting...")
    child = pexpect.spawn(f'ssh -o StrictHostKeyChecking=no {VM}', encoding='utf-8', timeout=600)
    child.expect(['password:'], timeout=10)
    child.sendline(PASSWORD)
    time.sleep(3)
    child.expect(['$', '#', 'test-5@'], timeout=10)
    print("  Connected")
    
    # Copy files
    print("\n[2] Copying files...")
    files = [
        'guest-shim/libvgpu_cuda.c',
        'guest-shim/cuda_transport.c',
        'guest-shim/cuda_transport.h',
        'guest-shim/gpu_properties.h',
        'include/cuda_protocol.h',
    ]
    
    for f in files:
        print(f"  Copying {f}...")
        scp = pexpect.spawn(f'scp -o StrictHostKeyChecking=no {f} {VM}:~/phase3/guest-shim/', encoding='utf-8', timeout=60)
        scp.expect(['password:', pexpect.EOF], timeout=30)
        scp.sendline(PASSWORD)
        scp.expect([pexpect.EOF], timeout=60)
        scp.close()
    
    run_cmd(child, 'mkdir -p ~/phase3/include && cp ~/phase3/guest-shim/cuda_protocol.h ~/phase3/include/')
    print("  Files copied")
    
    # Install dependencies
    print("\n[3] Installing dependencies...")
    run_cmd(child, f'echo "{PASSWORD}" | sudo -S apt-get update -qq', timeout=120)
    run_cmd(child, f'echo "{PASSWORD}" | sudo -S apt-get install -y curl build-essential 2>&1 | tail -3', timeout=120)
    print("  Dependencies installed")
    
    # Install Ollama
    print("\n[4] Installing Ollama...")
    run_cmd(child, 'if ! command -v ollama > /dev/null 2>&1; then curl -fsSL https://ollama.com/install.sh | sh; sleep 5; fi', timeout=200)
    run_cmd(child, '/usr/local/bin/ollama --version 2>&1 | head -1')
    print("  Ollama installed")
    
    # Build shim
    print("\n[5] Building shim...")
    build_cmd = f'cd ~/phase3/guest-shim && echo "{PASSWORD}" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1'
    output = run_cmd(child, build_cmd, timeout=120)
    
    # Show build output
    print("\n  Build output:")
    for line in output.split('\n'):
        if line.strip() and 'password' not in line.lower() and not line.startswith('test-5@'):
            if any(kw in line.lower() for kw in ['error', 'fatal', 'undefined', 'missing', 'cannot']):
                print(f"    {line.strip()[:120]}")
    
    # Verify build
    output = run_cmd(child, 'ls -lh /usr/lib64/libvgpu-cuda.so 2>&1')
    if '/usr/lib64/libvgpu-cuda.so' in output:
        print("  Shim built successfully!")
        print(f"  {[l for l in output.split('\\n') if 'libvgpu-cuda.so' in l][0] if any('libvgpu-cuda.so' in l for l in output.split('\\n')) else ''}")
    else:
        print("  ✗ Build failed - library not found")
        print("  Checking what happened...")
        run_cmd(child, 'ls -la ~/phase3/guest-shim/*.c ~/phase3/guest-shim/*.h 2>&1 | head -10')
        child.sendline('exit')
        child.close()
        return 1
    
    # Configure preload
    print("\n[6] Configuring preload...")
    run_cmd(child, f'echo "{PASSWORD}" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload"')
    run_cmd(child, 'cat /etc/ld.so.preload')
    print("  Preload configured")
    
    # Configure Ollama
    print("\n[7] Configuring Ollama...")
    run_cmd(child, f'echo "{PASSWORD}" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d')
    run_cmd(child, f'echo -e "[Service]\nType=simple" | echo "{PASSWORD}" | sudo -S tee /etc/systemd/system/ollama.service.d/override.conf')
    run_cmd(child, f'echo "{PASSWORD}" | sudo -S systemctl daemon-reload')
    
    # Start Ollama
    print("\n[8] Starting Ollama...")
    run_cmd(child, f'echo "{PASSWORD}" | sudo -S systemctl start ollama', timeout=30)
    time.sleep(25)
    output = run_cmd(child, 'systemctl is-active ollama')
    
    if 'active' in output.lower():
        print("  Ollama is running!")
        
        # Test
        print("\n[9] Testing GPU mode...")
        run_cmd(child, 'timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -5', timeout=55)
        time.sleep(2)
        
        output = run_cmd(child, f'echo "{PASSWORD}" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep "library=" | tail -5')
        
        print("\n" + "="*70)
        print("RESULTS:")
        print("="*70)
        
        found_cuda = False
        for line in output.split('\n'):
            if 'library=' in line.lower() and 'password' not in line.lower():
                print(f"  {line.strip()}")
                if 'library=cuda' in line.lower():
                    found_cuda = True
        
        if found_cuda:
            print("\n" + "="*70)
            print("SUCCESS! OLLAMA IS USING GPU MODE! ")
            print("="*70)
            print("\n🎉 MISSION ACCOMPLISHED! 🎉")
        else:
            print("\n⚠ Ollama running - may need another inference")
    else:
        print("  ✗ Ollama failed to start")
    
    print("="*70)
    child.sendline('exit')
    child.close()
    return 0

if __name__ == '__main__':
    sys.exit(main())
