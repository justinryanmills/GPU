#!/bin/bash

VM="test-5@10.25.33.15"
PASSWORD="Calvin@123"

echo "======================================================================"
echo "COMPLETE DEPLOYMENT - CIRCULAR DEPENDENCY FIX"
echo "======================================================================"

echo ""
echo "[1] Copying fixed libvgpu_cuda.c..."
python3 << 'PYEOF'
import pexpect
import time

VM = "test-5@10.25.33.15"
PASSWORD = "Calvin@123"

for attempt in range(3):
    try:
        scp = pexpect.spawn('scp -o StrictHostKeyChecking=no /home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c test-5@10.25.33.15:~/phase3/guest-shim/libvgpu_cuda.c', encoding='utf-8', timeout=180)
        scp.expect(['password:', pexpect.EOF], timeout=30)
        scp.sendline(PASSWORD)
        scp.expect([pexpect.EOF], timeout=180)
        scp.close()
        print("  File copied")
        break
    except:
        if attempt < 2:
            time.sleep(3)
        else:
            print("  ⚠ Copy may have failed")
PYEOF

echo ""
echo "[2] Deploying on VM..."
python3 << 'PYEOF'
import pexpect
import time
import sys

VM = "test-5@10.25.33.15"
PASSWORD = "Calvin@123"

for attempt in range(5):
    try:
        child = pexpect.spawn(f'ssh -o StrictHostKeyChecking=no {VM}', encoding='utf-8', timeout=600)
        child.expect(['password:', pexpect.EOF], timeout=15)
        if 'password' in (child.before or '') or (hasattr(child, 'after') and child.after == 'password:'):
            child.sendline(PASSWORD)
        time.sleep(3)
        child.expect(['$', '#', 'test-5@'], timeout=10)
        break
    except:
        if attempt < 4:
            time.sleep(5)
            try:
                child.close()
            except:
                pass
        else:
            print("  ✗ Connection failed")
            sys.exit(1)

print("  Building shim...")
child.sendline('cd ~/phase3/guest-shim && echo "Calvin@123" | sudo -S gcc -shared -fPIC -o /usr/lib64/libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I../include -I. -ldl -lpthread -O2 -Wall 2>&1 | tee /tmp/build.log && test -f /usr/lib64/libvgpu-cuda.so && echo "BUILD_OK" || echo "BUILD_FAIL"')
child.expect(['BUILD_OK', 'BUILD_FAIL'], timeout=120)

if 'BUILD_FAIL' in child.before:
    child.sendline('cat /tmp/build.log | grep -i error | head -10')
    child.expect(['$', '#', 'test-5@'], timeout=30)
    print("  Build errors:")
    for line in child.before.split('\n'):
        if 'error' in line.lower() and 'password' not in line.lower():
            print(f"    {line.strip()[:100]}")
else:
    print("  Build successful!")
    
    print("  Testing lspci...")
    child.sendline('lspci 2>&1 | head -5')
    child.expect(['$', '#', 'test-5@'], timeout=20)
    
    if 'libvgpu-cuda' not in child.before:
        print("  lspci not intercepted!")
    else:
        print("  ✗ lspci still intercepted")
    
    child.sendline('lspci | grep -i "2331" || lspci | grep -i "3d controller"')
    child.expect(['$', '#', 'test-5@'], timeout=20)
    
    if '2331' in child.before or '3d controller' in child.before.lower():
        print("  Device visible!")
    else:
        print("  ⚠ Device not visible")
    
    print("  Restarting Ollama...")
    child.sendline('echo "Calvin@123" | sudo -S systemctl restart ollama && sleep 25 && systemctl is-active ollama && echo "OLLAMA_OK"')
    child.expect(['OLLAMA_OK', '$', '#', 'test-5@'], timeout=60)
    
    if 'OLLAMA_OK' in child.before:
        print("  Ollama running")
        
        child.sendline('timeout 50 ollama run llama3.2:1b "test" 2>&1 | head -3')
        child.expect(['$', '#', 'test-5@'], timeout=55)
        time.sleep(2)
        
        child.sendline('echo "Calvin@123" | sudo -S journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep "library=" | tail -3')
        child.expect(['$', '#', 'test-5@'], timeout=30)
        
        result = child.before
        print("\n  GPU Mode Check:")
        found_cuda = False
        for line in result.split('\n'):
            if 'library=' in line.lower():
                print(f"    {line.strip()}")
                if 'library=cuda' in line.lower():
                    found_cuda = True
        
        if found_cuda:
            print("\n  SUCCESS! GPU MODE ACTIVE! ")

child.sendline('exit')
child.close()
PYEOF

echo ""
echo "======================================================================"
echo "DEPLOYMENT COMPLETE"
echo "======================================================================"
