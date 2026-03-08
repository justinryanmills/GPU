#!/usr/bin/env python3
"""Installation and status check script that runs on VM"""
import subprocess
import sys
import os

PASS = "Calvin@123"

def run(cmd, desc=""):
    """Run command and print output"""
    if desc:
        print(f"\n{'='*70}")
        print(f"{desc}")
        print('='*70)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    return result.returncode == 0

print("="*70)
print("SHIM INSTALLATION AND STATUS CHECK")
print("="*70)

# Step 1: Install ld.so.preload
run(f'echo "{PASS}" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload && echo /usr/lib64/libvgpu-nvml.so >> /etc/ld.so.preload && chmod 644 /etc/ld.so.preload"', 
    "STEP 1: Creating /etc/ld.so.preload")
run(f'echo "{PASS}" | sudo -S cat /etc/ld.so.preload', "Verifying ld.so.preload")

# Step 2: Build LD_AUDIT
os.chdir(os.path.expanduser("~/phase3/guest-shim"))
run(f'echo "{PASS}" | sudo -S gcc -shared -fPIC -o /usr/lib64/libldaudit_cuda.so ld_audit_interceptor.c -ldl -O2 -Wall', 
    "STEP 2: Building LD_AUDIT interceptor")
run('ls -la /usr/lib64/libldaudit_cuda.so', "LD_AUDIT file")

# Step 3: Build force_load_shim
run(f'echo "{PASS}" | sudo -S gcc -o /usr/local/bin/force_load_shim force_load_shim.c -ldl -O2 -Wall', 
    "STEP 3: Building force_load_shim")
run('ls -la /usr/local/bin/force_load_shim', "force_load_shim file")

# Step 4: Test shim
run('gcc -o /tmp/test_shim_load test_shim_load.c -ldl', "STEP 4: Compiling test")
run('/tmp/test_shim_load', "Testing shim loading")

# Step 5: Restart Ollama
run(f'echo "{PASS}" | sudo -S systemctl restart ollama', "STEP 5: Restarting Ollama")
import time
time.sleep(5)

# Step 6: Check logs
run(f'echo "{PASS}" | sudo -S journalctl -u ollama -n 150 --no-pager | grep -iE "libvgpu|LOADED|cuInit" | head -20', 
    "STEP 6: Ollama logs (shim-related)")

# Step 7: Test Ollama
run('ollama info 2>&1 | head -50', "STEP 7: Ollama GPU detection")

# Step 8: Final status
run(f'echo "{PASS}" | sudo -S cat /etc/ld.so.preload', "STEP 8: Final ld.so.preload")
run('ls -la /usr/lib64/libldaudit_cuda.so /usr/local/bin/force_load_shim', "Final installed files")

print("\n" + "="*70)
print("INSTALLATION AND STATUS CHECK COMPLETE")
print("="*70)
