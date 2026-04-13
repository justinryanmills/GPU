#!/usr/bin/env python3
"""Final installation and verification script"""
import subprocess
import sys
import os
import time

PASS = "Calvin@123"

def run(cmd, desc=""):
    """Run command and return output"""
    if desc:
        print(f"\n{'='*70}")
        print(f"{desc}")
        print('='*70)
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    return result.stdout, result.stderr

print("="*70)
print("FINAL INSTALLATION AND VERIFICATION")
print("="*70)

# Step 1: Fix ld.so.preload
stdout, stderr = run(f'echo "{PASS}" | sudo -S bash -c "echo /usr/lib64/libvgpu-cuda.so > /etc/ld.so.preload && echo /usr/lib64/libvgpu-nvml.so >> /etc/ld.so.preload && chmod 644 /etc/ld.so.preload"', 
    "STEP 1: Configuring /etc/ld.so.preload")
stdout, stderr = run(f'echo "{PASS}" | sudo -S cat /etc/ld.so.preload', "ld.so.preload contents")

# Step 2: Build tools
os.chdir(os.path.expanduser("~/phase3/guest-shim"))
stdout, stderr = run(f'echo "{PASS}" | sudo -S gcc -shared -fPIC -o /usr/lib64/libldaudit_cuda.so ld_audit_interceptor.c -ldl -O2 -Wall', 
    "STEP 2: Building LD_AUDIT")
stdout, stderr = run(f'echo "{PASS}" | sudo -S gcc -o /usr/local/bin/force_load_shim force_load_shim.c -ldl -O2 -Wall', 
    "STEP 3: Building force_load_shim")

# Step 3: Systemd override
stdout, stderr = run(f'echo "{PASS}" | sudo -S mkdir -p /etc/systemd/system/ollama.service.d', "STEP 4: Creating systemd override")
override_content = '[Service]\nEnvironment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so /usr/lib64/libvgpu-nvml.so"\n'
with open('/tmp/override.conf', 'w') as f:
    f.write(override_content)
stdout, stderr = run(f'echo "{PASS}" | sudo -S cp /tmp/override.conf /etc/systemd/system/ollama.service.d/override.conf', "Copying override")

# Step 4: Restart Ollama
stdout, stderr = run(f'echo "{PASS}" | sudo -S systemctl daemon-reload', "STEP 5: Reloading systemd")
stdout, stderr = run(f'echo "{PASS}" | sudo -S systemctl restart ollama', "STEP 6: Restarting Ollama")
time.sleep(5)

# Step 5: Check logs
stdout, stderr = run(f'echo "{PASS}" | sudo -S journalctl -u ollama -n 200 --no-pager | grep -iE "libvgpu|LOADED|cuInit" | head -30', 
    "STEP 7: Ollama logs (shim-related)")

# Step 6: Check process libraries
stdout, stderr = run(f'echo "{PASS}" | sudo -S pidof ollama', "STEP 8: Finding Ollama PID")
pid = stdout.strip().split()[0] if stdout.strip() else None
if pid:
    stdout, stderr = run(f'echo "{PASS}" | sudo -S cat /proc/{pid}/maps | grep -E "libvgpu|libcuda" | head -10', 
        f"Ollama process libraries (PID {pid})")

# Step 7: Check shim log files
stdout, stderr = run('ls -la /tmp/vgpu-shim-cuda-*.log 2>&1 | head -10', "STEP 9: Shim log files")
if stdout.strip():
    for logfile in stdout.strip().split('\n'):
        if '.log' in logfile:
            logpath = logfile.split()[-1] if logfile.split() else None
            if logpath:
                stdout, stderr = run(f'cat {logpath}', f"Contents of {logpath}")

# Step 8: Test Ollama
stdout, stderr = run('ollama info 2>&1 | head -50', "STEP 10: Ollama GPU detection")

# Check for GPU indicators
output_lower = stdout.lower()
if 'gpu' in output_lower or 'cuda' in output_lower or 'nvidia' in output_lower:
    print("\n" + "="*70)
    print("✓ GPU DETECTION INDICATORS FOUND")
    print("="*70)
else:
    print("\n" + "="*70)
    print("✗ NO GPU DETECTION - SHIMS MAY NOT BE LOADED")
    print("="*70)

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
