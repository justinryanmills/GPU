#!/usr/bin/env python3
"""
Full setup on a fresh VM (test-3): create phase3 tree, copy all shim sources,
build libvgpu-cuda.so.1, install to /opt/vgpu/lib, restart ollama, and test.
Uses vm_config.py for VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3.
"""
import os
import sys
import base64
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

CHUNK_SIZE = 40000

def run_vm(cmd, timeout_sec=120):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""

def send_file(local_path, remote_path, timeout_sec=180):
    """Copy local file to VM at remote_path via chunked base64."""
    if not os.path.isfile(local_path):
        print(f"Missing: {local_path}")
        return False
    with open(local_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    def escape(s):
        return s.replace("'", "'\"'\"'")
    ok, _, _ = run_vm("rm -f /tmp/combined.b64", timeout_sec=30)
    if not ok:
        return False
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
        ok, _, _ = run_vm(cmd, timeout_sec=60)
        if not ok:
            return False
    # Decode and write to final path (parent dir must exist)
    ok, _, _ = run_vm(
        f"base64 -d /tmp/combined.b64 > /tmp/out.tmp && mv /tmp/out.tmp {remote_path}",
        timeout_sec=30,
    )
    return ok

def main():
    print("Target VM:", f"{VM_USER}@{VM_HOST}", f"phase3={REMOTE_PHASE3}")
    # 0) Ensure gcc is installed
    print("0. Ensuring gcc is installed...")
    ok, _, _ = run_vm("which gcc 2>/dev/null")
    if not ok:
        run_vm(f"echo {VM_PASSWORD} | sudo -S apt-get update -qq && echo {VM_PASSWORD} | sudo -S apt-get install -y -qq gcc make", timeout_sec=180)
    # 1) Create directories on VM
    print("1. Creating directories...")
    ok, out, err = run_vm(
        f"mkdir -p {REMOTE_PHASE3}/guest-shim {REMOTE_PHASE3}/include && "
        f"echo {VM_PASSWORD} | sudo -S mkdir -p /opt/vgpu/lib"
    )
    if not ok:
        print("Failed:", err or out)
        return 1
    # 2) Copy required files
    files = [
        (os.path.join(SCRIPT_DIR, "guest-shim", "libvgpu_cuda.c"), f"{REMOTE_PHASE3}/guest-shim/libvgpu_cuda.c"),
        (os.path.join(SCRIPT_DIR, "guest-shim", "cuda_transport.c"), f"{REMOTE_PHASE3}/guest-shim/cuda_transport.c"),
        (os.path.join(SCRIPT_DIR, "guest-shim", "cuda_transport.h"), f"{REMOTE_PHASE3}/guest-shim/cuda_transport.h"),
        (os.path.join(SCRIPT_DIR, "guest-shim", "gpu_properties.h"), f"{REMOTE_PHASE3}/guest-shim/gpu_properties.h"),
        (os.path.join(SCRIPT_DIR, "include", "cuda_protocol.h"), f"{REMOTE_PHASE3}/include/cuda_protocol.h"),
    ]
    for local, remote in files:
        print(f"   Sending {os.path.basename(local)} -> {remote}")
        if not send_file(local, remote):
            print(f"   Failed to send {local}")
            return 1
    # 3) Build
    print("2. Building libvgpu-cuda.so.1...")
    ok, out, err = run_vm(
        f"cd {REMOTE_PHASE3} && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cuda.so.1 guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread 2>&1",
        timeout_sec=300,
    )
    print(out or err)
    ok2, out2, _ = run_vm("ls -la /tmp/libvgpu-cuda.so.1 2>&1")
    if not ok2 or "No such file" in (out2 or ""):
        print("Build failed (binary not found).")
        return 1
    # 4) Install shim and restart ollama
    print("3. Installing shim to /opt/vgpu/lib and restarting ollama...")
    ok, out, err = run_vm(
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libcuda.so.1 && "
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama"
    )
    if not ok:
        print("Install/restart failed:", err or out)
        return 1
    # 5) Ensure ollama is running and has model
    print("4. Checking ollama and model...")
    ok, out, err = run_vm("systemctl is-active ollama 2>/dev/null || true; curl -s http://127.0.0.1:11434/api/tags 2>/dev/null || echo 'ollama not responding'")
    print(out)
    if "ollama not responding" in (out or ""):
        print("Ollama not running or not installed. Start it or install ollama first.")
    # 6) Pull model if missing (optional)
    run_vm("ollama pull llama3.2:1b 2>&1 | tail -3", timeout_sec=300)
    # 7) Test generate
    print("5. Testing api/generate...")
    ok, out, err = run_vm(
        "curl -s -X POST http://127.0.0.1:11434/api/generate -d '{\"model\":\"llama3.2:1b\",\"prompt\":\"Hi\",\"stream\":false}'"
    )
    print(out)
    if "error" in (out or ""):
        print("Generate returned an error (see above).")
        return 1
    print("Setup and test done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
