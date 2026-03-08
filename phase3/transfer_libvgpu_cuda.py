#!/usr/bin/env python3
"""Transfer libvgpu_cuda.c to VM in chunks via connect_vm, then build and install.

This version is **checksum-verified** end-to-end:
- Computes SHA-256 locally.
- Reconstructs the file on the VM.
- Computes SHA-256 on the VM via a small Python snippet.
- Compares the two hashes and aborts if they differ.
"""
import os
import sys
import base64
import hashlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

CHUNK_SIZE = 40000  # Stay under typical argv limits
SOURCE = os.path.join(SCRIPT_DIR, "guest-shim", "libvgpu_cuda.c")

def run_vm(cmd, timeout_sec=120):
    """Run command on VM via connect_vm.py."""
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""

def main():
    if not os.path.isfile(SOURCE):
        print(f"Source not found: {SOURCE}")
        return 1
    with open(SOURCE, "rb") as f:
        data = f.read()
    local_sha = hashlib.sha256(data).hexdigest()
    print(f"Local SHA256: {local_sha}")
    b64 = base64.b64encode(data).decode("ascii")
    # Escape for single-quoted shell: ' -> '\''
    def escape(s):
        return s.replace("'", "'\"'\"'")
    # Clear and send chunks
    ok, out, err = run_vm("rm -f /tmp/combined.b64")
    if not ok:
        print("Failed to clear remote file:", err or out)
        return 1
    n = 0
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        n += 1
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
        ok, out, err = run_vm(cmd)
        if not ok:
            print(f"Chunk {n} failed:", err or out)
            return 1
        print(f"Chunk {n} sent ({len(chunk)} chars)")
    # Decode and write on VM
    ok, out, err = run_vm(
        "base64 -d /tmp/combined.b64 > /tmp/libvgpu_cuda_new.c && "
        "wc -c /tmp/libvgpu_cuda_new.c"
    )
    print(out)
    if not ok:
        print("Decode failed:", err or out)
        return 1

    # Verify SHA256 on VM
    ok, out, err = run_vm(
        "python3 - << 'PYEOF'\n"
        "import hashlib\n"
        "p = '/tmp/libvgpu_cuda_new.c'\n"
        "with open(p, 'rb') as f:\n"
        "    d = f.read()\n"
        "print('REMOTE_SHA256=' + hashlib.sha256(d).hexdigest())\n"
        "PYEOF"
    )
    if not ok:
        print("Remote SHA256 computation failed:", err or out)
        return 1
    remote_sha = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("REMOTE_SHA256="):
            remote_sha = line.split("=", 1)[1].strip()
    if not remote_sha:
        print("Could not find REMOTE_SHA256 line in VM output:")
        print(out)
        return 1
    print(f"Remote SHA256: {remote_sha}")
    if remote_sha != local_sha:
        print("ERROR: SHA256 mismatch between local and VM copy. Aborting.")
        return 1
    print("SHA256 verified: local and VM copies match.")
    # Copy to phase3 tree and build (assume phase3 is in home)
    ok, out, err = run_vm(
        f"cp /tmp/libvgpu_cuda_new.c {REMOTE_PHASE3}/guest-shim/libvgpu_cuda.c 2>/dev/null || "
        "cp /tmp/libvgpu_cuda_new.c ~/phase3/guest-shim/libvgpu_cuda.c"
    )
    if not ok:
        print("Copy to guest-shim failed:", err or out)
        return 1
    # Build directly with gcc to /tmp so we don't depend on full 'make guest' (which builds multiple libs)
    build_cmd = (
        f"cd {REMOTE_PHASE3} && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cuda.so.1 guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread 2>&1; "
        "echo BUILD_EXIT=$?; ls -la /tmp/libvgpu-cuda.so.1 2>&1"
    )
    ok, out, err = run_vm(build_cmd, timeout_sec=300)
    print(out)
    if not ok:
        print("Build failed:", err or out)
        return 1
    if "BUILD_EXIT=0" not in (out or ""):
        print("Build may have failed (BUILD_EXIT not 0). Output:", out)
    ok, out, err = run_vm(
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1 && "
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama"
    )
    print(out)
    if not ok:
        print("Install/restart failed:", err or out)
        return 1
    print("Transfer, build, install, and restart done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
