#!/usr/bin/env python3
"""Transfer only libvgpu_cublas.c to VM (checksum-verified), build and install.
Use this to deploy CUBLAS shim changes without full phase3 SCP.
"""
import os
import sys
import hashlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

CHUNK_SIZE = 40000
SOURCE = os.path.join(SCRIPT_DIR, "guest-shim", "libvgpu_cublas.c")


def run_vm(cmd, timeout_sec=120):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec,
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
    import base64
    b64 = base64.b64encode(data).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

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
    ok, out, err = run_vm(
        "base64 -d /tmp/combined.b64 > /tmp/libvgpu_cublas_new.c && wc -c /tmp/libvgpu_cublas_new.c"
    )
    print(out)
    if not ok:
        print("Decode failed:", err or out)
        return 1

    ok, out, err = run_vm(
        "python3 - << 'PYEOF'\n"
        "import hashlib\n"
        "p = '/tmp/libvgpu_cublas_new.c'\n"
        "with open(p, 'rb') as f:\n"
        "    d = f.read()\n"
        "print('REMOTE_SHA256=' + hashlib.sha256(d).hexdigest())\n"
        "PYEOF"
    )
    if not ok:
        print("Remote SHA256 failed:", err or out)
        return 1
    remote_sha = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("REMOTE_SHA256="):
            remote_sha = line.split("=", 1)[1].strip()
    if not remote_sha:
        print("Could not find REMOTE_SHA256")
        return 1
    print(f"Remote SHA256: {remote_sha}")
    if remote_sha != local_sha:
        print("ERROR: SHA256 mismatch. Aborting.")
        return 1
    print("SHA256 verified.")

    ok, out, err = run_vm(
        f"cp /tmp/libvgpu_cublas_new.c {REMOTE_PHASE3}/guest-shim/libvgpu_cublas.c 2>/dev/null || "
        "cp /tmp/libvgpu_cublas_new.c ~/phase3/guest-shim/libvgpu_cublas.c"
    )
    if not ok:
        print("Copy to guest-shim failed:", err or out)
        return 1

    build_cmd = (
        f"cd {REMOTE_PHASE3} && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cublas.so.12 guest-shim/libvgpu_cublas.c -ldl 2>&1; "
        "echo BUILD_EXIT=$?; ls -la /tmp/libvgpu-cublas.so.12 2>&1"
    )
    ok, out, err = run_vm(build_cmd, timeout_sec=120)
    print(out)
    if not ok:
        print("Build failed:", err or out)
        return 1
    if "BUILD_EXIT=0" not in (out or ""):
        print("Build may have failed. Output:", out)

    ok, out, err = run_vm(
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cublas.so.12 /opt/vgpu/lib/libvgpu-cublas.so.12 && "
        f"echo {VM_PASSWORD} | sudo -S ln -sf /opt/vgpu/lib/libvgpu-cublas.so.12 /opt/vgpu/lib/libcublas.so.12 && "
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
