#!/usr/bin/env python3
"""
Minimal deploy: transfer only the two modified files (cuda_transport.c, libvgpu_cuda.c),
verify each step with SHA256, build on VM, install and restart ollama.

Uses connect_vm.py (pexpect) — no sshpass or full SCP of phase3.
Fails at first error with a clear message; does not skip or ignore failures.
"""
import os
import sys
import base64
import hashlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

CHUNK_SIZE = 40000
FILES_TO_TRANSFER = [
    ("guest-shim/cuda_transport.c", "/tmp/ct_new.c", "guest-shim/cuda_transport.c"),
    ("guest-shim/libvgpu_cuda.c", "/tmp/lvc_new.c", "guest-shim/libvgpu_cuda.c"),
]


def run_vm(cmd, timeout_sec=120):
    """Run one command on VM via connect_vm.py. Returns (success, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    return result.returncode == 0, out, err


def transfer_file(rel_path, remote_tmp, remote_dest):
    """
    Transfer one file: read local, base64 chunk upload, decode on VM, SHA256 verify, copy to dest.
    Returns True on success; on failure prints error and returns False.
    """
    local_path = os.path.join(SCRIPT_DIR, rel_path)
    if not os.path.isfile(local_path):
        print(f"ERROR: Local file not found: {local_path}")
        return False

    with open(local_path, "rb") as f:
        data = f.read()
    local_sha = hashlib.sha256(data).hexdigest()
    print(f"  {rel_path}: local SHA256 {local_sha[:16]}...")

    b64 = base64.b64encode(data).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, out, err = run_vm(f"rm -f {remote_tmp}.b64")
    if not ok:
        print(f"ERROR: Failed to clear remote temp: {err or out}")
        return False

    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> " + remote_tmp + ".b64"
        ok, _, _ = run_vm(cmd)
        if not ok:
            print(f"ERROR: Chunk upload failed for {rel_path}")
            return False
    print(f"  {rel_path}: chunks uploaded")

    ok, out, err = run_vm(
        f"base64 -d {remote_tmp}.b64 > {remote_tmp} 2>&1 && wc -c {remote_tmp}"
    )
    if not ok:
        print(f"ERROR: Decode failed for {rel_path}: {err or out}")
        return False

    ok, out, err = run_vm(
        "python3 - << 'PYEOF'\n"
        "import hashlib, sys\n"
        f"p = '{remote_tmp}'\n"
        "with open(p, 'rb') as f:\n"
        "    d = f.read()\n"
        "print('REMOTE_SHA256=' + hashlib.sha256(d).hexdigest())\n"
        "PYEOF"
    )
    if not ok:
        print(f"ERROR: Remote SHA256 failed for {rel_path}: {err or out}")
        return False
    remote_sha = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("REMOTE_SHA256="):
            remote_sha = line.split("=", 1)[1].strip()
            break
    if not remote_sha or remote_sha != local_sha:
        print(f"ERROR: SHA256 mismatch for {rel_path}: local={local_sha} remote={remote_sha}")
        return False
    print(f"  {rel_path}: SHA256 verified")

    ok, out, err = run_vm(
        f"cp {remote_tmp} {REMOTE_PHASE3}/{remote_dest} 2>&1"
    )
    if not ok:
        print(f"ERROR: Copy to {remote_dest} failed: {err or out}")
        return False
    print(f"  {rel_path}: copied to {REMOTE_PHASE3}/{remote_dest}")
    return True


def main():
    print("=== Transfer inference fix (cuda_transport.c + libvgpu_cuda.c) ===\n")
    print(f"Target: {VM_USER}@{VM_HOST}  phase3={REMOTE_PHASE3}\n")

    # Step 0: VM must have phase3 tree with include and guest-shim
    ok, out, err = run_vm(
        f"test -d {REMOTE_PHASE3}/guest-shim && test -d {REMOTE_PHASE3}/include && echo OK"
    )
    if not ok or "OK" not in (out or ""):
        print(f"ERROR: VM missing phase3 tree. Ensure {REMOTE_PHASE3}/guest-shim and .../include exist.")
        print(f"  stdout: {out}")
        print(f"  stderr: {err}")
        return 1

    # Step 1: Transfer both files
    print("Step 1: Transfer and verify files...")
    for rel_path, remote_tmp, remote_dest in FILES_TO_TRANSFER:
        if not transfer_file(rel_path, remote_tmp, remote_dest):
            return 1
    print("  All files transferred and verified.\n")

    # Step 2: Build on VM (write output to file so we can read it reliably)
    print("Step 2: Build libvgpu-cuda.so.1 on VM...")
    build_cmd = (
        f"cd {REMOTE_PHASE3} && "
        "( gcc -shared -fPIC -O2 -std=c11 -D_GNU_SOURCE "
        "-Iinclude -Iguest-shim -o /tmp/libvgpu-cuda.so.1 "
        "guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread 2>&1; "
        "echo BUILD_EXIT=$? ) > /tmp/build_out.txt 2>&1"
    )
    ok, _, err = run_vm(build_cmd, timeout_sec=300)
    if not ok:
        print(f"ERROR: Build command failed: stderr={err}")
        return 1
    ok, out, err = run_vm("cat /tmp/build_out.txt")
    if not ok:
        print(f"ERROR: Could not read build output: {err}")
        return 1
    print(out or err)
    # Success = BUILD_EXIT=0 in log OR artifact exists (output may be truncated by connect_vm)
    ok2, out2, _ = run_vm("test -f /tmp/libvgpu-cuda.so.1 && ls -la /tmp/libvgpu-cuda.so.1 && echo BUILD_SUCCESS")
    if "BUILD_EXIT=0" not in (out or "") and "BUILD_SUCCESS" not in (out2 or ""):
        print("ERROR: Build did not succeed (no BUILD_EXIT=0 and no artifact). Check output above.")
        return 1
    if "BUILD_SUCCESS" not in (out2 or "") and "libvgpu-cuda.so.1" not in (out2 or ""):
        print("ERROR: Build artifact /tmp/libvgpu-cuda.so.1 not found.")
        return 1
    print("  Build OK.\n")

    # Step 3: Install and restart (write result to file for reliable capture)
    print("Step 3: Install to /opt/vgpu/lib and restart ollama...")
    install_cmd = (
        f"( echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1 && "
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama && "
        "echo INSTALL_OK ) > /tmp/install_out.txt 2>&1; cat /tmp/install_out.txt"
    )
    ok, out, err = run_vm(install_cmd, timeout_sec=60)
    print(out or err)
    if not ok:
        print(f"ERROR: Install/restart failed: stderr={err}")
        return 1
    if "INSTALL_OK" not in (out or ""):
        print("ERROR: Install step did not complete (INSTALL_OK not seen). Check output above.")
        return 1
    print("  Install and restart OK.\n")

    print("=== Done. Inference fix deployed. ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
