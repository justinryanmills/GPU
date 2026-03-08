#!/usr/bin/env python3
"""
Deploy a Hopper-capable libggml-cuda.so to the test-3 VM.

Use this after building libggml-cuda.so with CMAKE_CUDA_ARCHITECTURES=90
on a machine that has the CUDA toolkit (see BUILD_LIBGGML_CUDA_HOPPER.md).

Usage:
  python3 deploy_libggml_cuda_hopper.py [path/to/libggml-cuda.so]

If no path is given, looks for ./libggml-cuda.so in the script directory.

Steps:
  1. SCP the file to the VM /tmp/libggml-cuda.so (long timeout for large file).
  2. SSH: backup existing library, install new one under /usr/local/lib/ollama/cuda_v12/.
  3. Restart ollama.service.
"""
import os
import sys
import subprocess
import shutil
import shlex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD

USE_SSHPASS = shutil.which("sshpass") is not None

OLLAMA_CUDA_DIR = "/usr/local/lib/ollama/cuda_v12"
LIB_NAME = "libggml-cuda.so"
# Timeout for SCP of a large shared library (e.g. 1–2 GB)
SCP_TIMEOUT = 3600


def run_ssh(cmd, timeout_sec=120):
    """Run command on VM via ssh."""
    if USE_SSHPASS:
        full_cmd = [
            "sshpass", "-p", VM_PASSWORD,
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            f"{VM_USER}@{VM_HOST}",
            cmd,
        ]
        r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout_sec, cwd=SCRIPT_DIR)
        return r.returncode == 0, r.stdout or "", r.stderr or ""
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""


def scp_to_vm(local_path, remote_path, timeout_sec=SCP_TIMEOUT):
    """Copy file to VM via scp. Returns True on success."""
    dest = f"{VM_USER}@{VM_HOST}:{remote_path}"
    if USE_SSHPASS:
        full_cmd = [
            "sshpass", "-p", VM_PASSWORD,
            "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            local_path, dest,
        ]
        r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout_sec)
        return r.returncode == 0
    try:
        import pexpect
        scp_cmd = "scp -o StrictHostKeyChecking=no -o ConnectTimeout=15 " + shlex.quote(local_path) + " " + shlex.quote(dest)
        c = pexpect.spawn(scp_cmd, timeout=timeout_sec, encoding="utf-8")
        idx = c.expect(["password:", "Password:", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        if idx in (0, 1):
            c.sendline(VM_PASSWORD)
        idx2 = c.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=timeout_sec)
        c.close()
        return idx2 == 0 and (c.exitstatus is None or c.exitstatus == 0)
    except Exception:
        return False


def main():
    local_lib = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "libggml-cuda.so"))
    if not os.path.isfile(local_lib):
        print(f"Error: File not found: {local_lib}")
        print("Usage: python3 deploy_libggml_cuda_hopper.py [path/to/libggml-cuda.so]")
        return 1

    size_mb = os.path.getsize(local_lib) / (1024 * 1024)
    print(f"=== Deploy libggml-cuda.so (Hopper) to test-3 ===")
    print(f"Local: {local_lib} ({size_mb:.1f} MiB)")
    print(f"Target: {VM_USER}@{VM_HOST} -> {OLLAMA_CUDA_DIR}/")
    print()

    # 1) SCP to VM /tmp
    print("Step 1: Copying to VM /tmp/libggml-cuda.so (this may take several minutes)...")
    if not scp_to_vm(local_lib, "/tmp/libggml-cuda.so"):
        print("ERROR: SCP failed.")
        return 1
    print("  Done.\n")

    # 2) On VM: backup (best-effort), install, restart
    install_cmd = (
        f"sudo cp {OLLAMA_CUDA_DIR}/{LIB_NAME} {OLLAMA_CUDA_DIR}/{LIB_NAME}.bak 2>/dev/null || true; "
        f"sudo cp /tmp/libggml-cuda.so {OLLAMA_CUDA_DIR}/{LIB_NAME} && "
        f"sudo systemctl restart ollama && "
        f"echo OK"
    )
    print("Step 2: Installing and restarting ollama on VM...")
    ok, out, err = run_ssh(install_cmd, timeout_sec=180)
    print(out or err)
    if not ok or "OK" not in (out or ""):
        print("ERROR: Install or restart failed.")
        return 1
    print("  Done.\n")

    print("Deploy complete. Check: sudo journalctl -u ollama -f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
