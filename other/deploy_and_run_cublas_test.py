#!/usr/bin/env python3
"""Transfer libvgpu_cublas, build, install, run test on VM."""
import os
import sys
import base64
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3


def run_vm(cmd, timeout_sec=300):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def main():
    # Deploy libvgpu_cublas
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "transfer_libvgpu_cublas.py")],
                   cwd=SCRIPT_DIR, timeout=120, check=True)

    # Push test source
    test_src = os.path.join(SCRIPT_DIR, "guest-shim", "test_cublas_vm.c")
    with open(test_src, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, _, _ = run_vm("rm -f /tmp/test_b64.b64")
    for i in range(0, len(b64), 40000):
        chunk = b64[i : i + 40000]
        ok, _, _ = run_vm("echo -n '" + escape(chunk) + "' >> /tmp/test_b64.b64")
        if not ok:
            print("Failed to push test source")
            return 1

    ok, out, _ = run_vm(
        f"base64 -d /tmp/test_b64.b64 > {REMOTE_PHASE3}/guest-shim/test_cublas_vm.c && "
        f"gcc -O2 -o {REMOTE_PHASE3}/guest-shim/test_cublas_vm "
        f"{REMOTE_PHASE3}/guest-shim/test_cublas_vm.c -ldl 2>&1"
    )
    print(out)
    if "error:" in out or not ok:
        print("Build failed")
        return 1

    # Run test
    ok, out, _ = run_vm(
        f"cd {REMOTE_PHASE3} && "
        "mkdir -p /tmp/vgpu-cublas-test && "
        "ln -sf /opt/vgpu/lib/libvgpu-cublas.so.12 /tmp/vgpu-cublas-test/libcublas.so.12 && "
        f"LD_LIBRARY_PATH=/tmp/vgpu-cublas-test:/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:$LD_LIBRARY_PATH "
        f"./guest-shim/test_cublas_vm 2>&1",
        timeout_sec=60
    )
    print("--- TEST OUTPUT ---")
    print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
