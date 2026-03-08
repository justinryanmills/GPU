#!/usr/bin/env python3
"""Transfer libvgpu_nvml.c to VM (chunked base64)."""
import os, sys, base64, subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import REMOTE_PHASE3

CHUNK = 40000
SRC = os.path.join(SCRIPT_DIR, "guest-shim", "libvgpu_nvml.c")

def run(cmd, timeout=120):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""

def main():
    with open(SRC, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    run("rm -f /tmp/nvml.b64")
    for i in range(0, len(b64), CHUNK):
        chunk = b64[i : i + CHUNK].replace("'", "'\"'\"'")
        ok, _, _ = run("echo -n '" + chunk + "' >> /tmp/nvml.b64")
        if not ok:
            print("Chunk failed")
            return 1
    ok, _, _ = run(
        "base64 -d /tmp/nvml.b64 > " + REMOTE_PHASE3 + "/guest-shim/libvgpu_nvml.c"
    )
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
