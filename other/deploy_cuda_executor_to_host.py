#!/usr/bin/env python3
"""Deploy cuda_executor.c to mediator host via connect_host.py."""
import os
import sys
import base64
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import MEDIATOR_HOST, MEDIATOR_USER, MEDIATOR_PASSWORD

REMOTE_PHASE3 = os.environ.get("REMOTE_PHASE3", "/root/phase3")
CHUNK_SIZE = 20000  # smaller to avoid command-line length limits


def run_host(cmd, timeout=90):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_host.py"), cmd],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""


def main():
    local_path = os.path.join(SCRIPT_DIR, "src", "cuda_executor.c")
    if not os.path.isfile(local_path):
        print("Error: src/cuda_executor.c not found", file=sys.stderr)
        return 1
    with open(local_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, _, _ = run_host("rm -f /tmp/ce.b64")
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/ce.b64"
        ok, out, err = run_host(cmd)
        if not ok:
            print(f"Chunk write failed: {err or out}", file=sys.stderr)
            return 1
    ok, out, err = run_host(
        f"base64 -d /tmp/ce.b64 > {REMOTE_PHASE3}/src/cuda_executor.c && rm -f /tmp/ce.b64 && wc -c {REMOTE_PHASE3}/src/cuda_executor.c"
    )
    if not ok:
        print(f"Decode/write failed: {err or out}", file=sys.stderr)
        return 1
    print("Deployed cuda_executor.c to host:", out.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
