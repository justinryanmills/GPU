#!/usr/bin/env python3
"""
Transfer only the modified cuda_executor.c to the mediator host, then rebuild
mediator_phase3. Does not transfer any other PHASE3 files.

Usage:
  1. Set MEDIATOR_HOST, MEDIATOR_USER, REMOTE_PHASE3 in this script (or env).
  2. Run: python3 transfer_cuda_executor_to_host.py
  3. On the mediator host, run the printed build and restart commands.

Requires: scp/ssh access to the host where mediator_phase3 runs (often dom0).
"""
import os
import subprocess
import sys

# --- Configure mediator host (where mediator_phase3 runs) ---
# Change these to match your setup. Not the VM — the host that has the GPU.
MEDIATOR_HOST = os.environ.get("MEDIATOR_HOST", "YOUR_MEDIATOR_HOST")
MEDIATOR_USER = os.environ.get("MEDIATOR_USER", "root")
REMOTE_PHASE3 = os.environ.get("REMOTE_PHASE3_HOST", "/root/phase3")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_FILE = os.path.join(SCRIPT_DIR, "src", "cuda_executor.c")
REMOTE_FILE = f"{REMOTE_PHASE3}/src/cuda_executor.c"


def main():
    if not os.path.isfile(LOCAL_FILE):
        print(f"Error: {LOCAL_FILE} not found.", file=sys.stderr)
        sys.exit(1)
    if "YOUR_MEDIATOR" in MEDIATOR_HOST or not MEDIATOR_HOST:
        print("Configure MEDIATOR_HOST (and optionally MEDIATOR_USER, REMOTE_PHASE3_HOST).", file=sys.stderr)
        print("  Example: export MEDIATOR_HOST=192.168.1.100", file=sys.stderr)
        print("  Or edit this script and set MEDIATOR_HOST at the top.", file=sys.stderr)
        sys.exit(1)

    dest = f"{MEDIATOR_USER}@{MEDIATOR_HOST}:{REMOTE_FILE}"
    print(f"Transferring single file to mediator host:")
    print(f"  {LOCAL_FILE}")
    print(f"  -> {dest}\n")

    r = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", LOCAL_FILE, dest],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        print(f"scp failed: {r.stderr or r.stdout}", file=sys.stderr)
        sys.exit(1)
    print("Transfer OK.\n")
    print("On the mediator host, run:")
    print(f"  cd {REMOTE_PHASE3}")
    print("  make")
    print("  # Then restart the mediator (e.g. stop old process and run ./mediator_phase3 again).")
    print("")
    print("Manual scp (if you prefer not to use this script):")
    print(f"  scp -o StrictHostKeyChecking=no {LOCAL_FILE} {MEDIATOR_USER}@{MEDIATOR_HOST}:{REMOTE_FILE}")


if __name__ == "__main__":
    main()
