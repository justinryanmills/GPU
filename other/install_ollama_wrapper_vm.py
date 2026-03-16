#!/usr/bin/env python3
"""Install ollama wrapper on VM (LD_LIBRARY_PATH, no LD_PRELOAD)."""
import os
import sys
import base64
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_PASSWORD

CHUNK_SIZE = 40000

WRAPPER = b"""#!/bin/bash
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama"
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_NUM_GPU="1"
exec /usr/local/bin/ollama.real "$@"
"""

def run_vm(cmd, timeout_sec=120):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""

def main():
    # Ensure ollama.real
    ok, _, _ = run_vm(
        "test -f /usr/local/bin/ollama.real || (echo " + repr(VM_PASSWORD) + " | sudo -S cp -a /usr/local/bin/ollama /usr/local/bin/ollama.real)"
    )
    if not ok:
        print("Failed to ensure ollama.real exists")
        return 1

    b64 = base64.b64encode(WRAPPER).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, _, _ = run_vm("rm -f /tmp/combined.b64")
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
        ok, out, err = run_vm(cmd)
        if not ok:
            print("Chunk write failed:", err or out)
            return 1

    install_cmd = (
        "base64 -d /tmp/combined.b64 > /tmp/ollama_wrapper.sh && "
        "echo " + repr(VM_PASSWORD) + " | sudo -S cp /tmp/ollama_wrapper.sh /usr/local/bin/ollama && "
        "echo " + repr(VM_PASSWORD) + " | sudo -S chmod +x /usr/local/bin/ollama && "
        "head -2 /usr/local/bin/ollama"
    )
    ok, out, err = run_vm(install_cmd)
    print(out or err)
    if not ok:
        print("Install failed")
        return 1
    if "#!/bin/bash" not in (out or ""):
        print("Wrapper content not found in ollama")
        return 1
    print("Wrapper installed. Restarting ollama.")
    ok, _, _ = run_vm("echo " + repr(VM_PASSWORD) + " | sudo -S systemctl start ollama")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
