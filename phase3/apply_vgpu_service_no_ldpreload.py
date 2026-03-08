#!/usr/bin/env python3
"""Apply ollama service override on VM: LD_LIBRARY_PATH with /opt/vgpu/lib first, NO LD_PRELOAD.
Run from phase3: python3 apply_vgpu_service_no_ldpreload.py
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD

CONF = """[Service]
ExecStart=
ExecStart=/usr/local/bin/ollama.bin serve
Environment=LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_NUM_GPU=1
Environment=OLLAMA_LLM_LIBRARY=cuda_v12
Environment=OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_LOAD_TIMEOUT=20m
"""

def run_vm(cmd, timeout=60):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""

def main():
    # Write conf to /tmp on VM via base64
    import base64
    b64 = base64.b64encode(CONF.encode()).decode()
    # Chunk to avoid long command lines
    chunk_size = 500
    ok, _, _ = run_vm("rm -f /tmp/vgpu_b64")
    for i in range(0, len(b64), chunk_size):
        chunk = b64[i:i+chunk_size].replace("'", "'\"'\"'")
        ok, o, e = run_vm(f"echo -n '{chunk}' >> /tmp/vgpu_b64")
        if not ok:
            print("Chunk write failed:", e or o)
            return 1
    ok, o, e = run_vm("base64 -d /tmp/vgpu_b64 > /tmp/vgpu.conf && cat /tmp/vgpu.conf")
    if not ok:
        print("Decode failed:", e or o)
        return 1
    # Install with sudo
    pw = VM_PASSWORD
    ok, o, e = run_vm(
        f"echo {pw} | sudo -S mkdir -p /etc/systemd/system/ollama.service.d && "
        f"echo {pw} | sudo -S cp /tmp/vgpu.conf /etc/systemd/system/ollama.service.d/vgpu.conf && "
        f"echo {pw} | sudo -S systemctl daemon-reload && "
        f"echo {pw} | sudo -S systemctl restart ollama"
    )
    if not ok:
        print("Install/restart failed:", e or o)
        return 1
    print("Service override applied (no LD_PRELOAD). Ollama restarted.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
