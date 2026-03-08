#!/usr/bin/env python3
"""
Deploy phase3 guest shims to test-3 VM and get Ollama into GPU mode.

Uses SCP for all file transfers (no chunked/base64) to avoid corruption.
Reads VM target from vm_config.py (test-3@10.25.33.11).

Steps:
  1. SCP phase3 tree to VM (so 'make guest' works).
  2. SSH: make guest in phase3.
  3. Install built shims to /opt/vgpu/lib with correct symlinks.
  4. Ensure Ollama systemd override (vgpu.conf) has LD_LIBRARY_PATH and OLLAMA_NUM_GPU.
  5. Restart Ollama service.
"""
import os
import sys
import subprocess
import shutil
import shlex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3, REMOTE_HOME

USE_SSHPASS = shutil.which("sshpass") is not None


def run_ssh(cmd, timeout_sec=300):
    """Run command on VM via ssh. Uses sshpass if available, else connect_vm.py."""
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


def scp_file(local_path, remote_path, recursive=False):
    """Copy file or directory to VM via scp (sshpass or pexpect). Returns True on success."""
    dest = f"{VM_USER}@{VM_HOST}:{remote_path}"
    if USE_SSHPASS:
        full_cmd = [
            "sshpass", "-p", VM_PASSWORD,
            "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
        ]
        if recursive:
            full_cmd.append("-r")
        full_cmd.extend([local_path, dest])
        r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=600)
        return r.returncode == 0
    try:
        import pexpect
        scp_cmd = "scp -o StrictHostKeyChecking=no -o ConnectTimeout=15"
        if recursive:
            scp_cmd += " -r"
        scp_cmd += " " + shlex.quote(local_path) + " " + shlex.quote(dest)
        # Long timeout for full tree (600+ files, ~5MB)
        c = pexpect.spawn(scp_cmd, timeout=1200, encoding="utf-8")
        idx = c.expect(["password:", "Password:", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        if idx in (0, 1):
            c.sendline(VM_PASSWORD)
        idx2 = c.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=1200)
        c.close()
        # Success only if we got EOF (transfer finished), not TIMEOUT
        return idx2 == 0 and (c.exitstatus is None or c.exitstatus == 0)
    except Exception:
        return False


def main():
    print("=== Deploy to test-3 (Ollama GPU mode) ===\n")
    print(f"Target: {VM_USER}@{VM_HOST}  phase3={REMOTE_PHASE3}\n")

    # 1) Deploy phase3 tree via SCP (no chunked transfer)
    local_phase3 = SCRIPT_DIR
    if not os.path.isdir(local_phase3):
        print(f"ERROR: Local phase3 not found: {local_phase3}")
        return 1
    print("Step 1: Copying phase3 tree to VM (SCP)...")
    if not scp_file(local_phase3, REMOTE_HOME + "/", recursive=True):
        print("ERROR: SCP deploy failed.")
        return 1
    print("  Done.\n")

    # 2) Build guest shims on VM
    print("Step 2: Building guest shims on VM (make guest)...")
    ok, out, err = run_ssh(f"cd {REMOTE_PHASE3} && make guest 2>&1", timeout_sec=300)
    print(out or err)
    if not ok:
        print("ERROR: make guest failed.")
        return 1
    # Verify at least main lib exists
    ok2, out2, _ = run_ssh(f"test -f {REMOTE_PHASE3}/guest-shim/libvgpu-cuda.so.1 && ls -la {REMOTE_PHASE3}/guest-shim/libvgpu-*.so* 2>&1")
    if not ok2:
        print("ERROR: Guest shim libraries not found after build.")
        return 1
    print("  Build OK.\n")

    # 3) Install to /opt/vgpu/lib with symlinks for loader
    print("Step 3: Installing shims to /opt/vgpu/lib...")
    install_cmds = [
        f"echo {VM_PASSWORD} | sudo -S mkdir -p /opt/vgpu/lib",
        f"test -f {REMOTE_PHASE3}/guest-shim/libvgpu-exec-inject.so && echo {VM_PASSWORD} | sudo -S cp {REMOTE_PHASE3}/guest-shim/libvgpu-exec-inject.so /opt/vgpu/lib/ || true",
        f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_PHASE3}/guest-shim/libvgpu-cuda.so.1 /opt/vgpu/lib/",
        f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_PHASE3}/guest-shim/libvgpu-cudart.so /opt/vgpu/lib/",
        f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_PHASE3}/guest-shim/libvgpu-nvml.so /opt/vgpu/lib/",
        f"echo {VM_PASSWORD} | sudo -S ln -sf /opt/vgpu/lib/libvgpu-cuda.so.1 /opt/vgpu/lib/libcuda.so.1",
        f"echo {VM_PASSWORD} | sudo -S ln -sf /opt/vgpu/lib/libvgpu-cudart.so /opt/vgpu/lib/libcudart.so.12",
        f"echo {VM_PASSWORD} | sudo -S ln -sf /opt/vgpu/lib/libvgpu-nvml.so /opt/vgpu/lib/libnvidia-ml.so.1",
    ]
    # CUBLAS stubs if built
    for name, link in [("libvgpu-cublas.so.12", "libcublas.so.12"), ("libvgpu-cublasLt.so.12", "libcublasLt.so.12")]:
        install_cmds.append(
            f"test -f {REMOTE_PHASE3}/guest-shim/{name} && "
            f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_PHASE3}/guest-shim/{name} /opt/vgpu/lib/ && "
            f"echo {VM_PASSWORD} | sudo -S ln -sf /opt/vgpu/lib/{name} /opt/vgpu/lib/{link} || true"
        )
    full_install = " && ".join(install_cmds)
    ok, out, err = run_ssh(full_install, timeout_sec=60)
    if not ok:
        print("ERROR: Install failed:", err or out)
        return 1
    print("  Done.\n")

    # 4) Ollama systemd override: run ollama.bin directly with full LD_PRELOAD (avoids SEGV when bash runs the script)
    # Use custom ollama.service drop-in so ExecStart=ollama.bin serve and full LD_PRELOAD are set.
    override_dir = "/etc/systemd/system/ollama.service.d"
    vgpu_conf_local = os.path.join(SCRIPT_DIR, "ollama.service.d_vgpu.conf")
    print("Step 4: Installing ollama.service.d/vgpu.conf (ExecStart=ollama.bin, full LD_PRELOAD)...")
    if os.path.isfile(vgpu_conf_local):
        if not scp_file(vgpu_conf_local, REMOTE_HOME + "/vgpu.conf", recursive=False):
            print("WARNING: SCP vgpu.conf failed.")
        else:
            ok, out, err = run_ssh(
                f"echo {VM_PASSWORD} | sudo -S mkdir -p {override_dir} && "
                f"echo {VM_PASSWORD} | sudo -S mv {REMOTE_HOME}/vgpu.conf {override_dir}/vgpu.conf && "
                f"echo {VM_PASSWORD} | sudo -S systemctl daemon-reload",
                timeout_sec=30,
            )
            if not ok:
                print("WARNING: vgpu.conf install failed:", err or out)
            else:
                print("  Done.\n")
    else:
        print("  (ollama.service.d_vgpu.conf not found, skipping drop-in)\n")

    # 5) Restart Ollama
    print("Step 5: Restarting Ollama...")
    ok, out, err = run_ssh(
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama.service 2>&1",
        timeout_sec=30,
    )
    print(out or err)
    if not ok:
        print("WARNING: Restart may have failed.")
    else:
        print("  Restarted.\n")

    print("=== Deploy complete ===\n")
    print("Verify GPU mode:")
    print("  python3 connect_vm.py \"journalctl -u ollama -n 30 --no-pager | grep -E 'library=|total_vram|inference compute'\"")
    print("  python3 connect_vm.py \"ollama run llama3.2:1b 'Hi'\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
