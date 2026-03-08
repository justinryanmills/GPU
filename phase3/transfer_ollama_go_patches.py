#!/usr/bin/env python3
"""
Apply vGPU patches in memory, transfer patched device.go and server.go to the VM,
then build ollama.bin and install. Uses chunked base64 transfer (no full SCP of phase3).

Usage: python3 transfer_ollama_go_patches.py [REMOTE_OLLAMA]
  REMOTE_OLLAMA defaults to /home/<VM_USER>/ollama (from vm_config).
"""
import os
import sys
import base64
import hashlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_HOME

CHUNK_SIZE = 40000
OLLAMA_SRC = os.path.join(SCRIPT_DIR, "ollama-src")
REMOTE_OLLAMA = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(REMOTE_HOME, "ollama")).rstrip("/")


def run_vm(cmd, timeout_sec=180):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec, cwd=SCRIPT_DIR,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def patch_device_go(content: str) -> str:
    old = 'return d.Library == "ROCm" || d.Library == "CUDA"'
    new = 'return d.Library == "ROCm"'
    if new in content and old not in content:
        return content
    if old not in content:
        raise SystemExit("device.go: target line not found")
    return content.replace(old, new, 1)


def patch_server_go(content: str) -> str:
    # 1) Prepend /opt/vgpu/lib to libraryPaths
    needle1 = "\tlibraryPaths := append([]string{}, gpuLibs...)\n\tif libraryPath"
    insert1 = (
        "\tlibraryPaths := append([]string{}, gpuLibs...)\n"
        "\t// vGPU guest: ensure runner loads shims from /opt/vgpu/lib first\n"
        '\tif len(libraryPaths) == 0 || libraryPaths[0] != "/opt/vgpu/lib" {\n'
        '\t\tlibraryPaths = append([]string{"/opt/vgpu/lib"}, libraryPaths...)\n'
        "\t}\n"
        "\tif libraryPath"
    )
    if insert1 not in content and needle1 in content:
        content = content.replace(needle1, insert1, 1)

    # 2) OLLAMA_LIBRARY_PATH prepend
    old_ollama = 'cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + strings.Join(gpuLibs, string(filepath.ListSeparator))'
    new_ollama = 'cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + "/opt/vgpu/lib:" + strings.Join(gpuLibs, string(filepath.ListSeparator))'
    if old_ollama in content and new_ollama not in content:
        content = content.replace(old_ollama, new_ollama, 1)
    old_append = 'cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(gpuLibs, string(filepath.ListSeparator)))'
    new_append = 'cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+"/opt/vgpu/lib:"+strings.Join(gpuLibs, string(filepath.ListSeparator)))'
    if old_append in content and new_append not in content:
        content = content.replace(old_append, new_append, 1)

    # 3) Remove LD_PRELOAD from runner env
    needle3 = (
        "\tfor k, done := range extraEnvsDone {\n\t\tif !done {\n\t\t\tcmd.Env = append(cmd.Env, k+\"=\"+extraEnvs[k])\n\t\t}\n\t}\n\n"
        "\tslog.Info(\"starting runner\""
    )
    insert3 = (
        "\tfor k, done := range extraEnvsDone {\n\t\tif !done {\n\t\t\tcmd.Env = append(cmd.Env, k+\"=\"+extraEnvs[k])\n\t\t}\n\t}\n\n"
        "\t// vGPU guest: runner must not inherit LD_PRELOAD (need real dlopen to load libggml-cuda)\n"
        "\tfiltered := cmd.Env[:0]\n"
        "\tfor _, e := range cmd.Env {\n"
        '\t\tif !strings.HasPrefix(e, "LD_PRELOAD=") {\n'
        "\t\t\tfiltered = append(filtered, e)\n"
        "\t\t}\n"
        "\t}\n"
        "\tcmd.Env = filtered\n\n"
        "\tslog.Info(\"starting runner\""
    )
    if insert3 not in content and needle3 in content:
        content = content.replace(needle3, insert3, 1)
    elif "filtered := cmd.Env[:0]" not in content and needle3 not in content:
        raise SystemExit("server.go: LD_PRELOAD filter block target not found")

    return content


def patch_discover_runner_go(content: str) -> str:
    # Put GPU lib dir (cuda_v12) before LibOllamaPath so backend loader finds CUDA before CPU
    old = "\t\t\tdirs = []string{ml.LibOllamaPath, dir}"
    new = "\t\t\tdirs = []string{dir, ml.LibOllamaPath}"
    if new in content and old not in content:
        return content
    if old not in content:
        raise SystemExit("discover/runner.go: target line not found")
    return content.replace(old, new, 1)


def transfer_file_to_vm(data: bytes, remote_path: str, label: str) -> bool:
    local_sha = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, _, _ = run_vm(f"rm -f /tmp/combined.b64")
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
        ok, out, err = run_vm(cmd)
        if not ok:
            print(f"{label}: chunk write failed:", err or out)
            return False
    ok, out, err = run_vm(f"base64 -d /tmp/combined.b64 > {remote_path} && wc -c {remote_path}")
    if not ok:
        print(f"{label}: decode failed:", err or out)
        return False
    ok, out, err = run_vm(
        "python3 - << 'PYEOF'\n"
        "import hashlib\n"
        f"p = '{remote_path}'\n"
        "with open(p, 'rb') as f:\n"
        "    d = f.read()\n"
        "print('REMOTE_SHA256=' + hashlib.sha256(d).hexdigest())\n"
        "PYEOF"
    )
    if not ok:
        return False
    remote_sha = None
    for line in (out or "").splitlines():
        if line.strip().startswith("REMOTE_SHA256="):
            remote_sha = line.split("=", 1)[1].strip()
    if remote_sha != local_sha:
        print(f"{label}: SHA256 mismatch")
        return False
    print(f"{label}: transferred and verified")
    return True


def main():
    device_go = os.path.join(OLLAMA_SRC, "ml", "device.go")
    server_go = os.path.join(OLLAMA_SRC, "llm", "server.go")
    discover_runner_go = os.path.join(OLLAMA_SRC, "discover", "runner.go")
    if not os.path.isfile(device_go) or not os.path.isfile(server_go):
        print("Error: ollama-src/ml/device.go or llm/server.go not found")
        return 1
    if not os.path.isfile(discover_runner_go):
        print("Error: ollama-src/discover/runner.go not found")
        return 1

    with open(device_go, "r") as f:
        device_content = f.read()
    with open(server_go, "r") as f:
        server_content = f.read()
    with open(discover_runner_go, "r") as f:
        discover_content = f.read()

    device_content = patch_device_go(device_content)
    server_content = patch_server_go(server_content)
    discover_content = patch_discover_runner_go(discover_content)
    print("Patches applied (device.go, server.go, discover/runner.go: GPU lib dir first).")

    if not transfer_file_to_vm(device_content.encode("utf-8"), "/tmp/device_patched.go", "device.go"):
        return 1
    if not transfer_file_to_vm(server_content.encode("utf-8"), "/tmp/server_patched.go", "server.go"):
        return 1
    if not transfer_file_to_vm(discover_content.encode("utf-8"), "/tmp/runner_patched.go", "discover/runner.go"):
        return 1

    # Copy to ollama tree and build (use Go 1.26 if available; VM may have 1.18 in PATH)
    copy_build = (
        f"cp /tmp/device_patched.go {REMOTE_OLLAMA}/ml/device.go && "
        f"cp /tmp/server_patched.go {REMOTE_OLLAMA}/llm/server.go && "
        f"cp /tmp/runner_patched.go {REMOTE_OLLAMA}/discover/runner.go && "
        f"cd {REMOTE_OLLAMA} && ( /usr/local/go/bin/go version 2>/dev/null && /usr/local/go/bin/go build -o ollama.bin . || go build -o ollama.bin . ) 2>&1; echo BUILD_EXIT=$?"
    )
    ok, out, err = run_vm(copy_build, timeout_sec=300)
    print(out)
    if not ok:
        print("Copy or build failed:", err or out)
        return 1
    if "BUILD_EXIT=0" not in (out or ""):
        print("Build did not succeed. Output above.")
        return 1

    install_cmd = (
        f"echo {VM_PASSWORD} | sudo -S systemctl stop ollama && "
        f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_OLLAMA}/ollama.bin /usr/local/bin/ollama.bin && "
        f"echo {VM_PASSWORD} | sudo -S systemctl start ollama"
    )
    ok, out, err = run_vm(install_cmd)
    print(out or err)
    if not ok:
        print("Install or restart failed")
        return 1
    print("Installed ollama.bin and restarted ollama service.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
