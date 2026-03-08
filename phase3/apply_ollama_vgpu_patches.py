#!/usr/bin/env python3
"""
Apply vGPU patches to Ollama source so GPU discovery and inference work on test-3.

Run from the repo that contains ollama-src (e.g. phase3), or set OLLAMA_SRC:
  cd /path/to/gpu/phase3 && python3 apply_ollama_vgpu_patches.py
  OLLAMA_SRC=/path/to/ollama python3 apply_ollama_vgpu_patches.py

Patches applied:
  1. ml/device.go: NeedsInitValidation() returns false for CUDA (only ROCm).
  2. llm/server.go: Prepend /opt/vgpu/lib to runner LD_LIBRARY_PATH and OLLAMA_LIBRARY_PATH;
     remove LD_PRELOAD from runner env so real dlopen loads libggml-cuda and shims.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_SRC = os.environ.get("OLLAMA_SRC", os.path.join(SCRIPT_DIR, "ollama-src"))
DEVICE_GO = os.path.join(OLLAMA_SRC, "ml", "device.go")
SERVER_GO = os.path.join(OLLAMA_SRC, "llm", "server.go")


def patch_device_go():
    path = DEVICE_GO
    if not os.path.isfile(path):
        print(f"Skip (not found): {path}")
        return True
    with open(path, "r") as f:
        content = f.read()
    old = 'return d.Library == "ROCm" || d.Library == "CUDA"'
    new = 'return d.Library == "ROCm"'
    if new in content and old not in content:
        print("ml/device.go: already patched (NeedsInitValidation)")
        return True
    if old not in content:
        print("ml/device.go: target line not found")
        return False
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("ml/device.go: patched NeedsInitValidation (CUDA skip)")
    return True


def patch_server_go():
    path = SERVER_GO
    if not os.path.isfile(path):
        print(f"Skip (not found): {path}")
        return True
    with open(path, "r") as f:
        content = f.read()

    # 1) Prepend /opt/vgpu/lib to libraryPaths
    needle1 = "\tlibraryPaths := append([]string{}, gpuLibs...)\n\tif libraryPath"
    if "/opt/vgpu/lib" in content and "libraryPaths[0] != \"/opt/vgpu/lib\"" in content:
        print("llm/server.go: prepend /opt/vgpu/lib already present")
    elif needle1 not in content:
        print("llm/server.go: prepend block target not found")
    else:
        insert = (
            "\tlibraryPaths := append([]string{}, gpuLibs...)\n"
            "\t// vGPU guest: ensure runner loads shims from /opt/vgpu/lib first\n"
            '\tif len(libraryPaths) == 0 || libraryPaths[0] != "/opt/vgpu/lib" {\n'
            '\t\tlibraryPaths = append([]string{"/opt/vgpu/lib"}, libraryPaths...)\n'
            "\t}\n"
            "\tif libraryPath"
        )
        content = content.replace(needle1, insert, 1)
        print("llm/server.go: added prepend /opt/vgpu/lib to libraryPaths")

    # 2) OLLAMA_LIBRARY_PATH: prepend /opt/vgpu/lib
    old_ollama = 'cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + strings.Join(gpuLibs, string(filepath.ListSeparator))'
    new_ollama = 'cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + "/opt/vgpu/lib:" + strings.Join(gpuLibs, string(filepath.ListSeparator))'
    if new_ollama in content:
        print("llm/server.go: OLLAMA_LIBRARY_PATH prepend already present")
    elif old_ollama in content:
        content = content.replace(old_ollama, new_ollama, 1)
        print("llm/server.go: prepend /opt/vgpu/lib to OLLAMA_LIBRARY_PATH (in loop)")
    old_ollama_append = 'cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(gpuLibs, string(filepath.ListSeparator)))'
    new_ollama_append = 'cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+"/opt/vgpu/lib:"+strings.Join(gpuLibs, string(filepath.ListSeparator)))'
    if new_ollama_append in content:
        pass
    elif old_ollama_append in content:
        content = content.replace(old_ollama_append, new_ollama_append, 1)
        print("llm/server.go: prepend /opt/vgpu/lib to OLLAMA_LIBRARY_PATH (append)")

    # 3) Remove LD_PRELOAD from runner env
    needle3 = "\tfor k, done := range extraEnvsDone {\n\t\tif !done {\n\t\t\tcmd.Env = append(cmd.Env, k+\"=\"+extraEnvs[k])\n\t\t}\n\t}\n\n\tslog.Info(\"starting runner\""
    if "LD_PRELOAD" in content and "filtered := cmd.Env[:0]" in content:
        print("llm/server.go: LD_PRELOAD filter already present")
    elif needle3 not in content:
        print("llm/server.go: LD_PRELOAD filter block target not found")
    else:
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
        content = content.replace(needle3, insert3, 1)
        print("llm/server.go: added LD_PRELOAD filter for runner env")

    with open(path, "w") as f:
        f.write(content)
    return True


def main():
    if not os.path.isdir(OLLAMA_SRC):
        print(f"Error: OLLAMA_SRC not a directory: {OLLAMA_SRC}")
        return 1
    ok = patch_device_go() and patch_server_go()
    print("Done." if ok else "One or more patches failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
