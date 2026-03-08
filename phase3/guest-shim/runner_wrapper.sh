#!/bin/bash
set -e

export LD_LIBRARY_PATH="/usr/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/lib64/libvgpu-cuda.so:${LD_PRELOAD:-}"

if command -v python3 >/dev/null 2>&1; then
    python3 << 'PYEOF'
import ctypes
import sys
try:
    lib = ctypes.CDLL("libcuda.so.1")
    print("[runner-wrapper] Pre-loaded libcuda.so.1", file=sys.stderr)
    try:
        cuInit = getattr(lib, "cuInit", None)
        if cuInit:
            result = cuInit(0)
            if result == 0:
                print("[runner-wrapper] Pre-initialized CUDA (cuInit succeeded)", file=sys.stderr)
            else:
                print(f"[runner-wrapper] cuInit returned {result}", file=sys.stderr)
    except Exception as e:
        print(f"[runner-wrapper] cuInit call failed: {e}", file=sys.stderr)
except Exception as e:
    print(f"[runner-wrapper] Failed to pre-load library: {e}", file=sys.stderr)
PYEOF
fi

exec "$@"
