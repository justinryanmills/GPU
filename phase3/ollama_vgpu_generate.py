#!/usr/bin/env python3
"""
Patient client for Ollama on vGPU: waits for deployment (model load) without
giving up, shows a progress bar (estimated %) and only cancels on Ctrl+C.

Usage:
  python3 ollama_vgpu_generate.py [MODEL] [PROMPT]
  python3 ollama_vgpu_generate.py llama3.2:1b "Say hello."
  OLLAMA_HOST=http://vm-ip:11434 python3 ollama_vgpu_generate.py llama3.2:1b "Hi"

Default: model=llama3.2:1b, prompt="Say hello.", base URL from OLLAMA_HOST or
http://localhost:11434.

During deployment (first load on vGPU), the script shows a copying progress bar
(estimated percentage over typical load time; real completion can be sooner or later).
There is no time limit; the client waits until deployment completes. Only Ctrl+C cancels.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

# No time limit: wait until deployment completes or user cancels (Ctrl+C).
REQUEST_TIMEOUT = 7 * 24 * 3600  # 7 days (effectively no limit)
# Typical vGPU model load time (for progress bar estimate). Bar advances over this; can take longer.
ESTIMATED_LOAD_SECONDS = 40 * 60  # 40 minutes
PROGRESS_BAR_WIDTH = 30


def format_elapsed(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:1b"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Say hello."
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{base}/api/generate"

    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    result = {"done": False, "response": None, "error": None}

    def do_request():
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as f:
                result["response"] = f.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            result["error"] = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
        except urllib.error.URLError as e:
            result["error"] = str(e.reason) if getattr(e, "reason", None) else str(e)
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True

    thread = threading.Thread(target=do_request, daemon=True)
    start = time.monotonic()
    thread.start()

    print(f"Model: {model}")
    print("Copying model to GPU (vGPU). Ctrl+C to cancel.")
    last_pct = -1
    try:
        while not result["done"]:
            time.sleep(0.5)
            elapsed = time.monotonic() - start
            # Estimated progress 0..99% over ESTIMATED_LOAD_SECONDS (cap at 99% until response)
            pct = min(99, int(100.0 * elapsed / ESTIMATED_LOAD_SECONDS)) if ESTIMATED_LOAD_SECONDS > 0 else 0
            if pct != last_pct:
                last_pct = pct
                filled = int(PROGRESS_BAR_WIDTH * pct / 100)
                bar = "=" * filled + ">" * (1 if filled < PROGRESS_BAR_WIDTH else 0) + " " * (PROGRESS_BAR_WIDTH - filled - (1 if filled < PROGRESS_BAR_WIDTH else 0))
                sys.stdout.write(f"\r  [{bar}] {pct}% ({format_elapsed(elapsed)})   ")
                sys.stdout.flush()
            thread.join(timeout=0.5)
        # Response received: show 100%
        filled = PROGRESS_BAR_WIDTH
        sys.stdout.write(f"\r  [{'=' * filled}] 100% ({format_elapsed(time.monotonic() - start)})   \n")
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        print("Cancelled by user (Ctrl+C).")
        sys.exit(130)

    if result["error"]:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result["response"])
    if "response" in data:
        print("\nResponse:", data["response"].strip())
    else:
        print(json.dumps(data, indent=2)[:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
