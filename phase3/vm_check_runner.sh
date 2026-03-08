#!/bin/bash
set -e
OUT=/tmp/runner_capture.txt
echo "Restarting ollama..." > "$OUT"
systemctl restart ollama.service
echo "Polling for runner (5s)..." >> "$OUT"
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
  for p in $(pgrep -f "ollama.bin" 2>/dev/null); do
    [ -r /proc/$p/cmdline ] || continue
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    echo "$cmd" | grep -q " runner " || continue
    echo "RUNNER_PID=$p" >> "$OUT"
    cat /proc/$p/environ 2>/dev/null | tr '\0' '\n' | grep -E "LD_PRELOAD|LD_LIBRARY" >> "$OUT"
    echo "DONE" >> "$OUT"
    cat "$OUT"
    exit 0
  done
  sleep 0.2
done
echo "NO_RUNNER_FOUND" >> "$OUT"
cat "$OUT"
