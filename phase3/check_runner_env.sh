#!/bin/bash
OUT=/tmp/runner_env_capture.txt
echo "Polling for runner (25s)..." > "$OUT"
i=0
while [ $i -lt 25 ]; do
  for pid in $(pgrep -f "ollama.bin"); do
    [ -r /proc/$pid/cmdline ] || continue
    cmdline=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    echo "$cmdline" | grep -q " runner " || continue
    echo "Found runner pid=$pid" >> "$OUT"
    sudo cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep -E "LD_PRELOAD|LD_LIBRARY" >> "$OUT"
    cat "$OUT"
    exit 0
  done
  sleep 1
  i=$((i+1))
done
echo "No runner found." >> "$OUT"
cat "$OUT"
