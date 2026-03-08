#!/bin/bash
set -e
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
[ -z "$OLLAMA_PID" ] && { echo "ollama not running"; exit 1; }
rm -f /tmp/ollama_full_strace.log
echo "Stracing PID $OLLAMA_PID (follow forks)..."
sudo strace -f -p $OLLAMA_PID -e trace=open,openat,read,pread,lseek -o /tmp/ollama_full_strace.log 2>/tmp/strace_outerr.txt &
STRACE_PID=$!
sleep 2
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' > /tmp/gen_out.json 2>&1 || true
sleep 4
sudo kill $STRACE_PID 2>/dev/null || true
sleep 1
echo "--- Strace log lines: $(wc -l < /tmp/ollama_full_strace.log)"
echo "--- Blob opens (sha256-74701a8c):"
grep "blobs/sha256-74701a8c" /tmp/ollama_full_strace.log || true
RUNNER_PID=$(grep "openat.*blobs/sha256-74701a8c" /tmp/ollama_full_strace.log | head -1 | sed -n 's/^\([0-9]*\).*/\1/p')
if [ -n "$RUNNER_PID" ]; then
  echo "--- Runner PID: $RUNNER_PID"
  echo "--- Reads from fd 7 returning 0:"
  grep "^$RUNNER_PID " /tmp/ollama_full_strace.log | grep "read(7," | grep "= 0)" || echo "(none)"
  echo "--- Total read(7) count for runner:"
  grep "^$RUNNER_PID " /tmp/ollama_full_strace.log | grep "read(7," | wc -l
  echo "--- Last 25 syscalls for runner:"
  grep "^$RUNNER_PID " /tmp/ollama_full_strace.log | tail -25
fi
echo "--- Generate response:"
head -1 /tmp/gen_out.json
