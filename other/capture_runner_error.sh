#!/bin/bash
# Run on VM to capture ollama runner CUDA error. Usage: ./capture_runner_error.sh or connect_vm.py 'bash -s' < capture_runner_error.sh
set -e
echo "=== Capturing runner error ==="

# Trigger generate in background
(
  sleep 1
  curl -s -X POST http://127.0.0.1:11434/api/generate \
    -H 'Content-Type: application/json' \
    -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' \
    > /tmp/generate_output.json 2>&1
) &
CURL_PID=$!

# Tail journalctl
sleep 8
sudo journalctl -u ollama -n 200 --no-pager 2>/dev/null \
  | grep -iE "CUDA error|error:|exit status|Load failed|runner|ggml" \
  | tail -30

echo ""
echo "=== Generate response ==="
cat /tmp/generate_output.json 2>/dev/null | head -3
echo ""
echo "=== Last 15 ollama log lines ==="
sudo journalctl -u ollama -n 15 --no-pager 2>/dev/null

wait $CURL_PID 2>/dev/null || true
echo "=== Done ==="
