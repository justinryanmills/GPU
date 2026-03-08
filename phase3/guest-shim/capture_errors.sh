#!/bin/bash
set -e

DURATION=${1:-60}  # Default 60 seconds (2x discovery timeout)
CAPTURE_DIR="/tmp/ollama_error_capture_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CAPTURE_DIR"

echo "=========================================="
echo "Ollama Error Capture Script"
echo "=========================================="
echo "Duration: ${DURATION} seconds"
echo "Capture directory: $CAPTURE_DIR"
echo ""

echo "[1/6] Cleaning old log files..."
rm -f /tmp/ollama_errors_full.log
rm -f /tmp/ollama_errors_filtered.log
rm -f /tmp/ollama_stderr.log
rm -f /tmp/ollama_strace.log

OLLAMA_PID=$(pgrep -f "ollama serve" || echo "")
if [ -z "$OLLAMA_PID" ]; then
    echo "[2/6] Starting Ollama service..."
    sudo systemctl restart ollama
    sleep 2
    OLLAMA_PID=$(pgrep -f "ollama serve" || echo "")
    if [ -z "$OLLAMA_PID" ]; then
        echo "ERROR: Could not find Ollama process"
        exit 1
    fi
else
    echo "[2/6] Ollama already running (PID: $OLLAMA_PID)"
fi

echo "[3/6] Starting strace capture in background..."
sudo strace -p "$OLLAMA_PID" -s 2000 -e trace=write,writev -o /tmp/ollama_strace.log 2>&1 &
STRACE_PID=$!

echo "[4/6] Capturing errors for ${DURATION} seconds..."
echo "      (Monitoring write interceptor logs, stderr, and strace)"

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))

while [ $(date +%s) -lt $END_TIME ]; do
    sleep 1
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "WARNING: Ollama process died during capture"
        break
    fi
done

echo "[5/6] Stopping strace..."
sudo kill $STRACE_PID 2>/dev/null || true
wait $STRACE_PID 2>/dev/null || true

echo "[6/6] Collecting captured data..."

if [ -f /tmp/ollama_errors_full.log ]; then
    cp /tmp/ollama_errors_full.log "$CAPTURE_DIR/errors_full.log"
    echo "  Captured full error log ($(wc -l < "$CAPTURE_DIR/errors_full.log" | tr -d ' ') lines)"
fi

if [ -f /tmp/ollama_errors_filtered.log ]; then
    cp /tmp/ollama_errors_filtered.log "$CAPTURE_DIR/errors_filtered.log"
    echo "  Captured filtered error log ($(wc -l < "$CAPTURE_DIR/errors_filtered.log" | tr -d ' ') lines)"
fi

if [ -f /tmp/ollama_stderr.log ]; then
    cp /tmp/ollama_stderr.log "$CAPTURE_DIR/stderr.log"
    echo "  Captured stderr log ($(wc -l < "$CAPTURE_DIR/stderr.log" | tr -d ' ') lines)"
fi

if [ -f /tmp/ollama_strace.log ]; then
    cp /tmp/ollama_strace.log "$CAPTURE_DIR/strace.log"
    echo "  Captured strace log ($(wc -l < "$CAPTURE_DIR/strace.log" | tr -d ' ') lines)"
fi

echo "  Capturing journalctl logs..."
journalctl -u ollama --since "2 minutes ago" --no-pager > "$CAPTURE_DIR/journalctl.log" 2>&1 || true
echo "    ($(wc -l < "$CAPTURE_DIR/journalctl.log" | tr -d ' ') lines)"

echo "  Capturing process information..."
ps aux | grep -E "(ollama|libvgpu)" > "$CAPTURE_DIR/processes.txt" 2>&1 || true

if [ -n "$OLLAMA_PID" ]; then
    cat /proc/$OLLAMA_PID/maps > "$CAPTURE_DIR/memory_maps.txt" 2>&1 || true
    echo "    Memory maps captured"
fi

cat > "$CAPTURE_DIR/SUMMARY.txt" <<EOF
Ollama Error Capture Summary
============================
Capture Time: $(date)
Duration: ${DURATION} seconds
Ollama PID: $OLLAMA_PID

Files Captured:
$(ls -lh "$CAPTURE_DIR" | tail -n +2)

Error Statistics:
- Full log lines: $(wc -l < "$CAPTURE_DIR/errors_full.log" 2>/dev/null || echo "0")
- Filtered log lines: $(wc -l < "$CAPTURE_DIR/errors_filtered.log" 2>/dev/null || echo "0")
- Stderr log lines: $(wc -l < "$CAPTURE_DIR/stderr.log" 2>/dev/null || echo "0")
- Strace log lines: $(wc -l < "$CAPTURE_DIR/strace.log" 2>/dev/null || echo "0")
- Journalctl log lines: $(wc -l < "$CAPTURE_DIR/journalctl.log" 2>/dev/null || echo "0")

Next Steps:
1. Review errors_filtered.log for key error messages
2. Check stderr.log for full error output
3. Analyze strace.log for syscall-level errors
4. Review journalctl.log for service-level errors
EOF

echo ""
echo "=========================================="
echo "Capture complete!"
echo "=========================================="
echo "All data saved to: $CAPTURE_DIR"
echo ""
echo "Summary:"
cat "$CAPTURE_DIR/SUMMARY.txt"
echo ""
echo "To analyze errors, run:"
echo "  ./analyze_errors.sh $CAPTURE_DIR"
