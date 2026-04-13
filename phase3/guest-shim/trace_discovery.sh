#!/bin/bash
# Aggressive Discovery Path Tracing Script
# Captures exactly what Ollama does during GPU discovery

set -e

LOG_DIR="/tmp/ollama_discovery_trace"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACE_LOG="$LOG_DIR/trace_${TIMESTAMP}.log"
SYSCALL_LOG="$LOG_DIR/syscalls_${TIMESTAMP}.log"
PROCESS_LOG="$LOG_DIR/processes_${TIMESTAMP}.log"
LIBRARY_LOG="$LOG_DIR/libraries_${TIMESTAMP}.log"

echo "=== Aggressive Ollama Discovery Tracing ===" | tee "$TRACE_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$TRACE_LOG"
echo "Log directory: $LOG_DIR" | tee -a "$TRACE_LOG"
echo "" | tee -a "$TRACE_LOG"

# Stop Ollama if running
echo "[1] Stopping Ollama service..." | tee -a "$TRACE_LOG"
systemctl stop ollama 2>/dev/null || true
sleep 2

# Function to trace syscalls
trace_syscalls() {
    echo "[2] Starting syscall tracing..." | tee -a "$TRACE_LOG"
    strace -e trace=open,openat,read,readv,pread,preadv,clone,fork,execve,stat,stat64,lstat,lstat64 \
           -f -o "$SYSCALL_LOG" \
           -s 256 \
           -yy \
           ollama serve 2>&1 &
    STRACE_PID=$!
    echo "strace PID: $STRACE_PID" | tee -a "$TRACE_LOG"
    sleep 8
    pkill -f 'ollama serve' || true
    sleep 2
    kill $STRACE_PID 2>/dev/null || true
}

# Function to trace library calls
trace_libraries() {
    echo "[3] Starting library call tracing..." | tee -a "$TRACE_LOG"
    LD_DEBUG=all \
    LD_DEBUG_OUTPUT="$LIBRARY_LOG" \
    ollama serve 2>&1 &
    OLLAMA_PID=$!
    echo "Ollama PID: $OLLAMA_PID" | tee -a "$TRACE_LOG"
    sleep 8
    pkill -f 'ollama serve' || true
    sleep 2
}

# Function to monitor processes
monitor_processes() {
    echo "[4] Monitoring Ollama processes..." | tee -a "$TRACE_LOG"
    systemctl start ollama
    sleep 3
    
    for pid in $(pgrep -f ollama); do
        echo "=== PID $pid ===" | tee -a "$PROCESS_LOG"
        echo "Command: $(ps -p $pid -o cmd=)" | tee -a "$PROCESS_LOG"
        echo "Environment:" | tee -a "$PROCESS_LOG"
        cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep -E "LD_PRELOAD|LD_LIBRARY_PATH|OLLAMA|CUDA|NVIDIA" | tee -a "$PROCESS_LOG"
        echo "Loaded libraries:" | tee -a "$PROCESS_LOG"
        cat /proc/$pid/maps 2>/dev/null | grep -E "libvgpu|libcuda|libnvidia-ml|libggml" | tee -a "$PROCESS_LOG"
        echo "Current syscall:" | tee -a "$PROCESS_LOG"
        cat /proc/$pid/syscall 2>/dev/null | tee -a "$PROCESS_LOG"
        echo "" | tee -a "$PROCESS_LOG"
    done
    
    systemctl stop ollama
}

# Function to analyze PCI device access
analyze_pci_access() {
    echo "[5] Analyzing PCI device access patterns..." | tee -a "$TRACE_LOG"
    
    if [ -f "$SYSCALL_LOG" ]; then
        echo "PCI-related syscalls:" | tee -a "$TRACE_LOG"
        grep -E "/sys/bus/pci|vendor|device|class" "$SYSCALL_LOG" | head -50 | tee -a "$TRACE_LOG"
        
        echo "" | tee -a "$TRACE_LOG"
        echo "File opens related to PCI:" | tee -a "$TRACE_LOG"
        grep -E "openat.*pci|open.*pci" "$SYSCALL_LOG" | tee -a "$TRACE_LOG"
        
        echo "" | tee -a "$TRACE_LOG"
        echo "Read operations on PCI files:" | tee -a "$TRACE_LOG"
        grep -E "read.*pci|pread.*pci" "$SYSCALL_LOG" | tee -a "$TRACE_LOG"
    fi
}

# Function to check subprocess spawning
check_subprocesses() {
    echo "[6] Checking for subprocess spawning..." | tee -a "$TRACE_LOG"
    
    if [ -f "$SYSCALL_LOG" ]; then
        echo "Process spawning syscalls:" | tee -a "$TRACE_LOG"
        grep -E "clone|fork|execve" "$SYSCALL_LOG" | head -30 | tee -a "$TRACE_LOG"
        
        echo "" | tee -a "$TRACE_LOG"
        echo "Execve calls (showing command and environment):" | tee -a "$TRACE_LOG"
        grep "execve" "$SYSCALL_LOG" | grep -v "ENOENT" | head -20 | tee -a "$TRACE_LOG"
    fi
}

# Run all tracing
trace_syscalls
trace_libraries
monitor_processes
analyze_pci_access
check_subprocesses

# Generate summary
echo "" | tee -a "$TRACE_LOG"
echo "=== TRACING SUMMARY ===" | tee -a "$TRACE_LOG"
echo "Syscall log: $SYSCALL_LOG" | tee -a "$TRACE_LOG"
echo "Process log: $PROCESS_LOG" | tee -a "$TRACE_LOG"
echo "Library log: $LIBRARY_LOG" | tee -a "$TRACE_LOG"
echo "" | tee -a "$TRACE_LOG"
echo "Key findings:" | tee -a "$TRACE_LOG"

if [ -f "$SYSCALL_LOG" ]; then
    PCI_OPENS=$(grep -c "openat.*pci" "$SYSCALL_LOG" 2>/dev/null || echo "0")
    PCI_READS=$(grep -c "read.*pci" "$SYSCALL_LOG" 2>/dev/null || echo "0")
    SUBPROCESSES=$(grep -c "execve" "$SYSCALL_LOG" 2>/dev/null || echo "0")
    
    echo "  PCI device file opens: $PCI_OPENS" | tee -a "$TRACE_LOG"
    echo "  PCI device file reads: $PCI_READS" | tee -a "$TRACE_LOG"
    echo "  Subprocess spawns: $SUBPROCESSES" | tee -a "$TRACE_LOG"
fi

echo "" | tee -a "$TRACE_LOG"
echo "Tracing complete. Review logs in $LOG_DIR" | tee -a "$TRACE_LOG"
