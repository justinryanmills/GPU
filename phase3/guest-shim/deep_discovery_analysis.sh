#!/bin/bash
# Deep Discovery Analysis Script
# Analyzes Ollama's GPU discovery logic step-by-step

set -e

LOG_DIR="/tmp/ollama_deep_analysis"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ANALYSIS_LOG="$LOG_DIR/analysis_${TIMESTAMP}.log"

echo "=== Deep Ollama Discovery Analysis ===" | tee "$ANALYSIS_LOG"
echo "Timestamp: $TIMESTAMP" | tee -a "$ANALYSIS_LOG"
echo "" | tee -a "$ANALYSIS_LOG"

# Stop Ollama
systemctl stop ollama 2>/dev/null || true
sleep 2

# Function to check what files Ollama accesses
check_file_access() {
    echo "[1] Checking what files Ollama accesses during discovery..." | tee -a "$ANALYSIS_LOG"
    
    # Start Ollama with strace
    strace -e trace=open,openat,stat,stat64,lstat,lstat64,access \
           -f -o "$LOG_DIR/strace_files.log" \
           -s 256 \
           ollama serve 2>&1 &
    STRACE_PID=$!
    
    sleep 5
    
    # Trigger discovery by making a request
    curl -s http://localhost:11434/api/tags > /dev/null 2>&1 || true
    
    sleep 3
    
    pkill -f 'ollama serve' || true
    sleep 2
    kill $STRACE_PID 2>/dev/null || true
    
    # Analyze file access
    echo "Files accessed related to NVIDIA/GPU:" | tee -a "$ANALYSIS_LOG"
    grep -E "nvidia|pci|drm|gpu|cuda" "$LOG_DIR/strace_files.log" | grep -v "ENOENT" | head -50 | tee -a "$ANALYSIS_LOG"
    echo "" | tee -a "$ANALYSIS_LOG"
}

# Function to check library calls
check_library_calls() {
    echo "[2] Checking library calls during discovery..." | tee -a "$ANALYSIS_LOG"
    
    # Start Ollama with ltrace (if available) or LD_DEBUG
    if command -v ltrace &> /dev/null; then
        ltrace -e 'nvml*|cu*' -f -o "$LOG_DIR/ltrace_libs.log" \
               ollama serve 2>&1 &
        LTRACE_PID=$!
        sleep 5
        curl -s http://localhost:11434/api/tags > /dev/null 2>&1 || true
        sleep 3
        pkill -f 'ollama serve' || true
        sleep 2
        kill $LTRACE_PID 2>/dev/null || true
        
        echo "NVML/CUDA library calls:" | tee -a "$ANALYSIS_LOG"
        grep -E "nvml|cu" "$LOG_DIR/ltrace_libs.log" | head -100 | tee -a "$ANALYSIS_LOG"
    else
        echo "ltrace not available, using LD_DEBUG instead..." | tee -a "$ANALYSIS_LOG"
        LD_DEBUG=libs LD_DEBUG_OUTPUT="$LOG_DIR/ld_debug.log" \
        ollama serve 2>&1 &
        OLLAMA_PID=$!
        sleep 5
        curl -s http://localhost:11434/api/tags > /dev/null 2>&1 || true
        sleep 3
        pkill -f 'ollama serve' || true
        sleep 2
        
        echo "Library loading:" | tee -a "$ANALYSIS_LOG"
        grep -E "libnvidia|libcuda|libvgpu" "$LOG_DIR/ld_debug.log" 2>/dev/null | head -50 | tee -a "$ANALYSIS_LOG"
    fi
    echo "" | tee -a "$ANALYSIS_LOG"
}

# Function to check runner subprocess
check_runner_subprocess() {
    echo "[3] Checking runner subprocess behavior..." | tee -a "$ANALYSIS_LOG"
    
    systemctl start ollama
    sleep 3
    
    # Find runner process
    RUNNER_PID=$(pgrep -f "ollama runner" | head -1)
    
    if [ -n "$RUNNER_PID" ]; then
        echo "Runner PID: $RUNNER_PID" | tee -a "$ANALYSIS_LOG"
        
        # Check environment
        echo "Runner environment:" | tee -a "$ANALYSIS_LOG"
        cat /proc/$RUNNER_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "LD_PRELOAD|LD_LIBRARY_PATH|OLLAMA|CUDA|NVIDIA" | tee -a "$ANALYSIS_LOG"
        
        # Check loaded libraries
        echo "Runner loaded libraries:" | tee -a "$ANALYSIS_LOG"
        cat /proc/$RUNNER_PID/maps 2>/dev/null | grep -E "libvgpu|libcuda|libnvidia-ml|libggml" | tee -a "$ANALYSIS_LOG"
        
        # Trace runner's syscalls
        echo "Tracing runner syscalls for 5 seconds..." | tee -a "$ANALYSIS_LOG"
        timeout 5 strace -p $RUNNER_PID -e trace=open,openat,read,readv,stat,stat64 \
                         -s 256 -o "$LOG_DIR/runner_syscalls.log" 2>&1 &
        STRACE_RUNNER_PID=$!
        sleep 5
        kill $STRACE_RUNNER_PID 2>/dev/null || true
        
        echo "Runner PCI-related syscalls:" | tee -a "$ANALYSIS_LOG"
        grep -E "pci|vendor|device|class" "$LOG_DIR/runner_syscalls.log" 2>/dev/null | head -30 | tee -a "$ANALYSIS_LOG"
    else
        echo "No runner process found" | tee -a "$ANALYSIS_LOG"
    fi
    
    systemctl stop ollama
    echo "" | tee -a "$ANALYSIS_LOG"
}

# Function to check validation steps
check_validation_steps() {
    echo "[4] Checking validation steps..." | tee -a "$ANALYSIS_LOG"
    
    # Check if Ollama looks for these files/devices
    VALIDATION_FILES=(
        "/proc/driver/nvidia/version"
        "/proc/driver/nvidia/params"
        "/dev/nvidia0"
        "/dev/nvidiactl"
        "/sys/class/drm/card0"
        "/sys/class/drm/card0/device"
    )
    
    echo "Checking validation file access:" | tee -a "$ANALYSIS_LOG"
    for file in "${VALIDATION_FILES[@]}"; do
        if grep -q "$file" "$LOG_DIR/strace_files.log" 2>/dev/null; then
            echo "  ✓ Ollama accessed: $file" | tee -a "$ANALYSIS_LOG"
            grep "$file" "$LOG_DIR/strace_files.log" | head -3 | tee -a "$ANALYSIS_LOG"
        else
            echo "  ✗ Ollama did NOT access: $file" | tee -a "$ANALYSIS_LOG"
        fi
    done
    echo "" | tee -a "$ANALYSIS_LOG"
}

# Function to check PCI-NVML matching
check_pci_nvml_matching() {
    echo "[5] Analyzing PCI-NVML matching logic..." | tee -a "$ANALYSIS_LOG"
    
    # Get PCI bus ID from filesystem
    PCI_BDF=$(ls -d /sys/bus/pci/devices/0000:00:05.0 2>/dev/null | xargs basename)
    if [ -n "$PCI_BDF" ]; then
        echo "PCI device found: $PCI_BDF" | tee -a "$ANALYSIS_LOG"
        
        # Read PCI values
        PCI_VENDOR=$(cat /sys/bus/pci/devices/$PCI_BDF/vendor 2>/dev/null || echo "NOT_FOUND")
        PCI_DEVICE=$(cat /sys/bus/pci/devices/$PCI_BDF/device 2>/dev/null || echo "NOT_FOUND")
        PCI_CLASS=$(cat /sys/bus/pci/devices/$PCI_BDF/class 2>/dev/null || echo "NOT_FOUND")
        
        echo "  Vendor: $PCI_VENDOR" | tee -a "$ANALYSIS_LOG"
        echo "  Device: $PCI_DEVICE" | tee -a "$ANALYSIS_LOG"
        echo "  Class: $PCI_CLASS" | tee -a "$ANALYSIS_LOG"
        
        # Get PCI bus ID from uevent
        PCI_BUS_ID=$(grep PCI_SLOT_NAME /sys/bus/pci/devices/$PCI_BDF/uevent 2>/dev/null | cut -d= -f2 || echo "NOT_FOUND")
        echo "  PCI Bus ID (from uevent): $PCI_BUS_ID" | tee -a "$ANALYSIS_LOG"
        
        # Test NVML to get bus ID
        echo "Testing NVML PCI bus ID..." | tee -a "$ANALYSIS_LOG"
        python3 << 'PYEOF' 2>&1 | tee -a "$ANALYSIS_LOG"
import ctypes
import sys

try:
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    
    # Initialize NVML
    result = nvml.nvmlInit_v2()
    if result != 0:
        print(f"NVML init failed: {result}")
        sys.exit(1)
    
    # Get device count
    count = ctypes.c_uint32()
    result = nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
    if result != 0:
        print(f"nvmlDeviceGetCount_v2 failed: {result}")
        sys.exit(1)
    
    print(f"NVML device count: {count.value}")
    
    # Get device handle
    device = ctypes.c_void_p()
    result = nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device))
    if result != 0:
        print(f"nvmlDeviceGetHandleByIndex_v2 failed: {result}")
        sys.exit(1)
    
    # Get PCI info
    class nvmlPciInfo_t(ctypes.Structure):
        _fields_ = [
            ("busId", ctypes.c_char * 64),
            ("domain", ctypes.c_uint32),
            ("bus", ctypes.c_uint8),
            ("device", ctypes.c_uint8),
            ("pciDeviceId", ctypes.c_uint32),
        ]
    
    pci_info = nvmlPciInfo_t()
    result = nvml.nvmlDeviceGetPciInfo_v3(device, ctypes.byref(pci_info))
    if result != 0:
        print(f"nvmlDeviceGetPciInfo_v3 failed: {result}")
        sys.exit(1)
    
    nvml_bus_id = pci_info.busId.decode('utf-8')
    print(f"NVML PCI Bus ID: {nvml_bus_id}")
    
    # Compare with filesystem
    with open('/sys/bus/pci/devices/0000:00:05.0/uevent', 'r') as f:
        for line in f:
            if line.startswith('PCI_SLOT_NAME='):
                fs_bus_id = line.split('=', 1)[1].strip()
                print(f"Filesystem PCI Bus ID: {fs_bus_id}")
                if nvml_bus_id == fs_bus_id:
                    print("✓ MATCH: NVML and filesystem bus IDs match!")
                else:
                    print(f"✗ MISMATCH: NVML='{nvml_bus_id}' vs Filesystem='{fs_bus_id}'")
                break
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
PYEOF
    else
        echo "No PCI device found at 0000:00:05.0" | tee -a "$ANALYSIS_LOG"
    fi
    echo "" | tee -a "$ANALYSIS_LOG"
}

# Function to check CUDA backend loading
check_cuda_backend() {
    echo "[6] Checking CUDA backend loading..." | tee -a "$ANALYSIS_LOG"
    
    # Check if libggml-cuda.so is loaded
    systemctl start ollama
    sleep 3
    
    for pid in $(pgrep -f ollama); do
        echo "Process $pid:" | tee -a "$ANALYSIS_LOG"
        if cat /proc/$pid/maps 2>/dev/null | grep -q "libggml-cuda"; then
            echo "  ✓ libggml-cuda.so is loaded" | tee -a "$ANALYSIS_LOG"
        else
            echo "  ✗ libggml-cuda.so is NOT loaded" | tee -a "$ANALYSIS_LOG"
        fi
        
        # Check for CUDA initialization errors
        if journalctl -u ollama --since "1 minute ago" --no-pager | grep -q "ggml_cuda_init"; then
            echo "  CUDA init messages:" | tee -a "$ANALYSIS_LOG"
            journalctl -u ollama --since "1 minute ago" --no-pager | grep "ggml_cuda_init" | tail -3 | tee -a "$ANALYSIS_LOG"
        fi
    done
    
    systemctl stop ollama
    echo "" | tee -a "$ANALYSIS_LOG"
}

# Run all checks
check_file_access
check_library_calls
check_runner_subprocess
check_validation_steps
check_pci_nvml_matching
check_cuda_backend

# Generate summary
echo "=== ANALYSIS SUMMARY ===" | tee -a "$ANALYSIS_LOG"
echo "Analysis complete. Review logs in $LOG_DIR" | tee -a "$ANALYSIS_LOG"
echo "Main log: $ANALYSIS_LOG" | tee -a "$ANALYSIS_LOG"
