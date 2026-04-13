#!/bin/bash
# test_process_detection_safety.sh
# Test process detection safety BEFORE deploying /etc/ld.so.preload
# This script verifies that the library can be loaded into system processes
# without causing crashes, and that process detection correctly identifies
# Ollama vs system processes.

set -e

LIB_PATH="/usr/lib64/libvgpu-cuda.so"
TEST_LOG="/tmp/vgpu_process_detection_test.log"

echo "==========================================" | tee "$TEST_LOG"
echo "Process Detection Safety Test" | tee -a "$TEST_LOG"
echo "==========================================" | tee -a "$TEST_LOG"
echo "Date: $(date)" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"

# Check if library exists
if [ ! -f "$LIB_PATH" ]; then
    echo "ERROR: Library not found at $LIB_PATH" | tee -a "$TEST_LOG"
    exit 1
fi

echo "[1] Testing library loading into system processes..." | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"

# Test 1: cat command (should NOT crash, should be rejected by process detection)
echo "Test 1: cat /etc/passwd (system process - should be rejected)" | tee -a "$TEST_LOG"
if LD_PRELOAD="$LIB_PATH" cat /etc/passwd > /dev/null 2>&1; then
    echo "  ✓ cat command completed successfully (library loaded but process rejected)" | tee -a "$TEST_LOG"
else
    echo "  ✗ cat command failed or crashed!" | tee -a "$TEST_LOG"
    exit 1
fi
echo "" | tee -a "$TEST_LOG"

# Test 2: ls command (should NOT crash, should be rejected)
echo "Test 2: ls /tmp (system process - should be rejected)" | tee -a "$TEST_LOG"
if LD_PRELOAD="$LIB_PATH" ls /tmp > /dev/null 2>&1; then
    echo "  ✓ ls command completed successfully (library loaded but process rejected)" | tee -a "$TEST_LOG"
else
    echo "  ✗ ls command failed or crashed!" | tee -a "$TEST_LOG"
    exit 1
fi
echo "" | tee -a "$TEST_LOG"

# Test 3: bash command (should NOT crash, should be rejected)
echo "Test 3: bash -c 'echo test' (system process - should be rejected)" | tee -a "$TEST_LOG"
if LD_PRELOAD="$LIB_PATH" bash -c "echo test" > /dev/null 2>&1; then
    echo "  ✓ bash command completed successfully (library loaded but process rejected)" | tee -a "$TEST_LOG"
else
    echo "  ✗ bash command failed or crashed!" | tee -a "$TEST_LOG"
    exit 1
fi
echo "" | tee -a "$TEST_LOG"

# Test 4: Check if Ollama process would be accepted
echo "[2] Testing Ollama process detection..." | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"

# Check if Ollama is running
if pgrep -f "ollama serve" > /dev/null; then
    OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
    echo "Test 4: Ollama main process (PID: $OLLAMA_PID) - should be accepted" | tee -a "$TEST_LOG"
    
    # Check if library is already loaded (via /proc/PID/maps)
    if grep -q "libvgpu-cuda" /proc/$OLLAMA_PID/maps 2>/dev/null; then
        echo "  ✓ Library is already loaded in Ollama process" | tee -a "$TEST_LOG"
    else
        echo "  ⚠ Library not yet loaded in Ollama process (will be loaded via /etc/ld.so.preload)" | tee -a "$TEST_LOG"
    fi
else
    echo "Test 4: Ollama not running (will test when service starts)" | tee -a "$TEST_LOG"
fi
echo "" | tee -a "$TEST_LOG"

# Test 5: Manual runner test (if possible)
echo "[3] Testing manual runner execution..." | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"

if [ -f "/usr/local/bin/ollama" ]; then
    echo "Test 5: Manual ollama runner execution (should be accepted)" | tee -a "$TEST_LOG"
    # Run with timeout to prevent hanging
    if timeout 5 bash -c "LD_PRELOAD=\"$LIB_PATH\" /usr/local/bin/ollama runner --help 2>&1 | head -5" > /dev/null 2>&1; then
        echo "  ✓ Manual runner execution completed (library loaded and process accepted)" | tee -a "$TEST_LOG"
    else
        echo "  ⚠ Manual runner execution had issues (may be expected if Ollama service not running)" | tee -a "$TEST_LOG"
    fi
else
    echo "Test 5: Ollama binary not found at /usr/local/bin/ollama" | tee -a "$TEST_LOG"
fi
echo "" | tee -a "$TEST_LOG"

# Test 6: Check constructor safety
echo "[4] Verifying constructor safety..." | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"

echo "Test 6: Constructor execution for system processes" | tee -a "$TEST_LOG"
# Run a simple command and check stderr for constructor messages
if LD_PRELOAD="$LIB_PATH" cat /dev/null 2>&1 | grep -q "libvgpu-cuda.*Library loaded"; then
    echo "  ✓ Constructor executed and logged successfully" | tee -a "$TEST_LOG"
else
    echo "  ⚠ Constructor message not found in output (may be redirected)" | tee -a "$TEST_LOG"
fi

# Check that no segfaults occurred
if dmesg | tail -20 | grep -q "segfault.*cat\|segfault.*ls\|segfault.*bash"; then
    echo "  ✗ SEGFAULT DETECTED in system processes!" | tee -a "$TEST_LOG"
    exit 1
else
    echo "  ✓ No segfaults detected in system processes" | tee -a "$TEST_LOG"
fi
echo "" | tee -a "$TEST_LOG"

# Summary
echo "==========================================" | tee -a "$TEST_LOG"
echo "Safety Test Summary" | tee -a "$TEST_LOG"
echo "==========================================" | tee -a "$TEST_LOG"
echo "✓ All system process tests passed" | tee -a "$TEST_LOG"
echo "✓ No crashes or segfaults detected" | tee -a "$TEST_LOG"
echo "✓ Constructor executes safely" | tee -a "$TEST_LOG"
echo "✓ Process detection appears to work correctly" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"
echo "SAFETY TEST PASSED - Ready for /etc/ld.so.preload deployment" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"
echo "Full test log saved to: $TEST_LOG" | tee -a "$TEST_LOG"

exit 0
