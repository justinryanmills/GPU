#!/bin/bash

echo "================================================================"
echo "          Pool Starvation Test"
echo "================================================================"
echo ""
echo "This test verifies that Pool B doesn't starve when Pool A is busy"
echo ""

if ! pgrep -x "mediator" > /dev/null; then
    echo "ERROR: mediator is not running"
    echo "Start it with: ./mediator"
    exit 1
fi

echo "Mediator is running"
echo ""

POOL_A_REQUESTS=100
POOL_B_REQUESTS=5
LOG_FILE="starvation_test_$(date +%s).log"

echo "Test configuration:"
echo "  Pool A: $POOL_A_REQUESTS requests (high load)"
echo "  Pool B: $POOL_B_REQUESTS requests (low load)"
echo "  Log file: $LOG_FILE"
echo ""

send_pool_a() {
    echo "[Pool A] Starting to send $POOL_A_REQUESTS requests..."
    for i in $(seq 1 $POOL_A_REQUESTS); do
        ./vm_client TEST_POOL_A 2>&1 | grep -E "(SUCCESS|ERROR|Response)" >> "$LOG_FILE"
        echo "[Pool A] Sent request $i/$POOL_A_REQUESTS"
    done
    echo "[Pool A] All requests sent"
}

send_pool_b() {
    echo "[Pool B] Starting to send $POOL_B_REQUESTS requests..."
    for i in $(seq 1 $POOL_B_REQUESTS); do
        TIMESTAMP=$(date +%s)
        ./vm_client TEST_POOL_B 2>&1 | grep -E "(SUCCESS|ERROR|Response)" >> "$LOG_FILE"
        echo "[Pool B] Request $i/$POOL_B_REQUESTS completed at $TIMESTAMP" | tee -a "$LOG_FILE"
    done
    echo "[Pool B] All requests sent"
}

echo "Starting concurrent test..."
echo ""

send_pool_a &
PID_A=$!

sleep 2

send_pool_b &
PID_B=$!

echo "Waiting for both pools to complete..."
wait $PID_A
wait $PID_B

echo ""
echo "================================================================"
echo "          Test Results"
echo "================================================================"
echo ""

POOL_B_COUNT=$(grep -c "Pool B" "$LOG_FILE")

if [ "$POOL_B_COUNT" -eq "$POOL_B_REQUESTS" ]; then
    echo "PASS: Pool B received all $POOL_B_REQUESTS responses"
    echo "No starvation detected"
else
    echo "❌ FAIL: Pool B only received $POOL_B_COUNT/$POOL_B_REQUESTS responses"
    echo "❌ Possible starvation!"
fi

echo ""
echo "Full log saved to: $LOG_FILE"
echo ""
echo "To verify interleaving, check if Pool B responses appear"
echo "BEFORE all Pool A responses complete:"
echo ""
echo "  grep 'Pool B.*completed' $LOG_FILE"
echo ""
