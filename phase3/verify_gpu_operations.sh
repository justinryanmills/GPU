#!/bin/bash
set -e

echo "=========================================="
echo "GPU Operations Verification Test"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -f /sys/bus/pci/devices/0000:00:05.0/vendor ]; then
    echo -e "${YELLOW}Running on VM (guest)${NC}"
    IS_VM=1
    VM_IP="10.25.33.11"
    HOST_IP="10.25.33.10"
else
    echo -e "${YELLOW}Running on host${NC}"
    IS_VM=0
    HOST_IP="localhost"
fi

echo ""
if [ "$IS_VM" = "1" ]; then
    ssh -o StrictHostKeyChecking=no root@${HOST_IP} "ps aux | grep -E 'mediator_phase3|mediator' | grep -v grep" || {
        echo -e "${RED}ERROR: Mediator not running on host${NC}"
        exit 1
    }
    echo -e "${GREEN}Mediator is running on host${NC}"
else
    ps aux | grep -E 'mediator_phase3|mediator' | grep -v grep || {
        echo -e "${RED}ERROR: Mediator not running${NC}"
        exit 1
    }
    echo -e "${GREEN}Mediator is running${NC}"
fi

echo ""
if [ "$IS_VM" = "1" ]; then
    ssh -o StrictHostKeyChecking=no test-3@${VM_IP} "timeout 30 ollama run llama3.2:1b 'Calculate 123*456 and show your work step by step' 2>&1 | tail -10" || true
else
    timeout 30 ollama run llama3.2:1b 'Calculate 123*456 and show your work step by step' 2>&1 | tail -10 || true
fi

echo ""
if [ "$IS_VM" = "1" ]; then
    ssh -o StrictHostKeyChecking=no root@${HOST_IP} "tail -50 /tmp/mediator.log 2>/dev/null || tail -50 /var/log/mediator_phase3.log 2>/dev/null || echo 'No mediator log found'" | grep -E 'cuLaunchKernel|cuMemcpy|cuMemAlloc|cuda-executor.*SUCCESS|cuda-executor.*FAILED' | tail -20 || echo -e "${YELLOW}No GPU operation logs found${NC}"
else
    tail -50 /tmp/mediator.log 2>/dev/null || tail -50 /var/log/mediator_phase3.log 2>/dev/null | grep -E 'cuLaunchKernel|cuMemcpy|cuMemAlloc|cuda-executor.*SUCCESS|cuda-executor.*FAILED' | tail -20 || echo -e "${YELLOW}No GPU operation logs found${NC}"
fi

echo ""
if [ "$IS_VM" = "1" ]; then
    ssh -o StrictHostKeyChecking=no test-3@${VM_IP} "journalctl -u ollama.service --since '2 minutes ago' --no-pager | grep -E 'cuda_transport_call|CUDA_CALL_LAUNCH|CUDA_CALL_MEMCPY' | tail -10" || echo -e "${YELLOW}No transport call logs found in VM${NC}"
fi

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
