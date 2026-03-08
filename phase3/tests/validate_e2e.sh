#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE3_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

VM_IP="${VM_IP:-}"
SKIP_BUILD=0
SKIP_DEPLOY=0

log()  { echo -e "${GREEN}[]${NC} $*"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}[✗]${NC} $*"; FAIL=$((FAIL+1)); }
skip() { echo -e "${YELLOW}[○]${NC} $*"; SKIP=$((SKIP+1)); }
info() { echo -e "${BLUE}[i]${NC} $*"; }

for arg in "$@"; do
    case "$arg" in
        --vm-ip=*) VM_IP="${arg#*=}" ;;
        --skip-build) SKIP_BUILD=1 ;;
        --skip-deploy) SKIP_DEPLOY=1 ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  VGPU CUDA Remoting — End-to-End Validation"
echo "═══════════════════════════════════════════════════════════════"
echo ""

info "PHASE 1: Host-side checks"
echo ""

if command -v nvcc &>/dev/null; then
    ver=$(nvcc --version 2>/dev/null | grep "release" | head -1)
    log "CUDA toolkit: $ver"
else
    fail "CUDA toolkit (nvcc) not found"
fi

if command -v nvidia-smi &>/dev/null; then
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    log "Physical GPU detected: $gpu"
else
    fail "nvidia-smi not found — no physical GPU?"
fi

for f in include/cuda_protocol.h include/cuda_executor.h \
         src/cuda_executor.c src/mediator_phase3.c src/vgpu-stub-enhanced.c \
         guest-shim/libvgpu_cuda.c guest-shim/libvgpu_nvml.c \
         guest-shim/cuda_transport.c guest-shim/install.sh; do
    if [[ -f "${PHASE3_DIR}/${f}" ]]; then
        log "Source: ${f}"
    else
        fail "Missing: ${f}"
    fi
done

echo ""

info "PHASE 2: Build validation"
echo ""

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    cd "$PHASE3_DIR"

    if make host 2>&1 | tail -5; then
        log "Host binaries built (mediator_phase3, vgpu-admin)"
    else
        fail "Host build failed"
    fi

    if make guest 2>&1 | tail -5; then
        log "Guest shim libraries built (libvgpu-cuda.so, libvgpu-nvml.so)"
    else
        fail "Guest shim build failed"
    fi
else
    skip "Build skipped (--skip-build)"
fi

for bin in mediator_phase3 vgpu-admin; do
    if [[ -x "${PHASE3_DIR}/${bin}" ]]; then
        log "Binary: ${bin}"
    else
        fail "Missing binary: ${bin}"
    fi
done

for lib in guest-shim/libvgpu-cuda.so guest-shim/libvgpu-nvml.so; do
    if [[ -f "${PHASE3_DIR}/${lib}" ]]; then
        log "Shim lib: ${lib}"
    else
        fail "Missing shim: ${lib}"
    fi
done

echo ""

info "PHASE 3: Mediator daemon"
echo ""

if pgrep -f mediator_phase3 &>/dev/null; then
    log "Mediator daemon is running"
else
    info "Mediator daemon not running — starting..."
    if [[ -x "${PHASE3_DIR}/mediator_phase3" ]]; then
        "${PHASE3_DIR}/mediator_phase3" &
        sleep 2
        if pgrep -f mediator_phase3 &>/dev/null; then
            log "Mediator daemon started"
        else
            fail "Mediator daemon failed to start"
        fi
    else
        fail "Mediator binary not found"
    fi
fi

if [[ -S /var/vgpu/mediator.sock ]]; then
    log "Mediator socket: /var/vgpu/mediator.sock"
else
    fail "Mediator socket not found"
fi

echo ""

info "PHASE 4: QEMU with vgpu-stub device"
echo ""

QEMU_BIN="/usr/lib64/xen/bin/qemu-system-i386"
if [[ -x "$QEMU_BIN" ]]; then
    if "$QEMU_BIN" -device help 2>/dev/null | grep -q "vgpu-stub"; then
        log "QEMU has vgpu-stub device registered"
    else
        fail "QEMU does not have vgpu-stub device (rebuild needed)"
    fi
else
    skip "QEMU binary not at expected path (XCP-ng only)"
fi

echo ""

info "PHASE 5: Guest VM validation"
echo ""

if [[ -z "$VM_IP" ]]; then
    skip "No --vm-ip provided — skipping guest VM tests"
    skip "  Run: $0 --vm-ip=<guest-ip>"
else
    info "Connecting to guest VM at ${VM_IP}..."

    if ssh -o ConnectTimeout=5 -o BatchMode=yes "root@${VM_IP}" true 2>/dev/null; then
        log "SSH access to guest VM"
    else
        fail "Cannot SSH to root@${VM_IP}"
        echo ""
        echo "Ensure SSH key-based access is configured."
        FAIL=$((FAIL+1))
    fi

    if [[ "$SKIP_DEPLOY" -eq 0 ]]; then
        info "Deploying shim libraries..."
        ssh "root@${VM_IP}" "mkdir -p /tmp/vgpu" 2>/dev/null || true

        scp -q "${PHASE3_DIR}/guest-shim/libvgpu-cuda.so" \
               "${PHASE3_DIR}/guest-shim/libvgpu-nvml.so" \
               "${PHASE3_DIR}/guest-shim/install.sh" \
               "${PHASE3_DIR}/guest-shim/libvgpu_cuda.c" \
               "${PHASE3_DIR}/guest-shim/libvgpu_nvml.c" \
               "${PHASE3_DIR}/guest-shim/cuda_transport.c" \
               "${PHASE3_DIR}/guest-shim/cuda_transport.h" \
               "${PHASE3_DIR}/guest-shim/gpu_properties.h" \
               "root@${VM_IP}:/tmp/vgpu/" 2>/dev/null

        scp -q "${PHASE3_DIR}/include/cuda_protocol.h" \
               "root@${VM_IP}:/tmp/vgpu/" 2>/dev/null

        log "Shim libraries deployed to guest"
    fi

    pci_out=$(ssh "root@${VM_IP}" "lspci -nn 2>/dev/null | grep -i '1af4:1111'" 2>/dev/null || true)
    if [[ -n "$pci_out" ]]; then
        log "VGPU-STUB PCI device visible in guest: $pci_out"
    else
        fail "VGPU-STUB PCI device NOT visible in guest"
    fi

    if ssh "root@${VM_IP}" "test -f /usr/lib64/libvgpu-cuda.so" 2>/dev/null; then
        log "libvgpu-cuda.so installed in guest"
    else
        info "Running install.sh in guest..."
        ssh "root@${VM_IP}" "cd /tmp/vgpu && sudo bash install.sh" 2>/dev/null || true
    fi

    if ssh "root@${VM_IP}" "test -e /dev/nvidia0" 2>/dev/null; then
        log "/dev/nvidia0 exists in guest"
    else
        fail "/dev/nvidia0 missing in guest"
    fi

    cuda_check=$(ssh "root@${VM_IP}" \
        "LD_LIBRARY_PATH=/usr/lib64 python3 -c 'import ctypes; cuda=ctypes.CDLL(\"libcuda.so.1\"); print(\"OK\")' 2>&1" \
        2>/dev/null || true)
    if [[ "$cuda_check" == *"OK"* ]]; then
        log "CUDA shim library loads successfully in guest"
    else
        fail "CUDA shim failed to load: $cuda_check"
    fi

    ollama_ver=$(ssh "root@${VM_IP}" "ollama --version 2>/dev/null" 2>/dev/null || true)
    if [[ -n "$ollama_ver" ]]; then
        log "Ollama installed: $ollama_ver"

        info "Testing Ollama with llama3.2:1b (this may take a moment)..."
        ollama_out=$(ssh -o ConnectTimeout=120 "root@${VM_IP}" \
            "timeout 120 ollama run llama3.2:1b 'Say hello in one word' 2>&1" \
            2>/dev/null || true)
        if [[ -n "$ollama_out" && "$ollama_out" != *"error"* ]]; then
            log "Ollama responded: $(echo "$ollama_out" | head -1)"
        else
            fail "Ollama test failed: $ollama_out"
        fi
    else
        skip "Ollama not installed in guest (use install.sh --with-ollama)"
    fi
fi

echo ""

echo "═══════════════════════════════════════════════════════════════"
echo -e "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
    echo "Some checks failed. Review the output above for details."
    exit 1
fi

echo "All checks passed!"
exit 0
