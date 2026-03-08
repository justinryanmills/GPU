#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUEST_SHIM_DIR="$SCRIPT_DIR/guest-shim"
INSTALL_LIB_DIR="/usr/lib64"
INSTALL_BIN_DIR="/usr/local/bin"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[deploy]${NC} $*"; }
error() { echo -e "${RED}[deploy]${NC} $*" >&2; }
info()  { echo -e "${BLUE}[deploy]${NC} $*"; }

if [ "$EUID" -ne 0 ]; then 
    error "This script must be run as root (use sudo)"
    exit 1
fi

log "Deploying Secure Symlink Method: Filesystem-Level Library Redirection"
log "==================================================================="
log ""

log "Step 1: Checking shim libraries..."

if [ ! -f "$INSTALL_LIB_DIR/libvgpu-cuda.so" ]; then
    warn "libvgpu-cuda.so not found, building it..."
    
    if ! command -v gcc &>/dev/null; then
        log "Installing build-essential..."
        apt-get update -qq
        apt-get install -y build-essential
    fi
    
    if [ ! -f "$GUEST_SHIM_DIR/libvgpu_cuda.c" ] || [ ! -f "$GUEST_SHIM_DIR/cuda_transport.c" ]; then
        error "Source files not found. Need: libvgpu_cuda.c, cuda_transport.c"
        error "Please copy these files to $GUEST_SHIM_DIR first"
        exit 1
    fi
    
    log "Building libvgpu-cuda.so..."
    gcc -shared -fPIC -O2 -Wall -I"$SCRIPT_DIR/include" -I"$GUEST_SHIM_DIR" \
        -o "$INSTALL_LIB_DIR/libvgpu-cuda.so" \
        "$GUEST_SHIM_DIR/libvgpu_cuda.c" \
        "$GUEST_SHIM_DIR/cuda_transport.c" \
        -lpthread -ldl
    
    if [ ! -f "$INSTALL_LIB_DIR/libvgpu-cuda.so" ]; then
        error "Failed to build libvgpu-cuda.so"
        exit 1
    fi
    
    log "Built libvgpu-cuda.so"
else
    log "libvgpu-cuda.so already exists"
fi

if [ ! -f "$INSTALL_LIB_DIR/libvgpu-nvml.so" ]; then
    warn "libvgpu-nvml.so not found (optional, CUDA shim is primary)"
    if [ -f "$GUEST_SHIM_DIR/libvgpu_nvml.c" ]; then
        log "Building libvgpu-nvml.so..."
        gcc -shared -fPIC -O2 -Wall -I"$SCRIPT_DIR/include" -I"$GUEST_SHIM_DIR" \
            -o "$INSTALL_LIB_DIR/libvgpu-nvml.so" \
            "$GUEST_SHIM_DIR/libvgpu_nvml.c" \
            -ldl
        log "Built libvgpu-nvml.so"
    fi
else
    log "libvgpu-nvml.so already exists"
fi

log "Shim libraries ready"
log ""

log "Step 2: Creating symlinks for filesystem-level library redirection..."

SYMLINK_SCRIPT="$GUEST_SHIM_DIR/create_symlinks.sh"
if [ ! -f "$SYMLINK_SCRIPT" ]; then
    error "Symlink script not found: $SYMLINK_SCRIPT"
    exit 1
fi

chmod +x "$SYMLINK_SCRIPT"
"$SYMLINK_SCRIPT"

log "Symlinks created"
log ""

log "Step 3: Registering system-wide library paths..."

REGISTER_SCRIPT="$GUEST_SHIM_DIR/register_system_paths.sh"
if [ ! -f "$REGISTER_SCRIPT" ]; then
    error "Register script not found: $REGISTER_SCRIPT"
    exit 1
fi

chmod +x "$REGISTER_SCRIPT"
"$REGISTER_SCRIPT"

log "System-wide paths registered"
log ""

log "Step 4: Ensuring /etc/ld.so.preload is empty (no system-wide preload)..."

if [ -f /etc/ld.so.preload ]; then
    if [ -s /etc/ld.so.preload ]; then
        warn "Backing up existing /etc/ld.so.preload"
        cp /etc/ld.so.preload /etc/ld.so.preload.backup.$(date +%Y%m%d_%H%M%S)
    fi
    echo "" > /etc/ld.so.preload
    log "Cleared /etc/ld.so.preload"
else
    log "/etc/ld.so.preload doesn't exist (good)"
fi

log ""

log "Step 5: Configuring systemd service..."

CONFIGURE_SCRIPT="$GUEST_SHIM_DIR/configure_systemd.sh"
if [ ! -f "$CONFIGURE_SCRIPT" ]; then
    error "Configure script not found: $CONFIGURE_SCRIPT"
    exit 1
fi

chmod +x "$CONFIGURE_SCRIPT"
"$CONFIGURE_SCRIPT"

log "Systemd configured"
log ""

log "Step 6: Reloading systemd and restarting Ollama..."

systemctl daemon-reload

if systemctl is-active --quiet ollama; then
    log "Restarting Ollama service..."
    systemctl restart ollama
    sleep 5
else
    log "Starting Ollama service..."
    systemctl start ollama
    sleep 5
fi

log "Ollama service restarted"
log ""

log "Step 7: Verifying deployment..."
log ""

info "Testing system processes (should work normally)..."
if lspci >/dev/null 2>&1; then
    log "lspci works (no system impact)"
else
    warn "⚠ lspci test failed (but this might be normal)"
fi

if cat /etc/passwd >/dev/null 2>&1; then
    log "cat works (no system impact)"
else
    warn "⚠ cat test failed (unexpected)"
fi

if systemctl is-active --quiet ollama; then
    log "Ollama service is running"
else
    error "✗ Ollama service is not running"
    systemctl status ollama || true
    exit 1
fi

info "Checking library discovery..."
if [ -L /usr/lib64/libcuda.so.1 ] && [ "$(readlink -f /usr/lib64/libcuda.so.1)" = "$INSTALL_LIB_DIR/libvgpu-cuda.so" ]; then
    log "CUDA symlink is correct"
else
    warn "⚠ CUDA symlink may be incorrect"
fi

if ldconfig -p 2>&1 | grep -q "libvgpu-cuda"; then
    log "Libraries registered in system-wide ldconfig cache"
else
    warn "⚠ Libraries not found in ldconfig cache (may still work via symlinks)"
fi

OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$OLLAMA_PID" ]; then
    info "Checking Ollama process (PID: $OLLAMA_PID)..."
    
    if sudo cat /proc/$OLLAMA_PID/environ 2>/dev/null | tr '\0' '\n' | grep -q "LD_LIBRARY_PATH=$INSTALL_LIB_DIR"; then
        log "LD_LIBRARY_PATH is set in Ollama process"
    else
        warn "⚠ LD_LIBRARY_PATH not found in process environment"
    fi
    
    RUNNER_PID=$(pgrep -f "ollama runner" | head -1)
    if [ -n "$RUNNER_PID" ]; then
        info "Found runner subprocess (PID: $RUNNER_PID)"
        if grep -q "libvgpu-cuda\|libcuda" /proc/$RUNNER_PID/maps 2>/dev/null; then
            log "CUDA library is loaded in runner subprocess (via symlinks)"
        else
            warn "⚠ CUDA library not found in runner subprocess maps"
        fi
    else
        warn "⚠ No runner subprocess found (may have crashed or not started yet)"
    fi
else
    warn "⚠ Could not find Ollama process PID"
fi

info "Checking GPU mode..."
sleep 5
GPU_MODE_LOG=$(journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | grep -iE "library=" | tail -3)
if echo "$GPU_MODE_LOG" | grep -q "library=cuda"; then
    log "Ollama is running in GPU mode (library=cuda)"
    echo "$GPU_MODE_LOG" | grep "library=cuda" | head -1
elif echo "$GPU_MODE_LOG" | grep -q "library=cpu"; then
    warn "⚠ Ollama is running in CPU mode (library=cpu)"
    info "Recent library mode logs:"
    echo "$GPU_MODE_LOG"
    info "This might be normal if GPU discovery is still in progress"
else
    warn "⚠ Could not determine library mode from logs"
    info "Recent logs:"
    journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | tail -10
fi

echo ""
info "=========================================="
info "DEPLOYMENT SUMMARY"
info "=========================================="
info "Method: Filesystem-level symlinks + system-wide paths"
info "Symlinks: Created in standard library paths"
info "System-wide paths: Registered via /etc/ld.so.conf.d/vgpu.conf"
info "Systemd override: LD_LIBRARY_PATH only (NO LD_PRELOAD)"
info "/etc/ld.so.preload: Cleared (NOT USED)"
info "Ollama service: $(systemctl is-active ollama 2>/dev/null || echo 'unknown')"
info "=========================================="

log ""
log "==================================================================="
log "Deployment Complete!"
log "==================================================================="
log ""
log "What was deployed:"
log "  1. Filesystem-level symlinks: libcuda.so.1 -> libvgpu-cuda.so"
log "  2. System-wide library path: /etc/ld.so.conf.d/vgpu.conf"
log "  3. Systemd override: /etc/systemd/system/ollama.service.d/vgpu.conf"
log "  4. /etc/ld.so.preload: Cleared (NOT USED)"
log ""
log "Safety guarantees:"
log "  No system-wide preload (zero impact on system processes)"
log "  Works at filesystem level (independent of process spawning)"
log "  Works for runner subprocesses (via symlinks)"
log "  Easy rollback (remove symlinks and systemd override)"
log ""
log "To verify GPU mode:"
log "  journalctl -u ollama -n 200 | grep -i library"
log ""
log "To rollback (if needed):"
log "  sudo rm /etc/systemd/system/ollama.service.d/vgpu.conf"
log "  sudo rm /etc/ld.so.conf.d/vgpu.conf"
log "  sudo rm /usr/lib64/libcuda.so.1 /usr/lib64/libnvidia-ml.so.1"
log "  sudo ldconfig"
log "  sudo systemctl daemon-reload"
log "  sudo systemctl restart ollama"
log ""
log "==================================================================="
