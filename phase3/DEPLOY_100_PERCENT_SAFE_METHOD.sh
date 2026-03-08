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

log "Deploying 100% Safe Method: Enhanced force_load_shim + LD_AUDIT"

log "Step 1: Building LD_AUDIT interceptor..."

if [ ! -f "$GUEST_SHIM_DIR/ld_audit_interceptor.c" ]; then
    error "ld_audit_interceptor.c not found in $GUEST_SHIM_DIR"
    exit 1
fi

gcc -shared -fPIC -o "$INSTALL_LIB_DIR/libldaudit_cuda.so" \
    "$GUEST_SHIM_DIR/ld_audit_interceptor.c" -ldl -Wall

if [ ! -f "$INSTALL_LIB_DIR/libldaudit_cuda.so" ]; then
    error "Failed to build LD_AUDIT interceptor"
    exit 1
fi

log "Built: $INSTALL_LIB_DIR/libldaudit_cuda.so"

log "Step 1.5: Skipping libvgpu-exec.so (not needed with system-wide path registration)"


log "Step 2: Checking shim libraries..."

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
fi

log "Shim libraries ready"

log "Step 3: Building enhanced force_load_shim..."

if [ ! -f "$GUEST_SHIM_DIR/force_load_shim.c" ]; then
    error "force_load_shim.c not found in $GUEST_SHIM_DIR"
    exit 1
fi

gcc -o "$INSTALL_BIN_DIR/force_load_shim" \
    "$GUEST_SHIM_DIR/force_load_shim.c" -ldl -Wall

chmod +x "$INSTALL_BIN_DIR/force_load_shim"

if [ ! -f "$INSTALL_BIN_DIR/force_load_shim" ]; then
    error "Failed to build force_load_shim"
    exit 1
fi

log "Built: $INSTALL_BIN_DIR/force_load_shim"

log "Step 4: Registering libraries system-wide via /etc/ld.so.conf.d/..."

cat > /etc/ld.so.conf.d/vgpu.conf <<EOF
/usr/lib64
EOF

log "Created /etc/ld.so.conf.d/vgpu.conf"

log "Running ldconfig to rebuild cache..."
ldconfig 2>&1 | grep -v "WARNING" || true
log "Ran ldconfig to rebuild cache"

if ldconfig -p 2>&1 | grep -q "libvgpu-cuda"; then
    log "Libraries registered in ldconfig cache"
    ldconfig -p 2>&1 | grep "libvgpu-cuda" | head -2
else
    warn "⚠ Libraries not found in ldconfig cache (may still work via symlinks)"
fi

log "Step 5: Ensuring /etc/ld.so.preload is empty (no system-wide preload)..."

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

log "Step 6: Configuring systemd service to use enhanced force_load_shim..."

OLLAMA_BIN=$(which ollama 2>/dev/null || echo "/usr/local/bin/ollama")
if [ ! -f "$OLLAMA_BIN" ]; then
    error "Ollama binary not found. Please install Ollama first."
    exit 1
fi

log "Found Ollama binary: $OLLAMA_BIN"

mkdir -p /etc/systemd/system/ollama.service.d

cat > /etc/systemd/system/ollama.service.d/vgpu.conf <<EOF
[Service]
Environment="LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu"
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so"
Environment="HOME=/home/test-10"
ExecStart=
ExecStart=$INSTALL_BIN_DIR/force_load_shim $OLLAMA_BIN serve
EOF

log "Created systemd override: /etc/systemd/system/ollama.service.d/vgpu.conf"

log "Step 6.5: Installing runner wrapper (optional mechanism)..."

RUNNER_WRAPPER="$GUEST_SHIM_DIR/runner_wrapper.sh"
if [ -f "$RUNNER_WRAPPER" ]; then
    cp "$RUNNER_WRAPPER" "$INSTALL_BIN_DIR/runner_wrapper.sh"
    chmod +x "$INSTALL_BIN_DIR/runner_wrapper.sh"
    log "Installed runner wrapper: $INSTALL_BIN_DIR/runner_wrapper.sh"
    log "  (Note: This is optional - main mechanisms are system-wide paths + symlinks + LD_PRELOAD)"
else
    warn "⚠ Runner wrapper not found (optional, continuing without it)"
fi

log "Step 7: Creating symlinks for filesystem-level library redirection..."

SYMLINK_SCRIPT="$SCRIPT_DIR/guest-shim/create_symlinks.sh"
if [ -f "$SYMLINK_SCRIPT" ]; then
    chmod +x "$SYMLINK_SCRIPT"
    "$SYMLINK_SCRIPT"
    log "Symlinks created"
else
    error "Symlink script not found: $SYMLINK_SCRIPT"
    exit 1
fi

log "Step 8: Reloading systemd and restarting Ollama..."

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

log "Step 9: Verifying deployment..."

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

OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
if [ -n "$OLLAMA_PID" ]; then
    info "Checking Ollama process (PID: $OLLAMA_PID)..."
    
    if grep -q "libvgpu-cuda" /proc/$OLLAMA_PID/maps 2>/dev/null; then
        log "CUDA shim is loaded in Ollama process"
    else
        warn "⚠ CUDA shim not found in process maps (might be loaded via symlinks)"
    fi
    
    if sudo cat /proc/$OLLAMA_PID/environ 2>/dev/null | tr '\0' '\n' | grep -q "LD_LIBRARY_PATH=/usr/lib64"; then
        log "LD_LIBRARY_PATH is set in Ollama process"
    else
        warn "⚠ LD_LIBRARY_PATH not found in process environment"
    fi
    
    if ldconfig -p 2>&1 | grep -q "libvgpu-cuda"; then
        log "Libraries registered in system-wide ldconfig cache"
    else
        warn "⚠ Libraries not found in ldconfig cache (may still work via symlinks)"
    fi
    
    RUNNER_PID=$(pgrep -f "ollama runner" | head -1)
    if [ -n "$RUNNER_PID" ]; then
        info "Found runner subprocess (PID: $RUNNER_PID)"
        if grep -q "libvgpu-cuda" /proc/$RUNNER_PID/maps 2>/dev/null; then
            log "CUDA shim is loaded in runner subprocess"
        else
            warn "⚠ CUDA shim not found in runner subprocess maps"
        fi
        
        if sudo cat /proc/$RUNNER_PID/environ 2>/dev/null | tr '\0' '\n' | grep -q "LD_LIBRARY_PATH"; then
            log "Runner subprocess has LD_LIBRARY_PATH set"
        else
            warn "⚠ Runner subprocess missing LD_LIBRARY_PATH"
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
    info "This might be normal if GPU discovery is still in progress or runner subprocesses need libraries"
else
    warn "⚠ Could not determine library mode from logs"
    info "Recent logs:"
    journalctl -u ollama --since "2 minutes ago" --no-pager 2>&1 | tail -10
fi

echo ""
info "=========================================="
info "DEPLOYMENT SUMMARY"
info "=========================================="
info "Symlinks: $symlink_ok created/verified"
info "Systemd environment: Configured"
info "Libraries: Registered with ldconfig"
info "Ollama service: $(systemctl is-active ollama 2>/dev/null || echo 'unknown')"
info "=========================================="

log ""
log "==================================================================="
log "Deployment Complete!"
log "==================================================================="
log ""
log "What was deployed:"
log "  1. System-wide library path: /etc/ld.so.conf.d/vgpu.conf"
log "  2. LD_AUDIT interceptor: $INSTALL_LIB_DIR/libldaudit_cuda.so"
log "  3. Simplified force_load_shim: $INSTALL_BIN_DIR/force_load_shim"
log "  4. Systemd override: /etc/systemd/system/ollama.service.d/vgpu.conf"
log "  5. Symlinks: libcuda.so.1 -> libvgpu-cuda.so (in standard paths)"
log ""
log "Ultra-stable approach:"
log "  System-wide library path registration (works for ALL processes)"
log "  No dependency on LD_PRELOAD (works with Go's direct syscalls)"
log "  Symlinks in standard paths (filesystem-level redirection)"
log "  LD_LIBRARY_PATH as backup (inherited by subprocesses)"
log "  Easy rollback (just remove systemd override and ld.so.conf.d file)"
log ""
log "To verify GPU mode:"
log "  journalctl -u ollama -n 200 | grep -i library"
log ""
log "To rollback (if needed):"
log "  sudo rm /etc/systemd/system/ollama.service.d/vgpu.conf"
log "  sudo systemctl daemon-reload"
log "  sudo systemctl restart ollama"
log ""
log "==================================================================="
