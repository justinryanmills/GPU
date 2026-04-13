#!/bin/bash
# ============================================================================
# Force Load Installation Script
# 
# This script implements multiple injection methods to ensure shims load
# into Ollama, even when LD_PRELOAD fails with Go binaries.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_LIB_DIR="/usr/lib64"
INSTALL_BIN_DIR="/usr/local/bin"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${GREEN}[install]${NC} $*"; }
warn()  { echo -e "${YELLOW}[install]${NC} $*"; }
error() { echo -e "${RED}[install]${NC} $*" >&2; }
info()  { echo -e "${BLUE}[install]${NC} $*"; }

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    error "This script must be run as root (use sudo)"
    exit 1
fi

log "Installing force-load mechanisms for Ollama shim injection..."

# ============================================================================
# Method 1: /etc/ld.so.preload (system-wide, works for all processes)
# ============================================================================
log "Method 1: Setting up /etc/ld.so.preload (system-wide)"

if [ -f "$INSTALL_LIB_DIR/libvgpu-cuda.so" ] && [ -f "$INSTALL_LIB_DIR/libvgpu-nvml.so" ]; then
    # Backup existing preload if it exists
    if [ -f /etc/ld.so.preload ]; then
        cp /etc/ld.so.preload /etc/ld.so.preload.backup.$(date +%Y%m%d_%H%M%S)
        warn "Backed up existing /etc/ld.so.preload"
    fi
    
    # Create new preload file
    cat > /etc/ld.so.preload <<EOF
$INSTALL_LIB_DIR/libvgpu-cuda.so
$INSTALL_LIB_DIR/libvgpu-nvml.so
EOF
    chmod 644 /etc/ld.so.preload
    log "✓ Created /etc/ld.so.preload"
else
    error "Shim libraries not found in $INSTALL_LIB_DIR"
    exit 1
fi

# ============================================================================
# Method 2: Build and install LD_AUDIT interceptor
# ============================================================================
log "Method 2: Building LD_AUDIT interceptor"

if [ -f "$SCRIPT_DIR/ld_audit_interceptor.c" ]; then
    gcc -shared -fPIC -o "$INSTALL_LIB_DIR/libldaudit_cuda.so" \
        "$SCRIPT_DIR/ld_audit_interceptor.c" -ldl -O2 -Wall
    chmod 755 "$INSTALL_LIB_DIR/libldaudit_cuda.so"
    log "✓ Built LD_AUDIT interceptor: $INSTALL_LIB_DIR/libldaudit_cuda.so"
else
    warn "ld_audit_interceptor.c not found, skipping LD_AUDIT method"
fi

# ============================================================================
# Method 3: Build and install force_load_shim wrapper
# ============================================================================
log "Method 3: Building force_load_shim wrapper"

if [ -f "$SCRIPT_DIR/force_load_shim.c" ]; then
    gcc -o "$INSTALL_BIN_DIR/force_load_shim" \
        "$SCRIPT_DIR/force_load_shim.c" -ldl -O2 -Wall
    chmod 755 "$INSTALL_BIN_DIR/force_load_shim"
    log "✓ Built force_load_shim: $INSTALL_BIN_DIR/force_load_shim"
else
    warn "force_load_shim.c not found, skipping wrapper method"
fi

# ============================================================================
# Method 4: Update Ollama systemd service to use force loading
# ============================================================================
log "Method 4: Updating Ollama systemd service"

OLLAMA_SERVICE="/etc/systemd/system/ollama.service"
if [ -f "$OLLAMA_SERVICE" ]; then
    # Backup service file
    cp "$OLLAMA_SERVICE" "${OLLAMA_SERVICE}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Check if we need to modify ExecStart
    if ! grep -q "force_load_shim\|LD_AUDIT\|ld.so.preload" "$OLLAMA_SERVICE"; then
        # Try Method A: Use force_load_shim wrapper
        if [ -f "$INSTALL_BIN_DIR/force_load_shim" ]; then
            sed -i 's|^ExecStart=.*ollama serve|ExecStart='"$INSTALL_BIN_DIR"'/force_load_shim /usr/local/bin/ollama serve|' "$OLLAMA_SERVICE"
            log "✓ Updated Ollama service to use force_load_shim"
        else
            # Method B: Use LD_AUDIT
            if [ -f "$INSTALL_LIB_DIR/libldaudit_cuda.so" ]; then
                # Add Environment line if it doesn't exist
                if ! grep -q "^Environment=" "$OLLAMA_SERVICE"; then
                    sed -i '/^\[Service\]/a Environment="LD_AUDIT='"$INSTALL_LIB_DIR"'/libldaudit_cuda.so"' "$OLLAMA_SERVICE"
                else
                    # Append to existing Environment
                    sed -i 's|^Environment=\(.*\)|Environment=\1 LD_AUDIT='"$INSTALL_LIB_DIR"'/libldaudit_cuda.so|' "$OLLAMA_SERVICE"
                fi
                log "✓ Updated Ollama service to use LD_AUDIT"
            fi
        fi
        
        systemctl daemon-reload
        log "✓ Reloaded systemd daemon"
    else
        info "Ollama service already configured for force loading"
    fi
else
    warn "Ollama service file not found at $OLLAMA_SERVICE"
fi

# ============================================================================
# Method 5: Ensure shim libraries have correct SONAME
# ============================================================================
log "Method 5: Verifying shim library SONAMEs"

for lib in "$INSTALL_LIB_DIR/libvgpu-cuda.so" "$INSTALL_LIB_DIR/libvgpu-nvml.so"; do
    if [ -f "$lib" ]; then
        soname=$(readelf -d "$lib" 2>/dev/null | grep SONAME | awk '{print $5}' || echo "")
        info "  $lib: SONAME=$soname"
        
        # Ensure symlinks exist for common names
        case "$lib" in
            *libvgpu-cuda.so)
                [ ! -L "$INSTALL_LIB_DIR/libcuda.so.1" ] && \
                    ln -sf libvgpu-cuda.so "$INSTALL_LIB_DIR/libcuda.so.1" && \
                    log "  ✓ Created symlink: libcuda.so.1"
                [ ! -L "$INSTALL_LIB_DIR/libcuda.so" ] && \
                    ln -sf libvgpu-cuda.so "$INSTALL_LIB_DIR/libcuda.so" && \
                    log "  ✓ Created symlink: libcuda.so"
                ;;
            *libvgpu-nvml.so)
                [ ! -L "$INSTALL_LIB_DIR/libnvidia-ml.so.1" ] && \
                    ln -sf libvgpu-nvml.so "$INSTALL_LIB_DIR/libnvidia-ml.so.1" && \
                    log "  ✓ Created symlink: libnvidia-ml.so.1"
                [ ! -L "$INSTALL_LIB_DIR/libnvidia-ml.so" ] && \
                    ln -sf libvgpu-nvml.so "$INSTALL_LIB_DIR/libnvidia-ml.so" && \
                    log "  ✓ Created symlink: libnvidia-ml.so"
                ;;
        esac
    fi
done

# Update ldconfig cache
ldconfig
log "✓ Updated dynamic linker cache"

# ============================================================================
# Summary
# ============================================================================
echo ""
log "Installation complete! Multiple injection methods installed:"
echo ""
echo "  1. /etc/ld.so.preload (system-wide, active for all processes)"
echo "  2. LD_AUDIT interceptor: $INSTALL_LIB_DIR/libldaudit_cuda.so"
echo "  3. Force load wrapper: $INSTALL_BIN_DIR/force_load_shim"
echo "  4. Ollama service updated (if found)"
echo "  5. Symlinks and ldconfig updated"
echo ""
warn "IMPORTANT: /etc/ld.so.preload affects ALL processes system-wide."
warn "Test thoroughly before deploying to production."
echo ""
log "To test:"
echo "  sudo systemctl restart ollama"
echo "  journalctl -u ollama -f | grep -i 'libvgpu\|cuda\|gpu'"
echo ""
