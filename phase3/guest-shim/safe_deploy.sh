#!/bin/bash
# Safe deployment script with error checking and recovery

set -e  # Exit on error

# Configuration
SHIM_DIR="$HOME/phase3/guest-shim"
SHIM_LIB="/usr/lib64/libvgpu-cuda.so"
BACKUP_LIB="/usr/lib64/libvgpu-cuda.so.backup"
PRELOAD_FILE="/etc/ld.so.preload"
SERVICE_NAME="ollama"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check disk space (need at least 100MB free)
    AVAIL_SPACE=$(df -m "$HOME" | tail -1 | awk '{print $4}')
    if [ "$AVAIL_SPACE" -lt 100 ]; then
        log_error "Insufficient disk space: ${AVAIL_SPACE}MB available (need at least 100MB)"
        exit 1
    fi
    log_info "Disk space OK: ${AVAIL_SPACE}MB available"
    
    # Check if source directory exists
    if [ ! -d "$SHIM_DIR" ]; then
        log_error "Source directory not found: $SHIM_DIR"
        exit 1
    fi
    
    # Check if source files exist
    if [ ! -f "$SHIM_DIR/libvgpu_cuda.c" ]; then
        log_error "Source file not found: $SHIM_DIR/libvgpu_cuda.c"
        exit 1
    fi
    
    if [ ! -f "$SHIM_DIR/cuda_transport.c" ]; then
        log_error "Source file not found: $SHIM_DIR/cuda_transport.c"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Backup existing library
backup_library() {
    log_info "Backing up existing library..."
    if [ -f "$SHIM_LIB" ]; then
        sudo cp "$SHIM_LIB" "$BACKUP_LIB"
        log_info "Backed up to: $BACKUP_LIB"
    else
        log_warn "No existing library to backup"
    fi
}

# Build shim library
build_shim() {
    log_info "Building shim library..."
    cd "$SHIM_DIR" || exit 1
    
    # Clean any previous build artifacts
    rm -f /tmp/libvgpu-cuda-build.log
    
    # Build with error capture
    if sudo gcc -shared -fPIC -o "$SHIM_LIB" \
        libvgpu_cuda.c cuda_transport.c \
        -I../include -I. -ldl -lpthread -O2 -Wall \
        2>&1 | tee /tmp/libvgpu-cuda-build.log; then
        log_info "Build successful"
        
        # Verify the library was created and is valid
        if [ ! -f "$SHIM_LIB" ]; then
            log_error "Build succeeded but library file not found"
            restore_backup
            exit 1
        fi
        
        # Check if it's a valid shared library
        if ! file "$SHIM_LIB" | grep -q "shared object"; then
            log_error "Built file is not a valid shared library"
            restore_backup
            exit 1
        fi
        
        log_info "Library verified: $(ls -lh "$SHIM_LIB" | awk '{print $5}')"
    else
        log_error "Build failed - check /tmp/libvgpu-cuda-build.log"
        restore_backup
        exit 1
    fi
}

# Restore backup if something goes wrong
restore_backup() {
    if [ -f "$BACKUP_LIB" ]; then
        log_warn "Restoring backup library..."
        sudo cp "$BACKUP_LIB" "$SHIM_LIB"
        log_info "Backup restored"
    fi
}

# Configure ld.so.preload safely
configure_preload() {
    log_info "Configuring /etc/ld.so.preload..."
    
    # Check if already configured
    if grep -q "libvgpu-cuda.so" "$PRELOAD_FILE" 2>/dev/null; then
        log_info "Already in $PRELOAD_FILE"
        return
    fi
    
    # Add to preload (create file if it doesn't exist)
    echo "$SHIM_LIB" | sudo tee -a "$PRELOAD_FILE" > /dev/null
    log_info "Added to $PRELOAD_FILE"
}

# Safely restart Ollama service
restart_service() {
    log_info "Restarting $SERVICE_NAME service..."
    
    # Check if service is running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Service is running, stopping..."
        
        # Stop with timeout
        if timeout 30 sudo systemctl stop "$SERVICE_NAME"; then
            log_info "Service stopped"
        else
            log_error "Failed to stop service (timeout)"
            # Try to kill if it's hung
            SERVICE_PID=$(systemctl show --property MainPID --value "$SERVICE_NAME" 2>/dev/null || echo "")
            if [ -n "$SERVICE_PID" ] && [ "$SERVICE_PID" != "0" ]; then
                log_warn "Attempting to kill hung process: $SERVICE_PID"
                sudo kill -TERM "$SERVICE_PID" 2>/dev/null || true
                sleep 2
                sudo kill -KILL "$SERVICE_PID" 2>/dev/null || true
            fi
        fi
        
        # Wait for service to fully stop
        sleep 2
        
        # Verify it's stopped
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            log_error "Service is still running after stop command"
            exit 1
        fi
    else
        log_info "Service is not running"
    fi
    
    # Start service
    log_info "Starting service..."
    if sudo systemctl start "$SERVICE_NAME"; then
        log_info "Service start command issued"
    else
        log_error "Failed to start service"
        restore_backup
        exit 1
    fi
    
    # Wait for service to start
    log_info "Waiting for service to start..."
    sleep 8
    
    # Verify service is running
    MAX_WAIT=30
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            log_info "Service is running"
            return 0
        fi
        sleep 1
        WAITED=$((WAITED + 1))
    done
    
    log_error "Service failed to start within ${MAX_WAIT} seconds"
    sudo systemctl status "$SERVICE_NAME" --no-pager -l | head -20
    restore_backup
    exit 1
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check if service is running
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        log_error "Service is not running"
        return 1
    fi
    
    # Check if shim is loaded
    SERVICE_PID=$(pgrep -f "$SERVICE_NAME serve" | head -1)
    if [ -z "$SERVICE_PID" ]; then
        log_warn "Could not find service process"
        return 1
    fi
    
    log_info "Service PID: $SERVICE_PID"
    
    # Check for shim log
    SHIM_LOG="/tmp/vgpu-shim-cuda-${SERVICE_PID}.log"
    if [ -f "$SHIM_LOG" ]; then
        log_info "Shim log found:"
        tail -10 "$SHIM_LOG" | sed 's/^/  /'
        
        if grep -q "Pre-initialization succeeded" "$SHIM_LOG"; then
            log_info "✓ cuInit pre-initialization successful"
        else
            log_warn "Pre-initialization message not found in log"
        fi
    else
        log_warn "Shim log not found: $SHIM_LOG"
    fi
    
    # Check loaded libraries
    if sudo cat /proc/$SERVICE_PID/maps 2>/dev/null | grep -q "libvgpu-cuda"; then
        log_info "✓ Shim library is loaded in process"
    else
        log_warn "Shim library not found in process maps"
    fi
    
    return 0
}

# Main deployment flow
main() {
    echo "=========================================="
    echo "Safe VGPU CUDA Shim Deployment"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    backup_library
    build_shim
    configure_preload
    restart_service
    verify_deployment
    
    echo ""
    echo "=========================================="
    log_info "Deployment completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run a test inference: ollama run llama3.2:1b 'test'"
    echo "2. Check library mode: sudo journalctl -u ollama --since '1 minute ago' | grep library="
    echo ""
}

# Run main function
main "$@"
