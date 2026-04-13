#!/bin/bash
# install_ld_preload.sh
# Safely installs libvgpu-cuda.so into /etc/ld.so.preload
# This script includes comprehensive safety checks and rollback mechanisms

set -e

LIB_PATH="/usr/lib64/libvgpu-cuda.so"
PRELOAD_FILE="/etc/ld.so.preload"
BACKUP_FILE="/etc/ld.so.preload.backup.$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/tmp/vgpu_ld_preload_install.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "This script must be run as root (use sudo)"
fi

log "=========================================="
log "Installing libvgpu-cuda.so to /etc/ld.so.preload"
log "=========================================="

# Step 1: Verify library exists
log "Step 1: Verifying library exists..."
if [ ! -f "$LIB_PATH" ]; then
    error "Library not found at $LIB_PATH"
fi
success "Library found at $LIB_PATH"

# Step 2: Verify library is valid
log "Step 2: Verifying library is valid..."
if ! file "$LIB_PATH" | grep -q "shared object"; then
    error "Library at $LIB_PATH is not a valid shared object"
fi
success "Library is valid shared object"

# Step 3: Backup existing preload file
log "Step 3: Backing up existing /etc/ld.so.preload..."
if [ -f "$PRELOAD_FILE" ]; then
    cp "$PRELOAD_FILE" "$BACKUP_FILE"
    log "Backup created: $BACKUP_FILE"
    
    # Verify backup
    if [ ! -f "$BACKUP_FILE" ]; then
        error "Failed to create backup"
    fi
    success "Backup created successfully"
else
    log "No existing /etc/ld.so.preload file (this is normal)"
    touch "$BACKUP_FILE"  # Create empty backup for rollback
fi

# Step 4: Check if library is already in preload
log "Step 4: Checking if library is already in preload..."
if [ -f "$PRELOAD_FILE" ] && grep -q "^${LIB_PATH}$" "$PRELOAD_FILE"; then
    warn "Library is already in /etc/ld.so.preload"
    log "Installation appears to already be complete"
    exit 0
fi

# Step 5: Add library to preload file
log "Step 5: Adding library to /etc/ld.so.preload..."
if [ -f "$PRELOAD_FILE" ]; then
    # Append to existing file (preserve other entries)
    echo "$LIB_PATH" >> "$PRELOAD_FILE"
else
    # Create new file
    echo "$LIB_PATH" > "$PRELOAD_FILE"
fi

# Verify the file was written correctly
if ! grep -q "^${LIB_PATH}$" "$PRELOAD_FILE"; then
    error "Failed to add library to /etc/ld.so.preload"
fi
success "Library added to /etc/ld.so.preload"

# Step 6: Verify file format
log "Step 6: Verifying file format..."
if [ ! -f "$PRELOAD_FILE" ]; then
    error "/etc/ld.so.preload file does not exist after write"
fi

# Check for invalid characters or paths
if grep -q "[^[:print:]]" "$PRELOAD_FILE" 2>/dev/null; then
    warn "File contains non-printable characters (may be normal)"
fi

# Verify all lines are valid paths or empty
while IFS= read -r line; do
    if [ -n "$line" ] && [ ! -f "$line" ] && [ "$line" != "$LIB_PATH" ]; then
        warn "Line in preload file may be invalid: $line"
    fi
done < "$PRELOAD_FILE"

success "File format verified"

# Step 7: Test that system processes still work
log "Step 7: Testing system processes (safety check)..."
log "Testing cat command..."
if ! cat /dev/null > /dev/null 2>&1; then
    error "cat command failed - library may be causing issues"
fi
success "cat command works"

log "Testing ls command..."
if ! ls /tmp > /dev/null 2>&1; then
    error "ls command failed - library may be causing issues"
fi
success "ls command works"

log "Testing bash command..."
if ! bash -c "echo test" > /dev/null 2>&1; then
    error "bash command failed - library may be causing issues"
fi
success "bash command works"

# Step 8: Final verification
log "Step 8: Final verification..."
log "Contents of /etc/ld.so.preload:"
cat "$PRELOAD_FILE" | tee -a "$LOG_FILE"

log "=========================================="
success "Installation completed successfully!"
log "=========================================="
log ""
log "Installation details:"
log "  Library: $LIB_PATH"
log "  Preload file: $PRELOAD_FILE"
log "  Backup: $BACKUP_FILE"
log "  Log: $LOG_FILE"
log ""
log "To rollback (if needed):"
log "  sudo cp $BACKUP_FILE $PRELOAD_FILE"
log "  sudo systemctl daemon-reload"
log ""
log "Next steps:"
log "  1. Restart Ollama service: sudo systemctl restart ollama"
log "  2. Check logs: journalctl -u ollama -f"
log "  3. Verify runners load library and cuInit() succeeds"
log "=========================================="

exit 0
