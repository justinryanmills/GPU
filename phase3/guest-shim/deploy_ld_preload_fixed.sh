#!/bin/bash
# deploy_ld_preload_fixed.sh
# FIXED: Properly handles /etc/ld.so.preload when it doesn't exist
# This script fixes the "cannot stat '/etc/ld.so.preload': No such file or directory" error

set -euo pipefail

LIB_PATH="/usr/lib64/libvgpu-cuda.so"
PRELOAD_FILE="/etc/ld.so.preload"
PASSWORD="${1:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Check if library exists
if [ ! -f "$LIB_PATH" ]; then
    error "Library not found at $LIB_PATH"
fi

success "Library found at $LIB_PATH"

# Step 1: Check if /etc/ld.so.preload exists
echo ""
echo "Step 1: Checking if /etc/ld.so.preload exists..."
if [ -f "$PRELOAD_FILE" ]; then
    FILE_EXISTS=1
    success "File exists - will backup before deployment"
else
    FILE_EXISTS=0
    success "File does not exist (normal) - will create new file"
fi

# Step 2: Backup ONLY if file exists
if [ "$FILE_EXISTS" -eq 1 ]; then
    echo ""
    echo "Step 2: Backing up existing /etc/ld.so.preload..."
    BACKUP_FILE="${PRELOAD_FILE}.backup.$(date +%s)"
    
    if [ -n "$PASSWORD" ]; then
        echo "$PASSWORD" | sudo -S cp "$PRELOAD_FILE" "$BACKUP_FILE" 2>&1 || error "Backup failed"
    else
        sudo cp "$PRELOAD_FILE" "$BACKUP_FILE" 2>&1 || error "Backup failed"
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        error "Backup file was not created"
    fi
    
    success "Backup created: $BACKUP_FILE"
else
    echo ""
    echo "Step 2: Skipping backup (file doesn't exist)"
fi

# Step 3: Deploy - this creates the file if it doesn't exist
echo ""
echo "Step 3: Deploying to /etc/ld.so.preload..."
# Use '>' to create/overwrite the file (works even if file doesn't exist)
if [ -n "$PASSWORD" ]; then
    echo "$PASSWORD" | sudo -S bash -c "echo '$LIB_PATH' > $PRELOAD_FILE" 2>&1 || error "Deployment failed"
else
    sudo bash -c "echo '$LIB_PATH' > $PRELOAD_FILE" 2>&1 || error "Deployment failed"
fi

# Step 4: Verify deployment
echo ""
echo "Step 4: Verifying deployment..."
if [ ! -f "$PRELOAD_FILE" ]; then
    error "File was not created"
fi

if ! grep -q "^${LIB_PATH}$" "$PRELOAD_FILE"; then
    error "Library path not found in preload file"
fi

success "Deployment verified - library is in /etc/ld.so.preload"

# Step 5: Test system processes immediately
echo ""
echo "Step 5: Testing system processes (safety check)..."
test_commands=("cat /dev/null" "ls /tmp | head -1" "echo test" "pwd")

for cmd in "${test_commands[@]}"; do
    if ! eval "$cmd" > /dev/null 2>&1; then
        error "System process failed: $cmd"
    fi
done

success "All system processes working"

# Step 6: Test critical processes
echo ""
echo "Step 6: Testing critical system processes..."
if ! ssh -V > /dev/null 2>&1; then
    error "SSH command failed"
fi

if ! systemctl --version > /dev/null 2>&1; then
    error "systemctl command failed"
fi

success "Critical system processes working"

# Final summary
echo ""
echo "=========================================="
success "Deployment completed successfully!"
echo "=========================================="
echo "Library: $LIB_PATH"
echo "Preload: $PRELOAD_FILE"
if [ "$FILE_EXISTS" -eq 1 ]; then
    echo "Backup: $BACKUP_FILE"
fi
echo "=========================================="

exit 0
