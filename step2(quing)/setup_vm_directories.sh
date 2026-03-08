#!/bin/bash

VGPU_BASE="/var/vgpu"

if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run as root (use sudo)"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "No VM IDs specified. Creating directories for common IDs: 1, 2, 100, 200"
    VM_IDS=(1 2 100 200)
else
    VM_IDS=("$@")
fi

echo "=================================================================================="
echo "          Setting Up Per-VM Directories for GPU Mediation"
echo "=================================================================================="
echo ""

if [ ! -d "$VGPU_BASE" ]; then
    echo "[CREATE] Creating base directory: $VGPU_BASE"
    mkdir -p "$VGPU_BASE"
    chmod 777 "$VGPU_BASE"
fi

for VM_ID in "${VM_IDS[@]}"; do
    VM_DIR="$VGPU_BASE/vm$VM_ID"
    REQUEST_FILE="$VM_DIR/request.txt"
    RESPONSE_FILE="$VM_DIR/response.txt"
    
    echo "[VM $VM_ID] Setting up directory..."
    
    if [ ! -d "$VM_DIR" ]; then
        mkdir -p "$VM_DIR"
        echo "  Created: $VM_DIR"
    else
        echo "  ℹ️  Already exists: $VM_DIR"
    fi
    
    chmod 777 "$VM_DIR"
    echo "  Permissions set: 777"
    
    touch "$REQUEST_FILE"
    chmod 666 "$REQUEST_FILE"
    echo "  Request file ready: $REQUEST_FILE"
    
    echo "0:Ready" > "$RESPONSE_FILE"
    chmod 666 "$RESPONSE_FILE"
    echo "  Response file initialized: $RESPONSE_FILE"
    
    echo ""
done

echo "=================================================================================="
echo "          Setup Complete!"
echo "=================================================================================="
echo ""
echo "Created directories:"
for VM_ID in "${VM_IDS[@]}"; do
    echo "  - $VGPU_BASE/vm$VM_ID/"
done
echo ""
echo "Next steps:"
echo "  1. Ensure NFS is exporting $VGPU_BASE"
echo "  2. VMs should mount: mount -t nfs <host-ip>:/var/vgpu /mnt/vgpu"
echo "  3. VMs can now send requests to /mnt/vgpu/vm<id>/request.txt"
echo ""
