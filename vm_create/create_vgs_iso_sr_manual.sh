#!/bin/bash

set -euo pipefail

SR_NAME="VGS ISO Storage"
MOUNT_POINT="/mnt/iso-storage"

echo "========================================================================"
echo "  Manual ISO SR Creation (Advanced Method)"
echo "========================================================================"
echo ""

LV_LINE=$(lvs --noheadings -o vg_name,lv_name 2>/dev/null | grep -i iso_download | head -1)
VG_NAME=$(echo "$LV_LINE" | awk '{print $1}')
LV_PATH="/dev/$VG_NAME/iso_download"
mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
    mount "$LV_PATH" "$MOUNT_POINT"
fi

HOST_UUID=$(xe host-list --minimal | head -1)

echo "Attempting Method 1: xe sr-introduce..."
echo "----------------------------------------"


echo "Trying 'other' type with file location..."
SR_UUID=$(xe sr-create \
    name-label="$SR_NAME" \
    type=other \
    device-config:location="$MOUNT_POINT" \
    content-type=iso \
    shared=false \
    2>&1) && {
    echo "Created as 'other' type: $SR_UUID"
    echo "Attempting to convert to ISO SR..."
    xe sr-param-set uuid="$SR_UUID" content-type=iso 2>/dev/null || true
} || {
    echo "Method 1 failed"
}

if [ -z "${SR_UUID:-}" ]; then
    echo ""
    echo "Attempting Method 2: Direct ISO type creation..."
    echo "------------------------------------------------"
    
    for SR_TYPE in "iso" "file" "ext"; do
        echo "Trying type: $SR_TYPE"
        SR_UUID=$(xe sr-create \
            name-label="$SR_NAME" \
            type="$SR_TYPE" \
            device-config:location="$MOUNT_POINT" \
            content-type=iso \
            shared=false \
            2>&1) && {
            echo "Success with type $SR_TYPE: $SR_UUID"
            break
        } || {
            echo "  Failed with type $SR_TYPE"
            SR_UUID=""
        }
    done
fi

if [ -z "${SR_UUID:-}" ]; then
    echo ""
    echo "Attempting Method 3: NFS-style (local path)..."
    echo "---------------------------------------------"
    
    SR_UUID=$(xe sr-create \
        name-label="$SR_NAME" \
        type=nfs \
        device-config:serverpath="$MOUNT_POINT" \
        device-config:server="localhost" \
        content-type=iso \
        shared=false \
        2>&1) && {
        echo "Created as NFS type: $SR_UUID"
    } || {
        echo "Method 3 failed"
        SR_UUID=""
    }
fi

if [ -n "${SR_UUID:-}" ]; then
    echo ""
    echo "SR created: $SR_UUID"
    echo "Setting up PBD..."
    
    PBD_UUID=$(xe pbd-create sr-uuid="$SR_UUID" host-uuid="$HOST_UUID" device-config:location="$MOUNT_POINT" 2>&1) || {
        PBD_UUID=$(xe pbd-list sr-uuid="$SR_UUID" params=uuid --minimal | head -1)
    }
    
    if [ -n "$PBD_UUID" ]; then
        echo "PBD: $PBD_UUID"
        xe pbd-plug uuid="$PBD_UUID" 2>&1 || true
        sleep 2
        xe sr-scan uuid="$SR_UUID" 2>&1 || true
        sleep 2
        
        SR_MOUNT="/var/run/sr-mount/$SR_UUID"
        if [ -d "$SR_MOUNT" ]; then
            echo "Mount point created: $SR_MOUNT"
            if mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
                echo "SR is mounted and accessible!"
                echo ""
                echo "SUCCESS! You can now run: bash create_test3_vm.sh"
                exit 0
            fi
        fi
    fi
fi

echo ""
echo "========================================================================"
echo "  All CLI methods failed"
echo "========================================================================"
echo ""
echo "XCP-ng does not support creating file-based ISO SRs via CLI."
echo ""
echo "FINAL WORKAROUND OPTION:"
echo "-----------------------"
echo "Since we can't create a new ISO SR, we have one last option:"
echo ""
echo "1. The ISO file is at: $MOUNT_POINT/ubuntu-*.iso"
echo "2. We could try to use 'xe-import' or other tools to register it"
echo "3. OR, we need to fix the SMB server (10.25.33.33) connectivity"
echo ""
echo "Would you like me to:"
echo "  A) Create a script that attempts to register the ISO file directly"
echo "     without using an ISO SR (advanced, may not work)"
echo "  B) Help diagnose/fix the SMB server connectivity issue"
echo "  C) Document the exact steps needed once you can access XCP-ng Center"
echo ""
