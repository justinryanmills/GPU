#!/bin/bash

set -euo pipefail

SR_NAME="VGS ISO Storage"
MOUNT_POINT="/mnt/iso-storage"

echo "========================================================================"
echo "  Creating VGS ISO Storage Repository via CLI"
echo "========================================================================"
echo ""

EXISTING_SR=$(xe sr-list name-label="$SR_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -n "$EXISTING_SR" ]; then
    echo "SR '$SR_NAME' already exists (UUID: $EXISTING_SR)"
    echo "Checking if it's accessible..."
    
    if [ -d "/var/run/sr-mount/$EXISTING_SR" ]; then
        echo "SR mount point exists"
        if mount | grep -F "/var/run/sr-mount/$EXISTING_SR" >/dev/null 2>&1; then
            echo "SR is mounted and accessible"
            echo ""
            echo "SR is ready to use!"
            exit 0
        else
            echo "⚠ SR exists but not mounted. Attempting to mount..."
        fi
    fi
fi

echo "STEP 1: Ensuring VGS volume is mounted..."
echo "------------------------------------------"

LV_LINE=$(lvs --noheadings -o vg_name,lv_name 2>/dev/null | grep -i iso_download | head -1)
if [ -z "$LV_LINE" ]; then
    echo "ERROR: Could not find iso_download volume"
    exit 1
fi

VG_NAME=$(echo "$LV_LINE" | awk '{print $1}')
LV_PATH="/dev/$VG_NAME/iso_download"

if [ ! -e "$LV_PATH" ]; then
    echo "Activating logical volume..."
    lvchange -ay "$VG_NAME/iso_download"
    sleep 1
fi

mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
    mount "$LV_PATH" "$MOUNT_POINT"
    echo "VGS volume mounted at $MOUNT_POINT"
else
    echo "VGS volume already mounted"
fi

if [ ! -r "$MOUNT_POINT" ]; then
    echo "ERROR: Cannot read from $MOUNT_POINT"
    exit 1
fi

echo ""

echo "STEP 2: Creating file-based ISO SR..."
echo "--------------------------------------"

echo "Attempting to create SR using xe sr-create..."

HOST_UUID=$(xe host-list --minimal | head -1)
if [ -z "$HOST_UUID" ]; then
    echo "ERROR: Could not get host UUID"
    exit 1
fi

SR_UUID=$(xe sr-create \
    name-label="$SR_NAME" \
    type=file \
    device-config:location="$MOUNT_POINT" \
    content-type=iso \
    shared=false \
    2>&1) || {
    
    echo "Method 1 failed, trying alternative method..."
    
    
    echo "Attempting alternative: creating as 'other' type and converting..."
    
    SR_UUID=$(xe sr-create \
        name-label="$SR_NAME" \
        type=iso \
        device-config:location="$MOUNT_POINT" \
        shared=false \
        2>&1) || {
        
        echo ""
        echo "ERROR: Could not create ISO SR via CLI"
        echo ""
        echo "XCP-ng CLI does not easily support creating file-based ISO SRs."
        echo "The standard method requires XCP-ng Center GUI."
        echo ""
        echo "ALTERNATIVE SOLUTION:"
        echo "-------------------"
        echo "Instead of creating a new ISO SR, we can modify create_test3_vm.sh"
        echo "to work with the VGS-mounted ISO directly by:"
        echo "  1. Using an existing accessible ISO SR (if any)"
        echo "  2. Or, copying ISO to an existing SR that IS accessible"
        echo ""
        echo "Would you like me to create a modified script that:"
        echo "  - Uses the SMB SR if we can make it accessible, OR"
        echo "  - Creates a workaround that doesn't require a new ISO SR?"
        exit 1
    }
}

echo "SR created: $SR_UUID"

echo ""
echo "STEP 3: Creating and attaching PBD..."
echo "--------------------------------------"

PBD_UUID=$(xe pbd-create sr-uuid="$SR_UUID" host-uuid="$HOST_UUID" device-config:location="$MOUNT_POINT" 2>&1) || {
    echo "ERROR: Could not create PBD"
    echo "Attempting to use existing PBD..."
    PBD_UUID=$(xe pbd-list sr-uuid="$SR_UUID" params=uuid --minimal | head -1)
}

if [ -z "$PBD_UUID" ]; then
    echo "ERROR: No PBD found for SR"
    exit 1
fi

echo "PBD UUID: $PBD_UUID"

echo "Plugging PBD..."
xe pbd-plug uuid="$PBD_UUID" || {
    echo "WARNING: PBD plug failed, but continuing..."
}

sleep 2

echo "Scanning SR..."
xe sr-scan uuid="$SR_UUID" || true

sleep 2

echo ""
echo "STEP 4: Verifying SR is accessible..."
echo "--------------------------------------"

SR_MOUNT="/var/run/sr-mount/$SR_UUID"
if [ -d "$SR_MOUNT" ]; then
    echo "Mount point exists: $SR_MOUNT"
    
    if mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
        echo "SR is mounted"
        
        FILE_COUNT=$(ls -la "$SR_MOUNT" 2>/dev/null | grep -v "^total" | grep -v "^\.$" | grep -v "^\.\.$" | wc -l)
        if [ "$FILE_COUNT" -gt 0 ]; then
            echo "Files are readable ($FILE_COUNT items)"
            echo ""
            echo "========================================================================"
            echo "  SUCCESS: VGS ISO Storage Repository is ready!"
            echo "========================================================================"
            echo "SR UUID: $SR_UUID"
            echo "SR Name: $SR_NAME"
            echo "Mount Point: $SR_MOUNT"
            echo ""
            echo "You can now run: bash create_test3_vm.sh"
            exit 0
        else
            echo "⚠ Mount exists but no files visible"
        fi
    else
        echo "⚠ Mount point exists but not in mount table"
    fi
else
    echo "⚠ Mount point does not exist yet"
fi

echo ""
echo "SR was created but may need manual verification."
echo "Check: xe sr-list name-label='$SR_NAME' params=uuid,name-label,type"
echo "Mount: ls -la /var/run/sr-mount/$SR_UUID"
