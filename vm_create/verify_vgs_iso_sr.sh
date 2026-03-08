#!/bin/bash

set -euo pipefail

SR_NAME="VGS ISO Storage"

echo "========================================================================"
echo "  Verifying VGS ISO Storage SR Status"
echo "========================================================================"
echo ""

SR_UUID=$(xe sr-list name-label="$SR_NAME" params=uuid --minimal 2>/dev/null | head -1)

if [ -z "$SR_UUID" ]; then
    echo "ERROR: SR '$SR_NAME' not found!"
    echo ""
    echo "Available SRs:"
    xe sr-list params=name-label,uuid,type,content-type | grep -i iso || echo "No ISO SRs found"
    exit 1
fi

echo "Found SR: $SR_NAME"
echo "UUID: $SR_UUID"
echo ""

echo "SR Details:"
echo "-----------"
xe sr-list uuid="$SR_UUID" params=name-label,type,content-type,shared
echo ""

echo "PBD Status:"
echo "-----------"
PBD_UUID=$(xe pbd-list sr-uuid="$SR_UUID" params=uuid --minimal | head -1)
if [ -z "$PBD_UUID" ]; then
    echo "ERROR: No PBD found for SR"
    exit 1
fi

echo "PBD UUID: $PBD_UUID"
ATTACHED=$(xe pbd-param-get uuid="$PBD_UUID" param-name=currently-attached 2>/dev/null || echo "false")
echo "Currently attached: $ATTACHED"
echo ""

SR_MOUNT="/var/run/sr-mount/$SR_UUID"
echo "Mount Point: $SR_MOUNT"
if [ -d "$SR_MOUNT" ]; then
    echo "Directory exists"
else
    echo "✗ Directory does not exist"
fi

if mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
    echo "Mounted in mount table"
else
    echo "✗ NOT mounted in mount table"
fi
echo ""

if [ -d "$SR_MOUNT" ]; then
    FILE_COUNT=$(ls -la "$SR_MOUNT" 2>/dev/null | grep -v "^total" | grep -v "^\.$" | grep -v "^\.\.$" | wc -l)
    echo "Files in mount: $FILE_COUNT"
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo "Files are readable"
        echo ""
        echo "Files found:"
        ls -lh "$SR_MOUNT" | head -10
    else
        echo "✗ No files readable (empty or corrupted)"
    fi
fi
echo ""

echo "========================================================================"
echo "  Summary"
echo "========================================================================"
if [ "$ATTACHED" = "true" ] && [ -d "$SR_MOUNT" ] && mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
    echo "SR is accessible and ready to use"
    exit 0
else
    echo "✗ SR is NOT accessible"
    echo ""
    echo "Attempting to fix..."
    echo ""
    
    if [ "$ATTACHED" != "true" ]; then
        echo "Attaching PBD..."
        xe pbd-plug uuid="$PBD_UUID" || {
            echo "ERROR: Failed to attach PBD"
            echo ""
            echo "PBD device-config:"
            xe pbd-param-get uuid="$PBD_UUID" param-name=device-config
            exit 1
        }
        echo "PBD attached"
        sleep 2
    fi
    
    if [ -d "$SR_MOUNT" ] && mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
        echo "SR is now accessible"
        exit 0
    else
        echo "✗ SR still not accessible after PBD attach"
        echo ""
        echo "PBD device-config:"
        xe pbd-param-get uuid="$PBD_UUID" param-name=device-config
        exit 1
    fi
fi
