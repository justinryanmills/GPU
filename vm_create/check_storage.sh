#!/bin/bash

set -euo pipefail

echo "========================================================================"
echo "  Storage and VM Status Check"
echo "========================================================================"
echo ""

echo "STEP 1: Checking Local Storage Repository..."
echo "----------------------------------------------"
LOCAL_SR_UUID=$(xe sr-list type=lvm content-type=user params=uuid --minimal | head -1)
if [ -z "$LOCAL_SR_UUID" ]; then
    echo "✗ ERROR: No local storage SR found"
    exit 1
fi

LOCAL_SR_NAME=$(xe sr-list uuid="$LOCAL_SR_UUID" params=name-label --minimal)
echo "SR Name: $LOCAL_SR_NAME"
echo "SR UUID: $LOCAL_SR_UUID"

TOTAL_SPACE=$(xe sr-param-get uuid="$LOCAL_SR_UUID" param-name=physical-size 2>/dev/null || echo "0")
USED_SPACE=$(xe sr-param-get uuid="$LOCAL_SR_UUID" param-name=physical-utilisation 2>/dev/null || echo "0")
FREE_SPACE=$((TOTAL_SPACE - USED_SPACE))

TOTAL_GB=$((TOTAL_SPACE / 1024 / 1024 / 1024))
USED_GB=$((USED_SPACE / 1024 / 1024 / 1024))
FREE_GB=$((FREE_SPACE / 1024 / 1024 / 1024))

echo "  Total space: ${TOTAL_GB} GiB"
echo "  Used space:  ${USED_GB} GiB"
echo "  Free space:  ${FREE_GB} GiB"
echo ""

echo "STEP 2: Checking for existing Test-3 VM..."
echo "-------------------------------------------"
TEST3_UUID=$(xe vm-list name-label="Test-3" params=uuid --minimal 2>/dev/null | head -1)
if [ -n "$TEST3_UUID" ]; then
    echo "⚠ WARNING: Test-3 VM already exists!"
    echo "  UUID: $TEST3_UUID"
    
    POWER_STATE=$(xe vm-param-get uuid="$TEST3_UUID" param-name=power-state 2>/dev/null || echo "unknown")
    echo "  Power state: $POWER_STATE"
    
    DISK_VBD=$(xe vbd-list vm-uuid="$TEST3_UUID" type=Disk params=uuid --minimal | head -1)
    if [ -n "$DISK_VBD" ]; then
        DISK_VDI=$(xe vbd-param-get uuid="$DISK_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
        if [ -n "$DISK_VDI" ]; then
            DISK_SIZE=$(xe vdi-param-get uuid="$DISK_VDI" param-name=virtual-size 2>/dev/null || echo "0")
            DISK_GB=$((DISK_SIZE / 1024 / 1024 / 1024))
            echo "  Disk size: ${DISK_GB} GiB"
        fi
    fi
    
    echo ""
    echo "  To delete this VM and free up space:"
    echo "    xe vm-shutdown uuid=$TEST3_UUID force=true"
    echo "    xe vm-destroy uuid=$TEST3_UUID"
    echo "    xe vm-uninstall uuid=$TEST3_UUID force=true"
    echo ""
else
    echo "No existing Test-3 VM found"
    echo ""
fi

echo "STEP 3: Checking all VMs and disk usage..."
echo "--------------------------------------------"
ALL_VMS=$(xe vm-list params=uuid,name-label --minimal | grep -v "^$" | head -20)
if [ -n "$ALL_VMS" ]; then
    echo "VMs found:"
    echo "$ALL_VMS" | while IFS=',' read -r VM_UUID VM_NAME; do
        if [ -z "$VM_UUID" ] || [ -z "$VM_NAME" ]; then
            continue
        fi
        if [ "$VM_NAME" = "Control domain on host" ]; then
            continue
        fi
        
        DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
        if [ -n "$DISK_VBD" ]; then
            DISK_VDI=$(xe vbd-param-get uuid="$DISK_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
            if [ -n "$DISK_VDI" ]; then
                DISK_SIZE=$(xe vdi-param-get uuid="$DISK_VDI" param-name=virtual-size 2>/dev/null || echo "0")
                DISK_GB=$((DISK_SIZE / 1024 / 1024 / 1024))
                echo "  $VM_NAME: ${DISK_GB} GiB"
            fi
        fi
    done
else
    echo "  No VMs found"
fi
echo ""

echo "========================================================================"
echo "  Summary"
echo "========================================================================"
echo ""
if [ "$FREE_GB" -lt 40 ]; then
    echo "⚠ WARNING: Insufficient space for 40 GiB disk!"
    echo "  Free space: ${FREE_GB} GiB"
    echo "  Required:   40 GiB"
    echo ""
    echo "Options:"
    echo "  1. Delete existing Test-3 VM (if it exists)"
    echo "  2. Use smaller disk size: bash create_vm.sh Test-3 --disk 20GiB"
    echo "  3. Free up space by deleting unused VMs or VDIs"
else
    echo "Sufficient space available (${FREE_GB} GiB free)"
    if [ -n "$TEST3_UUID" ]; then
        echo "⚠ But Test-3 VM already exists - delete it first!"
    fi
fi
echo ""
