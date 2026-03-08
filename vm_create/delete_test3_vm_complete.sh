#!/bin/bash

set -euo pipefail

VM_NAME="Test-3"

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "No existing Test-3 VM found"
    exit 0
fi

echo "========================================================================"
echo "  Deleting Test-3 VM Completely"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo ""

POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
echo "Current power state: $POWER_STATE"
echo ""

DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -n "$DISK_VBD" ]; then
    DISK_VDI=$(xe vbd-param-get uuid="$DISK_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
    if [ -n "$DISK_VDI" ]; then
        DISK_SIZE=$(xe vdi-param-get uuid="$DISK_VDI" param-name=virtual-size 2>/dev/null || echo "0")
        DISK_GB=$((DISK_SIZE / 1024 / 1024 / 1024))
        echo "Disk to be deleted: ${DISK_GB} GiB"
        echo ""
    fi
fi

if [ "$POWER_STATE" = "running" ]; then
    echo "STEP 1: Shutting down VM..."
    echo "----------------------------"
    xe vm-shutdown uuid="$VM_UUID" force=true 2>&1 || xe vm-destroy uuid="$VM_UUID" 2>&1 || true
    sleep 3
    
    POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
    if [ "$POWER_STATE" != "halted" ]; then
        echo "  Force destroying VM..."
        xe vm-destroy uuid="$VM_UUID" 2>&1 || true
        sleep 2
    fi
    echo "VM halted"
    echo ""
fi

echo "STEP 2: Destroying VM..."
echo "------------------------"
if xe vm-destroy uuid="$VM_UUID" 2>&1; then
    echo "VM destroyed"
else
    echo "⚠ VM may already be destroyed"
fi
echo ""

echo "STEP 3: Uninstalling VM and removing disks..."
echo "----------------------------------------------"
if xe vm-uninstall uuid="$VM_UUID" force=true 2>&1; then
    echo "VM uninstalled and disks removed"
else
    echo "⚠ VM uninstall may have failed, checking for orphaned VDIs..."
    
    if [ -n "$DISK_VDI" ]; then
        echo "  Attempting to remove disk VDI: $DISK_VDI"
        xe vdi-destroy uuid="$DISK_VDI" 2>&1 || echo "    (VDI may already be removed)"
    fi
fi
echo ""

echo "STEP 4: Verifying deletion..."
echo "-------------------------------"
REMAINING_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$REMAINING_UUID" ]; then
    echo "Test-3 VM completely deleted"
    echo ""
    echo "Storage freed: ${DISK_GB} GiB"
else
    echo "⚠ WARNING: VM may still exist (UUID: $REMAINING_UUID)"
    echo "  Try manual cleanup:"
    echo "    xe vm-destroy uuid=$REMAINING_UUID"
    echo "    xe vm-uninstall uuid=$REMAINING_UUID force=true"
fi
echo ""

echo "========================================================================"
echo "  Deletion complete"
echo "========================================================================"
echo ""
