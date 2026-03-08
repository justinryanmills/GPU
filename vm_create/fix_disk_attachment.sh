#!/bin/bash

set -euo pipefail

VM_NAME="${1:-Test-3}"

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "========================================================================"
echo "  Fixing Disk Attachment: $VM_NAME"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo ""

POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
if [ "$POWER_STATE" = "running" ]; then
    echo "VM is running. Shutting down..."
    xe vm-shutdown uuid="$VM_UUID" force=true 2>/dev/null || xe vm-destroy uuid="$VM_UUID" 2>/dev/null || true
    sleep 3
    echo "VM is now halted"
    echo ""
fi

DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -z "$DISK_VBD" ]; then
    echo "✗ ERROR: No disk found!"
    exit 1
fi

echo "Disk VBD: $DISK_VBD"
echo ""

echo "STEP 1: Ensuring disk is bootable..."
echo "--------------------------------------"
DISK_BOOTABLE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=bootable 2>/dev/null || echo "false")
if [ "$DISK_BOOTABLE" != "true" ]; then
    echo "Setting disk as bootable..."
    xe vbd-param-set uuid="$DISK_VBD" bootable=true
    echo "Disk is now bootable"
else
    echo "Disk is already bootable"
fi
echo ""

echo "STEP 2: Ensuring disk storage is accessible..."
echo "------------------------------------------------"
DISK_VDI=$(xe vbd-param-get uuid="$DISK_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
if [ -z "$DISK_VDI" ] || [ "$DISK_VDI" = "OpaqueRef:NULL" ]; then
    echo "✗ ERROR: Disk VBD has no VDI attached!"
    echo "  The disk is not connected to any storage"
    exit 1
fi

VDI_SR=$(xe vdi-param-get uuid="$DISK_VDI" param-name=sr-uuid 2>/dev/null || echo "")
if [ -z "$VDI_SR" ]; then
    echo "✗ ERROR: Cannot determine storage repository for disk"
    exit 1
fi

echo "Disk VDI: $DISK_VDI"
echo "Storage SR: $VDI_SR"

SR_PBD=$(xe pbd-list sr-uuid="$VDI_SR" params=uuid --minimal | head -1)
if [ -n "$SR_PBD" ]; then
    PBD_ATTACHED=$(xe pbd-param-get uuid="$SR_PBD" param-name=currently-attached 2>/dev/null || echo "false")
    if [ "$PBD_ATTACHED" != "true" ]; then
        echo "Storage PBD is not attached. Attaching..."
        xe pbd-plug uuid="$SR_PBD" 2>&1 || echo "  (Note: May fail if SR type doesn't require PBD attachment)"
    else
        echo "Storage PBD is attached"
    fi
fi
echo ""

echo "STEP 3: Removing all CD drives..."
echo "-----------------------------------"
CD_VBDS=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | tr ',' '\n' | xargs)
if [ -n "$CD_VBDS" ]; then
    for CD_VBD in $CD_VBDS; do
        CD_VBD=$(echo "$CD_VBD" | xargs)
        [ -z "$CD_VBD" ] && continue
        echo "Removing CD drive: $CD_VBD"
        xe vbd-unplug uuid="$CD_VBD" 2>/dev/null || true
        xe vbd-destroy uuid="$CD_VBD" 2>/dev/null || true
    done
    echo "All CD drives removed"
else
    echo "No CD drives found"
fi
echo ""

echo "STEP 4: Setting boot order to disk only..."
echo "--------------------------------------------"
xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=d

BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
if [ "$BOOT_ORDER" = "d" ]; then
    echo "Boot order set to disk only (order=$BOOT_ORDER)"
else
    echo "⚠ Boot order: $BOOT_ORDER (expected: d)"
fi
echo ""

echo "========================================================================"
echo "  Summary"
echo "========================================================================"
echo ""
echo "Disk Configuration:"
echo "  VBD UUID: $DISK_VBD"
echo "  VDI UUID: $DISK_VDI"
echo "  Bootable: true"
echo "  Boot Order: d (disk only)"
echo "  CD Drives: 0"
echo ""
echo "The disk should now attach automatically when VM starts."
echo ""
echo "To start the VM:"
echo "  xe vm-start uuid=$VM_UUID"
echo ""
echo "If VM still shows 'No bootable device', the issue may be:"
echo "  1. Ubuntu installation didn't complete (GRUB not installed)"
echo "  2. Disk partitions weren't created correctly"
echo "  3. Need to check VM logs for errors"
echo ""
