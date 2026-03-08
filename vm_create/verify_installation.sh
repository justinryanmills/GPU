#!/bin/bash

set -euo pipefail

VM_NAME="${1:-Test-3}"

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "========================================================================"
echo "  Verifying Ubuntu Installation: $VM_NAME"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo ""

DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -z "$DISK_VBD" ]; then
    echo "✗ ERROR: No disk found!"
    exit 1
fi

DISK_VDI=$(xe vbd-param-get uuid="$DISK_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
DISK_DEVICE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=device 2>/dev/null || echo "")

echo "Disk Information:"
echo "-----------------"
echo "  VBD UUID: $DISK_VBD"
echo "  VDI UUID: $DISK_VDI"
echo "  Device: $DISK_DEVICE"
echo ""

POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")

if [ "$POWER_STATE" != "running" ] || [ "$DOMID" = "-1" ]; then
    echo "⚠ VM is not running. Starting VM to check disk..."
    xe vm-start uuid="$VM_UUID"
    sleep 5
    
    DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")
    if [ "$DOMID" = "-1" ]; then
        echo "✗ ERROR: VM failed to start"
        exit 1
    fi
    echo "VM started (Domain ID: $DOMID)"
    echo ""
fi

echo "Checking disk partition table..."
echo "--------------------------------"


DISK_BACKEND="/dev/xen/vbd-$DOMID-$DISK_DEVICE"
if [ -b "$DISK_BACKEND" ]; then
    echo "Found disk backend: $DISK_BACKEND"
    echo ""
    echo "Partition table:"
    fdisk -l "$DISK_BACKEND" 2>&1 | head -20 || echo "  (Could not read partition table)"
    echo ""
    
    echo "Checking for boot sector..."
    if hexdump -C "$DISK_BACKEND" -n 512 2>/dev/null | grep -q "GRUB\|BOOT\|EFI"; then
        echo "  Boot sector found (likely has GRUB)"
    else
        echo "  ✗ No boot sector detected"
        echo "  This indicates installation may not have completed"
    fi
else
    echo "⚠ Could not find disk backend device"
    echo "  Path checked: $DISK_BACKEND"
    echo ""
    echo "Alternative: Check VM console/logs for installation errors"
fi

echo ""
echo "========================================================================"
echo "  Interpretation"
echo "========================================================================"
echo ""
echo "If no partitions are found:"
echo "  → Ubuntu installation did NOT complete"
echo "  → Disk was never partitioned"
echo "  → Solution: Reinstall Ubuntu completely"
echo ""
echo "If partitions exist but no boot sector:"
echo "  → Installation partially completed but GRUB wasn't installed"
echo "  → Solution: Reinstall or boot from live CD to install GRUB"
echo ""
echo "If boot sector exists:"
echo "  → Installation likely completed"
echo "  → Issue may be with boot order or disk attachment"
echo ""
