#!/bin/bash

set -euo pipefail

VM_NAME="${1:-Test-3}"

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "========================================================================"
echo "  Disk Boot Diagnostics: $VM_NAME"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo ""

POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
echo "Power State: $POWER_STATE"
echo ""

DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -z "$DISK_VBD" ]; then
    echo "✗ ERROR: No disk found!"
    exit 1
fi

echo "Disk VBD: $DISK_VBD"
echo ""

DISK_DEVICE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=device 2>/dev/null || echo "")
DISK_BOOTABLE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=bootable 2>/dev/null || echo "")
DISK_ATTACHED=$(xe vbd-param-get uuid="$DISK_VBD" param-name=currently-attached 2>/dev/null || echo "")
DISK_VDI=$(xe vbd-param-get uuid="$DISK_VBD" param-name=vdi-uuid 2>/dev/null || echo "")

echo "Disk Configuration:"
echo "-------------------"
echo "  Device: $DISK_DEVICE"
echo "  Bootable: $DISK_BOOTABLE"
echo "  Attached: $DISK_ATTACHED"
echo "  VDI UUID: $DISK_VDI"
echo ""

if [ -n "$DISK_VDI" ] && [ "$DISK_VDI" != "OpaqueRef:NULL" ]; then
    VDI_SIZE=$(xe vdi-param-get uuid="$DISK_VDI" param-name=virtual-size 2>/dev/null || echo "")
    VDI_SR=$(xe vdi-param-get uuid="$DISK_VDI" param-name=sr-uuid 2>/dev/null || echo "")
    VDI_NAME=$(xe vdi-param-get uuid="$DISK_VDI" param-name=name-label 2>/dev/null || echo "")
    
    echo "VDI Details:"
    echo "------------"
    echo "  Name: $VDI_NAME"
    echo "  Size: $VDI_SIZE"
    echo "  SR UUID: $VDI_SR"
    echo ""
    
    SR_MOUNT=$(xe sr-param-get uuid="$VDI_SR" param-name=name-label 2>/dev/null || echo "")
    echo "  Storage Repository: $SR_MOUNT"
    echo ""
fi

echo "Boot Configuration:"
echo "-------------------"
BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
echo "Boot Params: $BOOT_PARAMS"
echo "Boot Order: $BOOT_ORDER"
echo ""

CD_VBDS=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | tr ',' '\n' | xargs)
CD_COUNT=$(echo "$CD_VBDS" | wc -w)
echo "CD Drives: $CD_COUNT"
if [ "$CD_COUNT" -gt 0 ]; then
    echo "  ⚠ WARNING: $CD_COUNT CD drive(s) still exist!"
    for CD_VBD in $CD_VBDS; do
        echo "    CD VBD: $CD_VBD"
    done
else
    echo "  No CD drives"
fi
echo ""

echo "========================================================================"
echo "  Diagnosis Summary"
echo "========================================================================"
echo ""

ISSUES=0

if [ "$DISK_BOOTABLE" != "true" ]; then
    echo "✗ ISSUE: Disk is not marked as bootable"
    echo "  Fix: xe vbd-param-set uuid=$DISK_VBD bootable=true"
    ISSUES=$((ISSUES + 1))
fi

if [ "$POWER_STATE" = "running" ] && [ "$DISK_ATTACHED" != "true" ]; then
    echo "✗ CRITICAL: VM is running but disk is NOT attached!"
    echo "  This will cause 'No bootable device' error"
    echo "  Fix: xe vbd-plug uuid=$DISK_VBD"
    ISSUES=$((ISSUES + 1))
elif [ "$POWER_STATE" = "halted" ] && [ "$DISK_ATTACHED" != "true" ]; then
    echo "ℹ INFO: VM is halted, disk shows as not attached (this is normal)"
    echo "  Disk will attach automatically when VM starts"
fi

if [ "$BOOT_ORDER" != "d" ]; then
    echo "✗ ISSUE: Boot order is not 'd' (disk only)"
    echo "  Current: '$BOOT_ORDER'"
    echo "  Fix: xe vm-param-set uuid=$VM_UUID HVM-boot-params:order=d"
    ISSUES=$((ISSUES + 1))
fi

if [ "$CD_COUNT" -gt 0 ]; then
    echo "⚠ WARNING: CD drive(s) still present"
    echo "  Even if empty, BIOS may try to boot from CD first"
    echo "  Fix: Remove all CD drives"
    ISSUES=$((ISSUES + 1))
fi

if [ -z "$DISK_VDI" ] || [ "$DISK_VDI" = "OpaqueRef:NULL" ]; then
    echo "✗ CRITICAL: Disk VBD has no VDI!"
    echo "  The disk is not connected to any storage"
    ISSUES=$((ISSUES + 1))
fi

echo ""
if [ $ISSUES -eq 0 ]; then
    echo "No obvious configuration issues detected"
    echo ""
    echo "If VM still fails to boot, possible causes:"
    echo "  1. Ubuntu installation didn't complete properly"
    echo "  2. GRUB bootloader wasn't installed to disk"
    echo "  3. Disk partitions weren't created correctly"
    echo ""
    echo "To check if disk has boot sector (requires VM to be running):"
    echo "  # On dom0, check if disk device exists in VM"
    echo "  # Or check VM logs: tail -50 /var/log/xensource.log | grep -i $VM_NAME"
else
    echo "⚠ Found $ISSUES issue(s) that need to be fixed"
fi
echo ""
