#!/bin/bash

set -euo pipefail

VM_NAME="${1:-Test-3}"

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "========================================================================"
echo "  Debugging Boot Issue: $VM_NAME"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo ""

echo "All Virtual Block Devices (VBDs):"
echo "----------------------------------"
xe vbd-list vm-uuid="$VM_UUID" params=uuid,type,device,bootable,currently-attached,vdi-uuid --minimal | while IFS=',' read -r VBD_UUID TYPE DEVICE BOOTABLE ATTACHED VDI_UUID; do
    echo "VBD: $VBD_UUID"
    echo "  Type: $TYPE"
    echo "  Device: $DEVICE"
    echo "  Bootable: $BOOTABLE"
    echo "  Attached: $ATTACHED"
    echo "  VDI UUID: $VDI_UUID"
    
    if [ "$TYPE" = "CD" ]; then
        echo "  ⚠ CD DRIVE FOUND - This may be causing boot issues!"
        if [ -n "$VDI_UUID" ] && [ "$VDI_UUID" != "OpaqueRef:NULL" ]; then
            echo "  ⚠ CD has ISO inserted: $VDI_UUID"
        fi
    fi
    echo ""
done

echo "Boot Configuration:"
echo "-------------------"
BOOT_POLICY=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-policy 2>/dev/null || echo "")
BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
echo "Boot Policy: $BOOT_POLICY"
echo "Boot Params: $BOOT_PARAMS"
echo ""

CD_COUNT=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | wc -l)
echo "CD Drive Count: $CD_COUNT"
if [ "$CD_COUNT" -gt 0 ]; then
    echo "⚠ WARNING: $CD_COUNT CD drive(s) still exist!"
    echo "Even if empty, BIOS may try to boot from them first"
    echo ""
    echo "CD VBD UUIDs:"
    xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal
    echo ""
fi

DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -n "$DISK_VBD" ]; then
    echo "Disk VBD: $DISK_VBD"
    DISK_DEVICE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=device 2>/dev/null || echo "")
    DISK_BOOTABLE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=bootable 2>/dev/null || echo "")
    echo "  Device: $DISK_DEVICE"
    echo "  Bootable: $DISK_BOOTABLE"
    
    echo ""
    echo "Recommendations:"
    if [ "$CD_COUNT" -gt 0 ]; then
        echo "  1. Remove ALL CD drives completely:"
        for CD_VBD in $(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | tr ',' '\n'); do
            echo "     xe vbd-destroy uuid=$CD_VBD"
        done
    fi
    
    if [ "$DISK_BOOTABLE" != "true" ]; then
        echo "  2. Set disk as bootable:"
        echo "     xe vbd-param-set uuid=$DISK_VBD bootable=true"
    fi
    
    echo "  3. Verify boot order is 'd' (should already be set)"
    echo "  4. Start VM and check console"
fi

echo ""
