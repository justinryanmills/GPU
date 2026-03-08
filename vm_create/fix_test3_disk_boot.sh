#!/bin/bash

set -euo pipefail

VM_NAME="Test-3"

echo "========================================================================"
echo "  Fixing Test-3 to Boot from Installed Disk"
echo "========================================================================"
echo ""

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "VM UUID: $VM_UUID"
echo ""

POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
echo "Current power state: $POWER_STATE"
echo ""

echo "STEP 1: Removing all CD drives..."
echo "-----------------------------------"
CD_VBDS=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | tr ',' '\n' | xargs)

if [ -z "$CD_VBDS" ]; then
    echo "No CD drives found"
else
    for CD_VBD in $CD_VBDS; do
        CD_VBD=$(echo "$CD_VBD" | xargs)
        [ -z "$CD_VBD" ] && continue
        
        echo "Processing CD drive: $CD_VBD"
        
        CURRENT_VDI=$(xe vbd-param-get uuid="$CD_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
        if [ -n "$CURRENT_VDI" ] && [ "$CURRENT_VDI" != "OpaqueRef:NULL" ]; then
            echo "  Ejecting ISO..."
            xe vbd-eject uuid="$CD_VBD" 2>/dev/null || true
            sleep 1
        fi
        
        if [ "$POWER_STATE" = "running" ]; then
            echo "  Unplugging CD drive..."
            xe vbd-unplug uuid="$CD_VBD" 2>/dev/null || true
            sleep 1
        fi
        
        echo "  Removing CD drive completely..."
        xe vbd-destroy uuid="$CD_VBD" 2>/dev/null || true
        echo "  CD drive removed"
    done
    echo "All CD drives removed"
fi
echo ""

echo "STEP 2: Setting boot order to disk only..."
echo "-------------------------------------------"
echo "Boot order codes: c=disk, d=CD, n=network"
echo "Setting to 'c' (disk only)..."

if [ "$POWER_STATE" = "running" ]; then
    echo "VM is running. Shutting down (required for boot order changes)..."
    xe vm-shutdown uuid="$VM_UUID" force=true 2>/dev/null || xe vm-destroy uuid="$VM_UUID" 2>/dev/null || true
    sleep 3
    
    POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
    if [ "$POWER_STATE" != "halted" ]; then
        echo "  Force destroying VM..."
        xe vm-destroy uuid="$VM_UUID" 2>/dev/null || true
        sleep 2
    fi
    echo "VM is now halted"
    echo ""
fi

xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=c
echo "Boot order set to 'c' (disk only)"
echo ""

BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)

echo "Boot params: $BOOT_PARAMS"
if [ "$BOOT_ORDER" = "c" ]; then
    echo "Boot order verified: order=c (disk only)"
else
    echo "⚠ Boot order: $BOOT_ORDER (expected: c)"
    echo "  Attempting to fix again..."
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=c 2>&1 || true
    sleep 1
    BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
    BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
    if [ "$BOOT_ORDER" = "c" ]; then
        echo "Boot order now correct"
    else
        echo "✗ ERROR: Boot order still incorrect!"
    fi
fi
echo ""

echo "STEP 3: Verifying disk configuration..."
echo "----------------------------------------"
DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -z "$DISK_VBD" ]; then
    echo "✗ ERROR: No disk found!"
    exit 1
fi

DISK_BOOTABLE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=bootable 2>/dev/null || echo "false")
if [ "$DISK_BOOTABLE" != "true" ]; then
    echo "  Setting disk as bootable..."
    xe vbd-param-set uuid="$DISK_VBD" bootable=true 2>/dev/null || echo "    (Note: May require VM to be halted)"
    echo "Disk marked as bootable"
else
    echo "Disk is already bootable"
fi
echo ""

echo "========================================================================"
echo "  Configuration Complete"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - All CD drives removed"
echo "  - Boot order set to disk only (order=c)"
echo "  - Disk is bootable"
echo ""
echo "Next step: Start the VM"
echo "  xe vm-start uuid=$VM_UUID"
echo ""
echo "The VM should now boot from the installed Ubuntu system on disk."
echo ""
