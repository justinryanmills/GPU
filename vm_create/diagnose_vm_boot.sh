#!/bin/bash

set -euo pipefail

VM_NAME="${1:-}"

if [ -z "$VM_NAME" ]; then
    echo "ERROR: VM name required"
    echo "Usage: bash diagnose_vm_boot.sh <VM_NAME>"
    exit 1
fi

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "========================================================================"
echo "  VM Boot Diagnostics: $VM_NAME"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo ""

POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
echo "Power State: $POWER_STATE"
echo ""

echo "Boot Configuration:"
echo "-------------------"
BOOT_POLICY=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-policy 2>/dev/null || echo "unknown")
BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
echo "Boot Policy: $BOOT_POLICY"
echo "Boot Params: $BOOT_PARAMS"

BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1)
if [ -z "$BOOT_ORDER" ]; then
    BOOT_ORDER=$(echo "$BOOT_PARAMS" | sed -n 's/.*order[=:]\s*\([^;]*\).*/\1/p' | head -1)
fi
BOOT_ORDER=$(echo "$BOOT_ORDER" | xargs)

if [ "$BOOT_ORDER" = "d" ]; then
    echo "Boot Order: $BOOT_ORDER (disk only - correct)"
elif [ -n "$BOOT_ORDER" ]; then
    echo "⚠ Boot Order: $BOOT_ORDER (expected: d)"
    echo "  This may cause VM to try booting from CD and fail!"
else
    echo "⚠ Boot Order: Not set or unreadable"
fi
echo ""

echo "Virtual Block Devices (VBDs):"
echo "------------------------------"
for VBD_UUID in $(xe vbd-list vm-uuid="$VM_UUID" params=uuid --minimal | tr ',' '\n'); do
    VBD_UUID=$(echo "$VBD_UUID" | xargs)
    [ -z "$VBD_UUID" ] && continue
    
    VBD_TYPE=$(xe vbd-param-get uuid="$VBD_UUID" param-name=type 2>/dev/null || echo "unknown")
    VBD_DEVICE=$(xe vbd-param-get uuid="$VBD_UUID" param-name=device 2>/dev/null || echo "unknown")
    VBD_BOOTABLE=$(xe vbd-param-get uuid="$VBD_UUID" param-name=bootable 2>/dev/null || echo "false")
    VBD_ATTACHED=$(xe vbd-param-get uuid="$VBD_UUID" param-name=currently-attached 2>/dev/null || echo "false")
    
    if [ "$VBD_TYPE" = "CD" ]; then
        VBD_VDI=$(xe vbd-param-get uuid="$VBD_UUID" param-name=vdi-uuid 2>/dev/null || echo "")
        if [ -z "$VBD_VDI" ] || [ "$VBD_VDI" = "OpaqueRef:NULL" ]; then
            echo "  CD Drive (device $VBD_DEVICE): Empty, Attached=$VBD_ATTACHED"
        else
            echo "  ⚠ CD Drive (device $VBD_DEVICE): Has ISO, Attached=$VBD_ATTACHED"
            echo "    This will cause boot failure if boot order includes CD!"
        fi
    elif [ "$VBD_TYPE" = "Disk" ]; then
        echo "  Disk (device $VBD_DEVICE): Bootable=$VBD_BOOTABLE, Attached=$VBD_ATTACHED"
        if [ "$VBD_BOOTABLE" != "true" ]; then
            echo "    ⚠ WARNING: Disk is not marked as bootable!"
            echo "    Fix: xe vbd-param-set uuid=$VBD_UUID bootable=true"
        fi
        if [ "$VBD_ATTACHED" != "true" ]; then
            echo "    ✗ CRITICAL: Disk is not attached!"
            echo "    This will cause 'No bootable device' error"
            echo "    Fix: xe vbd-plug uuid=$VBD_UUID"
        fi
    fi
done
echo ""

echo "Potential Issues:"
echo "-----------------"
ISSUES=0

if [ "$BOOT_ORDER" != "d" ]; then
    echo "  ✗ Boot order is not 'd' (disk only)"
    echo "    Current: '$BOOT_ORDER'"
    echo "    Fix: xe vm-param-set uuid=$VM_UUID HVM-boot-params:order=d"
    ISSUES=$((ISSUES + 1))
fi

CD_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | head -1)
if [ -n "$CD_VBD" ]; then
    CD_VDI=$(xe vbd-param-get uuid="$CD_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
    CD_ATTACHED=$(xe vbd-param-get uuid="$CD_VBD" param-name=currently-attached 2>/dev/null || echo "false")
    
    if [ -n "$CD_VDI" ] && [ "$CD_VDI" != "OpaqueRef:NULL" ]; then
        echo "  ⚠ CD drive still has ISO inserted"
        echo "    This will cause boot failure if boot order includes CD"
        echo "    Fix: xe vbd-eject uuid=$CD_VBD"
        ISSUES=$((ISSUES + 1))
    fi
    
    if [ "$CD_ATTACHED" = "true" ] && [ "$BOOT_ORDER" != "d" ]; then
        echo "  ⚠ CD drive is attached and boot order may try to use it"
        echo "    Fix: xe vbd-unplug uuid=$CD_VBD"
        ISSUES=$((ISSUES + 1))
    fi
fi

DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -z "$DISK_VBD" ]; then
    echo "  ✗ No disk found - VM cannot boot!"
    ISSUES=$((ISSUES + 1))
else
    DISK_BOOTABLE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=bootable 2>/dev/null || echo "false")
    DISK_ATTACHED=$(xe vbd-param-get uuid="$DISK_VBD" param-name=currently-attached 2>/dev/null || echo "false")
    
    if [ "$DISK_BOOTABLE" != "true" ]; then
        echo "  ⚠ Disk is not marked as bootable"
        echo "    Fix: xe vbd-param-set uuid=$DISK_VBD bootable=true"
        ISSUES=$((ISSUES + 1))
    fi
    
    if [ "$DISK_ATTACHED" != "true" ]; then
        echo "  ✗ CRITICAL: Disk is not attached!"
        echo "    This causes 'No bootable device' error"
        echo "    Fix: xe vbd-plug uuid=$DISK_VBD"
        ISSUES=$((ISSUES + 1))
    fi
fi

echo ""
if [ $ISSUES -eq 0 ]; then
    echo "No obvious boot issues detected"
    echo "  If VM still shuts down, check VM logs:"
    echo "    tail -50 /var/log/xensource.log | grep -i $VM_NAME"
else
    echo "⚠ Found $ISSUES potential issue(s) that may cause boot failure"
fi
echo ""
