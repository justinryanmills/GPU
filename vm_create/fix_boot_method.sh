#!/bin/bash

set -euo pipefail

VM_NAME="${1:-Test-3}"

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "========================================================================"
echo "  Fixing Boot Method: $VM_NAME"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo ""

POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
echo "Current power state: $POWER_STATE"
if [ "$POWER_STATE" = "running" ]; then
    echo "⚠ VM is running. Boot method changes require VM to be halted."
    echo "  Shutting down VM..."
    xe vm-shutdown uuid="$VM_UUID" force=true 2>&1 || xe vm-destroy uuid="$VM_UUID" 2>&1 || true
    sleep 3
    POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
    if [ "$POWER_STATE" != "halted" ]; then
        echo "✗ ERROR: Failed to halt VM (current state: $POWER_STATE)"
        exit 1
    fi
    echo "VM is now halted"
fi
echo ""

echo "STEP 1: Checking firmware settings..."
echo "-------------------------------------"
PLATFORM_FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:firmware 2>/dev/null || echo "N/A")
BOOT_PARAMS_FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params:firmware 2>/dev/null || echo "N/A")
echo "  Platform firmware: $PLATFORM_FIRMWARE"
echo "  Boot params firmware: $BOOT_PARAMS_FIRMWARE"

if [ "$PLATFORM_FIRMWARE" != "uefi" ]; then
    echo "  Setting platform firmware to UEFI..."
    if xe vm-param-set uuid="$VM_UUID" platform:firmware=uefi 2>&1; then
        echo "  Platform firmware set to UEFI"
    else
        echo "  ✗ ERROR: Failed to set platform firmware to UEFI"
        exit 1
    fi
else
    echo "  Platform firmware already set to UEFI"
fi

if [ "$BOOT_PARAMS_FIRMWARE" != "uefi" ]; then
    echo "  Setting boot params firmware to UEFI..."
    if xe vm-param-set uuid="$VM_UUID" HVM-boot-params:firmware=uefi 2>&1; then
        echo "  Boot params firmware set to UEFI"
    else
        echo "  ✗ ERROR: Failed to set boot params firmware to UEFI"
        exit 1
    fi
else
    echo "  Boot params firmware already set to UEFI"
fi
echo ""

echo "STEP 2: Checking VIRIDIAN settings..."
echo "-------------------------------------"
PLATFORM_VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:viridian 2>/dev/null || echo "N/A")
BOOT_PARAMS_VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params:viridian 2>/dev/null || echo "N/A")
echo "  Platform VIRIDIAN: $PLATFORM_VIRIDIAN"
echo "  Boot params VIRIDIAN: $BOOT_PARAMS_VIRIDIAN"

if [ "$PLATFORM_VIRIDIAN" != "false" ]; then
    echo "  Disabling VIRIDIAN in platform params..."
    if xe vm-param-set uuid="$VM_UUID" platform:viridian=false 2>&1; then
        echo "  Platform VIRIDIAN disabled"
    else
        echo "  ⚠ Warning: Failed to set platform VIRIDIAN (may not be critical)"
    fi
else
    echo "  Platform VIRIDIAN already disabled"
fi

if [ "$BOOT_PARAMS_VIRIDIAN" != "false" ]; then
    echo "  Disabling VIRIDIAN in boot params..."
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:viridian=false 2>&1 || true
    echo "  Boot params VIRIDIAN disabled"
else
    echo "  Boot params VIRIDIAN already disabled"
fi
echo ""

echo "STEP 3: Verifying boot order..."
echo "--------------------------------"
BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
echo "  Current boot order: $BOOT_ORDER"

if [ "$BOOT_ORDER" != "d" ]; then
    echo "  Setting boot order to 'd' (disk only)..."
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=d 2>&1 || true
    sleep 1
    BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
    BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
    if [ "$BOOT_ORDER" = "d" ]; then
        echo "  Boot order set to disk only"
    else
        echo "  ⚠ WARNING: Boot order may not be set correctly (current: $BOOT_ORDER)"
    fi
else
    echo "  Boot order already set to disk only"
fi
echo ""

echo "STEP 4: Final verification..."
echo "------------------------------"
PLATFORM_FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:firmware 2>/dev/null || echo "N/A")
BOOT_PARAMS_FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params:firmware 2>/dev/null || echo "N/A")
PLATFORM_VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:viridian 2>/dev/null || echo "N/A")
BOOT_PARAMS_VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params:viridian 2>/dev/null || echo "N/A")
BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)

echo "  Platform firmware: $PLATFORM_FIRMWARE"
echo "  Boot params firmware: $BOOT_PARAMS_FIRMWARE"
echo "  Platform VIRIDIAN: $PLATFORM_VIRIDIAN"
echo "  Boot params VIRIDIAN: $BOOT_PARAMS_VIRIDIAN"
echo "  Boot order: $BOOT_ORDER"
echo "  Boot params: $BOOT_PARAMS"
echo ""

echo "========================================================================"
if [ "$BOOT_PARAMS_FIRMWARE" = "uefi" ] && [ "$PLATFORM_VIRIDIAN" = "false" ] && [ "$BOOT_ORDER" = "d" ]; then
    echo "  Boot method configuration complete!"
    echo "========================================================================"
    echo ""
    echo "Configuration now matches working VMs (Test-1, Test-2):"
    echo "  Boot params firmware: UEFI"
    echo "  Platform VIRIDIAN: Disabled"
    echo "  Boot order: Disk only"
    echo ""
    echo "Next steps:"
    echo "  1. Start the VM: xe vm-start uuid=$VM_UUID"
    echo "  2. Monitor boot: xe vm-list uuid=$VM_UUID params=power-state"
    echo "  3. Connect via VNC: bash connect_vnc.sh $VM_NAME"
    echo ""
    echo "⚠ NOTE: If Ubuntu installation didn't complete, the VM may still"
    echo "  fail to boot. In that case, you'll need to reinstall Ubuntu."
else
    echo "  ⚠ WARNING: Some settings may not be correct"
    echo "========================================================================"
    echo ""
    if [ "$BOOT_PARAMS_FIRMWARE" != "uefi" ]; then
        echo "✗ Boot params firmware is not UEFI (current: $BOOT_PARAMS_FIRMWARE)"
    fi
    if [ "$PLATFORM_VIRIDIAN" != "false" ]; then
        echo "✗ Platform VIRIDIAN is not disabled (current: $PLATFORM_VIRIDIAN)"
    fi
    if [ "$BOOT_ORDER" != "d" ]; then
        echo "✗ Boot order is not 'd' (current: $BOOT_ORDER)"
    fi
    echo ""
    echo "Try running this script again, or fix manually:"
    echo "  xe vm-param-set uuid=$VM_UUID platform:firmware=uefi"
    echo "  xe vm-param-set uuid=$VM_UUID HVM-boot-params:firmware=uefi"
    echo "  xe vm-param-set uuid=$VM_UUID platform:viridian=false"
    echo "  xe vm-param-set uuid=$VM_UUID HVM-boot-params:viridian=false"
    echo "  xe vm-param-set uuid=$VM_UUID HVM-boot-params:order=d"
fi
echo ""
