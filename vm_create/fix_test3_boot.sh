#!/bin/bash

set -euo pipefail

VM_NAME="Test-3"

echo "========================================================================"
echo "  Fixing Test-3 VM Boot Configuration"
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

ISO_SR_UUID=$(xe sr-list name-label="VGS ISO Storage" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$ISO_SR_UUID" ]; then
    echo "ERROR: VGS ISO Storage SR not found"
    exit 1
fi

echo "ISO SR UUID: $ISO_SR_UUID"
echo ""

ISO_NAME="ubuntu-22.04.5-desktop-amd64.iso"
ISO_VDI_UUID=$(xe vdi-list sr-uuid="$ISO_SR_UUID" name-label="$ISO_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$ISO_VDI_UUID" ]; then
    echo "ERROR: ISO '$ISO_NAME' not found in SR"
    exit 1
fi

echo "ISO VDI UUID: $ISO_VDI_UUID"
echo ""

echo "STEP 1: Checking CD drive..."
echo "-----------------------------"
CD_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$CD_VBD" ]; then
    echo "  Creating CD drive..."
    CD_VBD=$(xe vbd-create vm-uuid="$VM_UUID" device=3 type=CD mode=RO)
    echo "  CD drive created: $CD_VBD"
else
    echo "  CD drive exists: $CD_VBD"
fi

CURRENT_ISO=$(xe vbd-param-get uuid="$CD_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
if [ -z "$CURRENT_ISO" ] || [ "$CURRENT_ISO" != "$ISO_VDI_UUID" ]; then
    echo "  ISO not inserted or wrong ISO. Inserting correct ISO..."
    
    VBD_PLUGGED=$(xe vbd-param-get uuid="$CD_VBD" param-name=currently-attached 2>/dev/null || echo "false")
    
    if [ "$POWER_STATE" = "running" ] && [ "$VBD_PLUGGED" = "true" ]; then
        echo "  VM is running and CD drive is plugged. Unplugging..."
        xe vbd-unplug uuid="$CD_VBD" 2>/dev/null || true
        sleep 1
    fi
    
    xe vbd-insert uuid="$CD_VBD" vdi-uuid="$ISO_VDI_UUID"
    echo "  ISO inserted"
    
    if [ "$POWER_STATE" = "running" ]; then
        echo "  Plugging CD drive..."
        xe vbd-plug uuid="$CD_VBD"
        sleep 1
        echo "  CD drive plugged"
    fi
else
    echo "  Correct ISO already inserted"
    
    if [ "$POWER_STATE" = "running" ]; then
        VBD_PLUGGED=$(xe vbd-param-get uuid="$CD_VBD" param-name=currently-attached 2>/dev/null || echo "false")
        if [ "$VBD_PLUGGED" != "true" ]; then
            echo "  Plugging CD drive..."
            xe vbd-plug uuid="$CD_VBD"
            sleep 1
            echo "  CD drive plugged"
        fi
    fi
fi
echo ""

echo "STEP 2: Checking boot configuration..."
echo "--------------------------------------"
BOOT_POLICY=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-policy 2>/dev/null || echo "")
BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:firmware 2>/dev/null || echo "")

echo "  Current boot policy: $BOOT_POLICY"
echo "  Current boot params: $BOOT_PARAMS"
echo "  Current firmware: $FIRMWARE"
echo ""

echo "  Setting boot order to CD first, then disk..."
xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=dc
echo "  Boot order set: CD first (d), then disk (c)"
echo ""

if [ "$FIRMWARE" != "uefi" ]; then
    echo "  Setting firmware to UEFI..."
    xe vm-param-set uuid="$VM_UUID" platform:firmware=uefi
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:firmware=uefi
    echo "  Firmware set to UEFI"
    echo ""
fi

VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:viridian 2>/dev/null || echo "")
if [ "$VIRIDIAN" != "false" ]; then
    echo "  Disabling VIRIDIAN..."
    xe vm-param-set uuid="$VM_UUID" platform:viridian=false
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:viridian=false 2>/dev/null || true
    echo "  VIRIDIAN disabled"
    echo ""
fi

echo "STEP 3: Final verification..."
echo "-------------------------------"
FINAL_ISO=$(xe vbd-param-get uuid="$CD_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
if [ "$FINAL_ISO" = "$ISO_VDI_UUID" ]; then
    echo "  ISO is correctly inserted"
else
    echo "  ✗ ERROR: ISO insertion failed"
    exit 1
fi

FINAL_BOOT=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
if echo "$FINAL_BOOT" | grep -q "order.*dc\|order.*d"; then
    echo "  Boot order is correct (CD first)"
else
    echo "  ⚠ WARNING: Boot order may not be correct: $FINAL_BOOT"
fi
echo ""

echo "========================================================================"
echo "  Configuration Complete"
echo "========================================================================"
echo ""
echo "If the VM is currently in UEFI shell, you need to restart it:"
echo "  1. In Xen Orchestra, click 'Shutdown' or 'Reboot' for Test-3"
echo "  2. Or from dom0: xe vm-reboot uuid=$VM_UUID"
echo ""
echo "After restart, the VM should boot from the Ubuntu ISO."
echo ""
