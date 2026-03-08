#!/bin/bash

set -euo pipefail

echo "========================================================================"
echo "  Checking Boot Configuration of Working VMs"
echo "========================================================================"
echo ""

for VM_NAME in "Test-1" "Test-2"; do
    VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
    if [ -z "$VM_UUID" ]; then
        echo "⚠ VM '$VM_NAME' not found, skipping..."
        echo ""
        continue
    fi
    
    echo "VM: $VM_NAME ($VM_UUID)"
    echo "----------------------------------------"
    
    BOOT_POLICY=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-policy 2>/dev/null || echo "N/A")
    echo "  Boot Policy: $BOOT_POLICY"
    
    BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "N/A")
    echo "  Boot Params: $BOOT_PARAMS"
    
    PLATFORM=$(xe vm-param-get uuid="$VM_UUID" param-name=platform 2>/dev/null || echo "N/A")
    echo "  Platform: $PLATFORM"
    
    FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:firmware 2>/dev/null || echo "N/A")
    echo "  Firmware: $FIRMWARE"
    
    VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params:viridian 2>/dev/null || echo "N/A")
    echo "  Viridian: $VIRIDIAN"
    
    echo "  All Platform Params:"
    xe vm-param-get uuid="$VM_UUID" param-name=platform 2>/dev/null | tr ',' '\n' | sed 's/^/    /' || echo "    (Could not read)"
    
    echo ""
done

echo "========================================================================"
echo "  Checking Test-3 (Current Problem VM)"
echo "========================================================================"
echo ""

VM_NAME="Test-3"
VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "⚠ VM '$VM_NAME' not found"
else
    echo "VM: $VM_NAME ($VM_UUID)"
    echo "----------------------------------------"
    
    BOOT_POLICY=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-policy 2>/dev/null || echo "N/A")
    echo "  Boot Policy: $BOOT_POLICY"
    
    BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "N/A")
    echo "  Boot Params: $BOOT_PARAMS"
    
    PLATFORM=$(xe vm-param-get uuid="$VM_UUID" param-name=platform 2>/dev/null || echo "N/A")
    echo "  Platform: $PLATFORM"
    
    FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:firmware 2>/dev/null || echo "N/A")
    echo "  Firmware: $FIRMWARE"
    
    VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params:viridian 2>/dev/null || echo "N/A")
    echo "  Viridian: $VIRIDIAN"
    
    echo "  All Platform Params:"
    xe vm-param-get uuid="$VM_UUID" param-name=platform 2>/dev/null | tr ',' '\n' | sed 's/^/    /' || echo "    (Could not read)"
fi

echo ""
