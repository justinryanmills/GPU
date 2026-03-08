#!/bin/bash

set -euo pipefail

VM_NAME="${1:-}"
DELETE_EXISTING=false
CUSTOM_IP=""
CUSTOM_MEMORY=""
CUSTOM_DISK=""
CUSTOM_CPUS=""

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --delete-existing)
            DELETE_EXISTING=true
            shift
            ;;
        --ip)
            CUSTOM_IP="$2"
            shift 2
            ;;
        --memory)
            CUSTOM_MEMORY="$2"
            shift 2
            ;;
        --disk)
            CUSTOM_DISK="$2"
            shift 2
            ;;
        --cpus)
            CUSTOM_CPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VM_NAME" ]; then
    echo "ERROR: VM name required"
    echo ""
    echo "Usage: bash create_vm.sh <VM_NAME> [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  bash create_vm.sh Test-3"
    echo "  bash create_vm.sh Test-4 --delete-existing"
    echo "  bash create_vm.sh Test-5 --ip 10.25.33.15 --memory 8GiB"
    exit 1
fi

VM_NUMBER=$(echo "$VM_NAME" | sed -n 's/.*Test-\([0-9]*\)/\1/p')
if [ -z "$VM_NUMBER" ]; then
    echo "ERROR: VM name must be in format 'Test-X' where X is a number"
    echo "Example: Test-3, Test-4, Test-10"
    exit 1
fi

if [ -z "$CUSTOM_IP" ]; then
    IP_LAST_OCTET=$((10 + VM_NUMBER))
    VM_IP="10.25.33.$IP_LAST_OCTET"
else
    VM_IP="$CUSTOM_IP"
fi

VNC_PORT=$((5900 + VM_NUMBER))

GATEWAY="10.25.33.254"
DNS="8.8.8.8"
DISK_SIZE="${CUSTOM_DISK:-40GiB}"
MEMORY="${CUSTOM_MEMORY:-4GiB}"
VCPUS="${CUSTOM_CPUS:-4}"
CD_DEVICE=3

echo "========================================================================"
echo "  Creating VM: $VM_NAME"
echo "========================================================================"
echo "Configuration:"
echo "  IP:      $VM_IP/24"
echo "  Gateway: $GATEWAY"
echo "  DNS:     $DNS"
echo "  Memory:  $MEMORY"
echo "  Disk:    $DISK_SIZE"
echo "  CPUs:    $VCPUS"
echo ""

EXISTING_VM=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -n "$EXISTING_VM" ]; then
    if [ "$DELETE_EXISTING" = "true" ]; then
        echo "Existing VM found. Deleting..."
        POWER_STATE=$(xe vm-param-get uuid="$EXISTING_VM" param-name=power-state 2>/dev/null || echo "halted")
        if [ "$POWER_STATE" != "halted" ]; then
            xe vm-shutdown uuid="$EXISTING_VM" force=true 2>/dev/null || true
            sleep 2
        fi
        xe vm-destroy uuid="$EXISTING_VM" 2>/dev/null || true
        xe vm-uninstall uuid="$EXISTING_VM" force=true
        echo "Existing VM deleted"
        echo ""
    else
        echo "ERROR: VM '$VM_NAME' already exists (UUID: $EXISTING_VM)"
        echo "Use --delete-existing to replace it, or choose a different name"
        exit 1
    fi
fi

verify_sr_accessible() {
    local SR_UUID="$1"
    local SR_MOUNT="/var/run/sr-mount/$SR_UUID"
    
    echo "Verifying SR accessibility: $SR_UUID"
    
    if [ ! -e "$SR_MOUNT" ]; then
        echo "  ✗ ERROR: Mount point missing: $SR_MOUNT"
        return 1
    fi
    
    if [ -L "$SR_MOUNT" ]; then
        local TARGET=$(readlink -f "$SR_MOUNT" 2>/dev/null || true)
        if [ -n "$TARGET" ] && [ -d "$TARGET" ]; then
            echo "  Mount point is symlink to: $TARGET"
        else
            echo "  ✗ ERROR: Symlink target is invalid: $TARGET"
            return 1
        fi
    elif [ -d "$SR_MOUNT" ]; then
        echo "  Mount point directory exists"
        local SR_TYPE=$(xe sr-list uuid="$SR_UUID" params=type --minimal 2>/dev/null || echo "unknown")
        if [ "$SR_TYPE" = "iso" ]; then
            echo "  File-based ISO SR (mount table check not required)"
        elif mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
            echo "  Mount is in mount table"
        else
            echo "  ⚠ WARNING: Not in mount table, but continuing (may be file-based)"
        fi
    else
        echo "  ✗ ERROR: Mount point exists but is not a directory or symlink"
        return 1
    fi
    
    local PBD_UUID=$(xe pbd-list sr-uuid="$SR_UUID" params=uuid --minimal | head -1)
    if [ -z "$PBD_UUID" ]; then
        echo "  ✗ ERROR: No PBD found for SR"
        return 1
    fi
    
    local ATTACHED=$(xe pbd-param-get uuid="$PBD_UUID" param-name=currently-attached 2>/dev/null || echo "false")
    if [ "$ATTACHED" != "true" ]; then
        echo "  ✗ ERROR: PBD not attached (currently-attached=$ATTACHED)"
        return 1
    fi
    echo "  PBD is attached"
    
    local ACTUAL_PATH="$SR_MOUNT"
    if [ -L "$SR_MOUNT" ]; then
        ACTUAL_PATH=$(readlink -f "$SR_MOUNT")
    fi
    
    local FILE_COUNT=$(ls -la "$ACTUAL_PATH" 2>/dev/null | grep -v "^total" | grep -v "^\.$" | grep -v "^\.\.$" | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        echo "  ✗ ERROR: Mount exists but no files readable (empty or corrupted)"
        return 1
    fi
    echo "  Files are readable from mount ($FILE_COUNT items)"
    
    echo "  SR is fully accessible"
    return 0
}

echo "STEP 1: Mounting VGS volume and finding ISO..."
echo "------------------------------------------------"

LV_LINE=$(lvs --noheadings -o vg_name,lv_name 2>/dev/null | grep -i iso_download | head -1)

if [ -z "$LV_LINE" ]; then
    echo "ERROR: Could not find iso_download volume"
    exit 1
fi

VG_NAME=$(echo "$LV_LINE" | awk '{print $1}')
echo "Found VG: $VG_NAME"

LV_PATH="/dev/$VG_NAME/iso_download"
MOUNT_POINT="/mnt/iso-storage"

if [ ! -e "$LV_PATH" ]; then
    lvchange -ay "$VG_NAME/iso_download"
    sleep 1
fi

mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
    mount "$LV_PATH" "$MOUNT_POINT"
    echo "Mounted VGS volume"
fi

ISO_FILE=$(find "$MOUNT_POINT" -name "*ubuntu*server*.iso" -type f 2>/dev/null | head -1)
if [ -z "$ISO_FILE" ]; then
    ISO_FILE=$(find "$MOUNT_POINT" -name "*ubuntu*.iso" -type f 2>/dev/null | head -1)
fi

if [ -z "$ISO_FILE" ]; then
    echo "ERROR: No Ubuntu ISO file found in VGS volume"
    exit 1
fi

ISO_NAME=$(basename "$ISO_FILE")
echo "Found ISO: $ISO_NAME"
echo ""

echo "STEP 2: Finding accessible ISO SR..."
echo "--------------------------------------"

ISO_SR_UUID=""
ISO_SR_NAME=""

for SR_UUID in $(xe sr-list content-type=iso params=uuid --minimal | tr ',' '\n'); do
    SR_UUID=$(echo "$SR_UUID" | xargs)
    [ -z "$SR_UUID" ] && continue
    
    SR_TYPE=$(xe sr-list uuid="$SR_UUID" params=type --minimal 2>/dev/null || true)
    if [ "$SR_TYPE" = "udev" ]; then
        continue
    fi
    
    if verify_sr_accessible "$SR_UUID"; then
        ISO_SR_UUID="$SR_UUID"
        ISO_SR_NAME=$(xe sr-list uuid="$SR_UUID" params=name-label --minimal 2>/dev/null || true)
        echo "Using ISO SR: $ISO_SR_NAME ($ISO_SR_UUID)"
        break
    else
        echo "  Skipping SR $SR_UUID (not accessible)"
    fi
done

if [ -z "$ISO_SR_UUID" ]; then
    echo "ERROR: No accessible ISO SR found"
    exit 1
fi

SR_MOUNT="/var/run/sr-mount/$ISO_SR_UUID"
echo ""

echo "STEP 3: Registering ISO to SR..."
echo "----------------------------------"

EXISTING_ISO=$(xe vdi-list sr-uuid="$ISO_SR_UUID" name-label="$ISO_NAME" params=uuid --minimal 2>/dev/null | head -1)

if [ -n "$EXISTING_ISO" ]; then
    echo "ISO already registered in SR"
    ISO_VDI_UUID="$EXISTING_ISO"
else
    echo "Copying ISO to SR..."
    cp "$ISO_FILE" "$SR_MOUNT/$ISO_NAME"
    echo "ISO copied to SR"
    
    echo "Scanning SR to register ISO..."
    xe sr-scan uuid="$ISO_SR_UUID"
    sleep 2
    
    ISO_VDI_UUID=$(xe vdi-list sr-uuid="$ISO_SR_UUID" name-label="$ISO_NAME" params=uuid --minimal 2>/dev/null | head -1)
    
    if [ -z "$ISO_VDI_UUID" ]; then
        xe sr-scan uuid="$ISO_SR_UUID"
        sleep 2
        ISO_VDI_UUID=$(xe vdi-list sr-uuid="$ISO_SR_UUID" name-label="$ISO_NAME" params=uuid --minimal 2>/dev/null | head -1)
    fi
    
    if [ -z "$ISO_VDI_UUID" ]; then
        echo "ERROR: ISO not registered after scan"
        exit 1
    fi
    echo "ISO registered: $ISO_VDI_UUID"
fi

echo "ISO VDI UUID: $ISO_VDI_UUID"
echo ""

echo "STEP 4: Getting required UUIDs..."
echo "-----------------------------------"

TEMPLATE_UUID=$(xe template-list name-label="Other install media" params=uuid --minimal)
if [ -z "$TEMPLATE_UUID" ]; then
    echo "ERROR: Template Other install media not found"
    exit 1
fi

LOCAL_SR_UUID=$(xe sr-list type=lvm content-type=user params=uuid --minimal | head -1)
if [ -z "$LOCAL_SR_UUID" ]; then
    echo "ERROR: No local storage SR found"
    exit 1
fi

NET_UUID=$(xe network-list bridge=xenbr0 params=uuid --minimal | head -1)
if [ -z "$NET_UUID" ]; then
    echo "ERROR: Network xenbr0 not found"
    exit 1
fi

echo "Template UUID: $TEMPLATE_UUID"
echo "Local SR UUID: $LOCAL_SR_UUID"
echo "Network UUID: $NET_UUID"
echo ""

echo "STEP 5: Creating VM..."
echo "----------------------"

VM_UUID=$(xe vm-install template-uuid="$TEMPLATE_UUID" new-name-label="$VM_NAME")
echo "VM created: $VM_UUID"

xe vm-param-set uuid="$VM_UUID" affinity="$(xe host-list --minimal)"

xe vm-memory-limits-set uuid="$VM_UUID" \
    static-min="$MEMORY" static-max="$MEMORY" \
    dynamic-min="$MEMORY" dynamic-max="$MEMORY"
echo "Memory set: $MEMORY"

xe vm-param-set uuid="$VM_UUID" VCPUs-max="$VCPUS"
xe vm-param-set uuid="$VM_UUID" VCPUs-at-startup="$VCPUS"
echo "CPUs set: $VCPUS"
echo ""

echo "STEP 6: Adding disk..."
echo "----------------------"

DISK_VDI=$(xe vdi-create sr-uuid="$LOCAL_SR_UUID" name-label="${VM_NAME}-disk0" virtual-size="$DISK_SIZE" type=user)
DISK_VBD=$(xe vbd-create vm-uuid="$VM_UUID" vdi-uuid="$DISK_VDI" device=0 mode=RW type=Disk bootable=true)
echo "Disk created: $DISK_SIZE (device 0)"
echo ""

echo "STEP 7: Adding network interface..."
echo "------------------------------------"

VIF_UUID=$(xe vif-create vm-uuid="$VM_UUID" network-uuid="$NET_UUID" device=0)
echo "Network interface created (device 0, bridge xenbr0)"
echo ""

echo "STEP 8: Adding CD drive and inserting ISO..."
echo "--------------------------------------------"

CD_VBD=$(xe vbd-create vm-uuid="$VM_UUID" device="$CD_DEVICE" type=CD mode=RO)
echo "CD drive created (device $CD_DEVICE)"

if ! verify_sr_accessible "$ISO_SR_UUID"; then
    echo "ERROR: ISO SR became inaccessible"
    exit 1
fi

xe vbd-insert uuid="$CD_VBD" vdi-uuid="$ISO_VDI_UUID"
echo "ISO inserted: $ISO_NAME"
echo ""

echo "STEP 9: Configuring UEFI boot and platform settings..."
echo "-------------------------------------------------------"

xe vm-param-set uuid="$VM_UUID" platform:firmware=uefi
echo "Platform firmware set to UEFI"

xe vm-param-set uuid="$VM_UUID" HVM-boot-params:firmware=uefi
echo "Boot params firmware set to UEFI"

xe vm-param-set uuid="$VM_UUID" platform:viridian=false
echo "Platform VIRIDIAN disabled"

xe vm-param-set uuid="$VM_UUID" HVM-boot-params:viridian=false 2>/dev/null || true
echo "Boot params VIRIDIAN disabled"

xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=dc
echo "Boot order: CD first, then disk"
echo ""

echo "STEP 10: Final verification..."
echo "-------------------------------"

if ! verify_sr_accessible "$ISO_SR_UUID"; then
    echo "ERROR: ISO SR became inaccessible"
    exit 1
fi

CD_ISO=$(xe vbd-param-get uuid="$CD_VBD" param-name=vdi-uuid 2>/dev/null || true)
if [ -z "$CD_ISO" ] || [ "$CD_ISO" != "$ISO_VDI_UUID" ]; then
    echo "ERROR: CD drive does not have ISO inserted"
    exit 1
fi
echo "CD drive has ISO inserted"
echo ""

echo "STEP 11: Starting VM..."
echo "-----------------------"

xe vm-start uuid="$VM_UUID"
sleep 3

DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")
POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")

if [ "$DOMID" != "-1" ] && [ "$POWER_STATE" = "running" ]; then
    echo "VM started successfully!"
    echo "  Domain ID: $DOMID"
    echo "  Power state: $POWER_STATE"
    echo ""
    echo "VNC socket: /var/run/xen/vnc-$DOMID"
    echo ""
    echo "To connect via VNC:"
    echo "  1. On dom0: bash connect_vnc.sh $VM_NAME"
    echo "  2. From Ubuntu: ssh -N -L $VNC_PORT:127.0.0.1:$VNC_PORT root@10.25.33.10"
    echo "  3. Connect VNC client to: 127.0.0.1:$VNC_PORT"
    echo ""
    echo "Ubuntu installer static IP configuration:"
    echo "  IP:      $VM_IP/24"
    echo "  Gateway: $GATEWAY"
    echo "  DNS:     $DNS"
    echo ""
    echo "After installation, run post-installation script:"
    echo "  bash post_install_vm.sh $VM_NAME"
else
    echo "✗ VM failed to start"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  SUCCESS: VM $VM_NAME created and started"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo "Domain ID: $DOMID"
echo "ISO: ${ISO_NAME} - ${ISO_VDI_UUID}"
echo ""
