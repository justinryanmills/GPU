#!/bin/bash

set -euo pipefail

VM_NAME="Test-3"
VM_IP="10.25.33.13"          # Static IP for Ubuntu installer
GATEWAY="10.25.33.254"
DNS="8.8.8.8"
DISK_SIZE="40GiB"
MEMORY="4GiB"
VCPUS=4
CD_DEVICE=3                  # Numeric slot for CD drive (NOT "xvdd")
VNC_PORT=5901

echo "========================================================================"
echo "  Creating VM: $VM_NAME"
echo "========================================================================"
echo ""

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
        if mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
            echo "  Mount is in mount table"
        else
            local SR_TYPE=$(xe sr-list uuid="$SR_UUID" params=type --minimal 2>/dev/null || echo "unknown")
            if [ "$SR_TYPE" = "iso" ]; then
                echo "  File-based ISO SR (mount table check not required)"
            else
                echo "  ⚠ WARNING: Not in mount table, but continuing (may be file-based)"
            fi
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
    echo "Available volumes:"
    lvs | grep -i iso || echo "No ISO volumes found"
    exit 1
fi

VG_NAME=$(echo "$LV_LINE" | awk '{print $1}')
echo "Found VG: $VG_NAME"

LV_PATH="/dev/$VG_NAME/iso_download"
MOUNT_POINT="/mnt/iso-storage"

if [ ! -e "$LV_PATH" ]; then
    echo "Activating logical volume..."
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
    echo "Available files:"
    find "$MOUNT_POINT" -name "*.iso" -type f 2>/dev/null || echo "No ISO files found"
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
        continue  # Skip DVD drives
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
    echo ""
    echo "ERROR: No accessible ISO SR found"
    echo "All ISO SRs are either:"
    echo "  - udev type (DVD drives)"
    echo "  - Not actually mounted (directory exists but mount failed)"
    echo "  - PBD not attached"
    echo "  - Empty or unreadable"
    echo ""
    echo "The ISO file is available at: $ISO_FILE"
    echo "You need to fix an ISO SR or create a local ISO SR via XCP-NG Center GUI"
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
        echo "WARNING: ISO not yet registered, scanning again..."
        xe sr-scan uuid="$ISO_SR_UUID"
        sleep 2
        ISO_VDI_UUID=$(xe vdi-list sr-uuid="$ISO_SR_UUID" name-label="$ISO_NAME" params=uuid --minimal 2>/dev/null | head -1)
    fi
    
    if [ -z "$ISO_VDI_UUID" ]; then
        echo "ERROR: ISO not registered after scan. Check SR mount: $SR_MOUNT"
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
    echo "ERROR: Template 'Other install media' not found"
    exit 1
fi
echo "Template UUID: $TEMPLATE_UUID"

LOCAL_SR_UUID=$(xe sr-list type=lvm content-type=user params=uuid --minimal | head -1)
if [ -z "$LOCAL_SR_UUID" ]; then
    echo "ERROR: No local storage SR found"
    exit 1
fi
echo "Local SR UUID: $LOCAL_SR_UUID"

NET_UUID=$(xe network-list bridge=xenbr0 params=uuid --minimal | head -1)
if [ -z "$NET_UUID" ]; then
    echo "ERROR: Network xenbr0 not found"
    exit 1
fi
echo "Network UUID: $NET_UUID"
echo ""

echo "STEP 5: Creating VM..."
echo "----------------------"

EXISTING_VM=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -n "$EXISTING_VM" ]; then
    echo "WARNING: VM '$VM_NAME' already exists (UUID: $EXISTING_VM)"
    echo "Delete it first or choose a different name"
    exit 1
fi

VM_UUID=$(xe vm-install template-uuid="$TEMPLATE_UUID" new-name-label="$VM_NAME")
echo "VM created: $VM_UUID"

xe vm-param-set uuid="$VM_UUID" affinity=$(xe host-list --minimal)

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
    echo "ERROR: ISO SR became inaccessible. Cannot insert ISO."
    exit 1
fi

xe vbd-insert uuid="$CD_VBD" vdi-uuid="$ISO_VDI_UUID"
echo "ISO inserted: $ISO_NAME"
echo ""

echo "STEP 9: Configuring boot order..."
echo "----------------------------------"

xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=dc
echo "Boot order: CD first, then disk"
echo ""

echo "STEP 10: Final verification..."
echo "-------------------------------"

if ! verify_sr_accessible "$ISO_SR_UUID"; then
    echo ""
    echo "ERROR: ISO SR became inaccessible. VM will not start."
    echo "Fix the SR mount issue before starting the VM."
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
    echo "  1. On dom0, bridge socket to TCP:"
    echo "     socat TCP-LISTEN:$VNC_PORT,fork,reuseaddr UNIX-CONNECT:/var/run/xen/vnc-$DOMID &"
    echo ""
    echo "  2. On your workstation, create SSH tunnel:"
    echo "     ssh -N -L $VNC_PORT:127.0.0.1:$VNC_PORT root@<DOM0_IP>"
    echo ""
    echo "  3. Connect VNC client to: 127.0.0.1:$VNC_PORT"
    echo ""
    echo "Ubuntu installer static IP configuration:"
    echo "  IP:      $VM_IP/24"
    echo "  Gateway: $GATEWAY"
    echo "  DNS:     $DNS"
else
    echo "✗ VM failed to start"
    echo "  Domain ID: $DOMID"
    echo "  Power state: $POWER_STATE"
    echo ""
    echo "Check logs:"
    echo "  tail -50 /var/log/xensource.log | grep -i error"
    if [ "$DOMID" != "-1" ]; then
        echo "  tail -50 /var/log/xen/qemu-dm-$DOMID.log"
    fi
    exit 1
fi

echo ""
echo "========================================================================"
echo "  SUCCESS: VM $VM_NAME created and started"
echo "========================================================================"
echo "VM UUID: $VM_UUID"
echo "Domain ID: $DOMID"
echo "ISO: $ISO_NAME ($ISO_VDI_UUID)"
echo ""
