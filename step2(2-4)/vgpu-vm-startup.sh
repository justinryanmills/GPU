#!/bin/bash

VGPU_ADMIN="/etc/vgpu/vgpu-admin"
DB_PATH="/etc/vgpu/vgpu_config.db"

if [ -z "$1" ]; then
    echo "Error: VM UUID required" >&2
    echo "Usage: $0 <vm-uuid>" >&2
    exit 1
fi

VM_UUID="$1"

if [ ! -f "$DB_PATH" ]; then
    echo "Warning: Database not found at $DB_PATH" >&2
    echo "Run: sqlite3 $DB_PATH < /etc/vgpu/init_db.sql" >&2
    exit 1
fi

if [ ! -x "$VGPU_ADMIN" ]; then
    echo "Error: vgpu-admin not found at $VGPU_ADMIN" >&2
    exit 1
fi

CONFIG=$("$VGPU_ADMIN" show-vm --vm-uuid="$VM_UUID" 2>/dev/null)

if [ $? -ne 0 ] || [ -z "$CONFIG" ]; then
    exit 0
fi

POOL_ID=$(echo "$CONFIG" | grep "Pool:" | awk '{print $2}')
PRIORITY_STR=$(echo "$CONFIG" | grep "Priority:" | awk '{print $2}')
VM_ID=$(echo "$CONFIG" | grep "VM ID:" | awk '{print $3}')

case "$PRIORITY_STR" in
    low)    PRIORITY=0 ;;
    medium) PRIORITY=1 ;;
    high)   PRIORITY=2 ;;
    *)      PRIORITY=1 ;;  # Default to medium
esac

xe vm-param-set uuid="$VM_UUID" \
   platform:device-model-args="-device vgpu-stub,pool_id=$POOL_ID,priority=$PRIORITY,vm_id=$VM_ID" \
   2>/dev/null

if [ $? -eq 0 ]; then
    echo "vGPU configuration applied to VM $VM_UUID: Pool=$POOL_ID, Priority=$PRIORITY_STR, VM_ID=$VM_ID"
else
    echo "Warning: Failed to set device-model-args for VM $VM_UUID" >&2
    exit 1
fi

exit 0
