#!/bin/bash

SR_UUID="fc99ca21-ebc8-89f5-dbd3-b7e9a0b2ae49"
SR_MOUNT="/var/run/sr-mount/$SR_UUID"

echo "Checking VGS ISO Storage SR mount..."
echo "SR UUID: $SR_UUID"
echo "Mount point: $SR_MOUNT"
echo ""

if [ -d "$SR_MOUNT" ]; then
    echo "Mount point directory exists"
else
    echo "✗ Mount point directory does NOT exist"
fi

if mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
    echo "Mounted in mount table"
else
    echo "✗ NOT in mount table"
fi

PBD_UUID=$(xe pbd-list sr-uuid="$SR_UUID" params=uuid --minimal)
ATTACHED=$(xe pbd-param-get uuid="$PBD_UUID" param-name=currently-attached)
echo "PBD attached: $ATTACHED"
echo ""

echo "PBD device-config:"
xe pbd-param-get uuid="$PBD_UUID" param-name=device-config
echo ""

if [ -d "$SR_MOUNT" ]; then
    echo "Files in mount:"
    ls -lh "$SR_MOUNT" | head -10
    echo ""
    FILE_COUNT=$(ls -la "$SR_MOUNT" 2>/dev/null | grep -v "^total" | grep -v "^\.$" | grep -v "^\.\.$" | wc -l)
    echo "File count: $FILE_COUNT"
fi
