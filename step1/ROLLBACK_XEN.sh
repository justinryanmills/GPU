#!/bin/bash

echo "=== ROLLING BACK TO ORIGINAL XEN ==="

if [ ! -f /boot/xen-4.17.5-13.gz.backup ]; then
    echo "ERROR: Backup file not found!"
    echo "Looking for alternatives..."
    ls -la /boot/xen*.gz*
    exit 1
fi

echo "Restoring original Xen..."
cp /boot/xen-4.17.5-13.gz.backup /boot/xen-4.17.5-13.gz

echo "Updating symlink..."
ln -sf /boot/xen-4.17.5-13.gz /boot/xen.gz

echo ""
echo "Verification:"
ls -la /boot/xen.gz
ls -lh /boot/xen-4.17.5-13.gz

echo ""
echo "Original Xen restored. Ready to reboot."
echo "Run: reboot"

