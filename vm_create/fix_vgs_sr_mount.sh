#!/bin/bash

set -euo pipefail

SR_UUID="fc99ca21-ebc8-89f5-dbd3-b7e9a0b2ae49"
SR_MOUNT="/var/run/sr-mount/$SR_UUID"
ACTUAL_PATH="/mnt/iso-storage"

echo "========================================================================"
echo "  Fixing VGS ISO Storage SR Mount Point"
echo "========================================================================"
echo ""

if [ ! -d "$ACTUAL_PATH" ]; then
    echo "ERROR: Actual path does not exist: $ACTUAL_PATH"
    exit 1
fi

echo "Actual path exists: $ACTUAL_PATH"
echo "Files in actual path:"
ls -lh "$ACTUAL_PATH" | head -5
echo ""

if [ -d "$SR_MOUNT" ]; then
    echo "Mount point already exists: $SR_MOUNT"
    if [ -L "$SR_MOUNT" ]; then
        echo "It's a symlink, checking target..."
        ls -l "$SR_MOUNT"
    fi
else
    echo "Creating mount point directory..."
    mkdir -p "$SR_MOUNT"
    echo "Created: $SR_MOUNT"
fi

if [ -L "$SR_MOUNT" ]; then
    TARGET=$(readlink -f "$SR_MOUNT")
    if [ "$TARGET" = "$ACTUAL_PATH" ]; then
        echo "Mount point is already correctly symlinked to $ACTUAL_PATH"
    else
        echo "⚠ Mount point is symlinked to different location: $TARGET"
        echo "Removing old symlink..."
        rm "$SR_MOUNT"
        echo "Creating new symlink..."
        ln -s "$ACTUAL_PATH" "$SR_MOUNT"
        echo "Created symlink: $SR_MOUNT -> $ACTUAL_PATH"
    fi
else
    if [ "$(ls -A "$SR_MOUNT" 2>/dev/null)" ]; then
        echo "⚠ Mount point exists and has files. Checking if it matches actual path..."
        if [ "$(realpath "$SR_MOUNT")" = "$(realpath "$ACTUAL_PATH")" ]; then
            echo "Mount point already points to correct location"
        else
            echo "⚠ Mount point has different content. Backing up and creating symlink..."
            mv "$SR_MOUNT" "${SR_MOUNT}.backup"
            ln -s "$ACTUAL_PATH" "$SR_MOUNT"
            echo "Created symlink: $SR_MOUNT -> $ACTUAL_PATH"
        fi
    else
        echo "Mount point is empty, replacing with symlink..."
        rmdir "$SR_MOUNT"
        ln -s "$ACTUAL_PATH" "$SR_MOUNT"
        echo "Created symlink: $SR_MOUNT -> $ACTUAL_PATH"
    fi
fi

echo ""
echo "Verifying mount point..."
if [ -L "$SR_MOUNT" ]; then
    TARGET=$(readlink -f "$SR_MOUNT")
    echo "Symlink: $SR_MOUNT -> $TARGET"
    if [ "$TARGET" = "$ACTUAL_PATH" ]; then
        echo "Target is correct"
    fi
fi

if [ -d "$SR_MOUNT" ]; then
    echo "Mount point is accessible"
    echo ""
    echo "Files visible through mount point:"
    ls -lh "$SR_MOUNT" | head -5
    echo ""
    FILE_COUNT=$(ls -la "$SR_MOUNT" 2>/dev/null | grep -v "^total" | grep -v "^\.$" | grep -v "^\.\.$" | wc -l)
    echo "File count: $FILE_COUNT"
    echo ""
    echo "========================================================================"
    echo "  SR mount point is now accessible!"
    echo "========================================================================"
else
    echo "✗ ERROR: Mount point is still not accessible"
    exit 1
fi
