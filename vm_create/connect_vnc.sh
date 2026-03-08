#!/bin/bash

set -euo pipefail

VM_NAME="${1:-}"

if [ -z "$VM_NAME" ]; then
    echo "ERROR: VM name required"
    echo ""
    echo "Usage: bash connect_vnc.sh <VM_NAME> [PORT]"
    echo ""
    echo "Examples:"
    echo "  bash connect_vnc.sh Test-3"
    echo "  bash connect_vnc.sh Test-4 5902"
    exit 1
fi

if [ -z "${2:-}" ]; then
    VM_NUMBER=$(echo "$VM_NAME" | sed -n 's/.*Test-\([0-9]*\)/\1/p')
    if [ -n "$VM_NUMBER" ]; then
        VNC_PORT=$((5900 + VM_NUMBER))
    else
        VNC_PORT=5901
    fi
else
    VNC_PORT="$2"
fi

echo "========================================================================"
echo "  VNC Connection Helper: $VM_NAME"
echo "========================================================================"
echo ""

VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")
POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")

if [ "$POWER_STATE" != "running" ] || [ "$DOMID" = "-1" ]; then
    echo "ERROR: VM is not running"
    echo "  Power state: $POWER_STATE"
    echo "  Domain ID: $DOMID"
    echo ""
    echo "Start the VM first:"
    echo "  xe vm-start uuid=$VM_UUID"
    exit 1
fi

VNC_SOCKET="/var/run/xen/vnc-$DOMID"

if [ ! -S "$VNC_SOCKET" ]; then
    echo "ERROR: VNC socket does not exist: $VNC_SOCKET"
    echo "VM may still be starting up. Wait a few seconds and try again."
    exit 1
fi

echo "VM UUID: $VM_UUID"
echo "Domain ID: $DOMID"
echo "VNC socket: $VNC_SOCKET"
echo "VNC port: $VNC_PORT"
echo ""

echo "STEP 1: Checking for existing socat processes..."
echo "------------------------------------------------"
EXISTING_PIDS=$(lsof -ti:$VNC_PORT 2>/dev/null || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "Found existing process(es) on port $VNC_PORT:"
    ps -p $EXISTING_PIDS -o pid,cmd || true
    echo ""
    echo "Killing existing process(es)..."
    kill $EXISTING_PIDS 2>/dev/null || true
    sleep 1
    echo "Old processes terminated"
else
    echo "No existing processes on port $VNC_PORT"
fi
echo ""

echo "STEP 2: Starting VNC bridge..."
echo "-------------------------------"
socat TCP-LISTEN:$VNC_PORT,fork,reuseaddr UNIX-CONNECT:"$VNC_SOCKET" &
SOCAT_PID=$!
sleep 1

if ps -p $SOCAT_PID > /dev/null 2>&1; then
    echo "VNC bridge started (PID: $SOCAT_PID)"
    echo "  Listening on: 0.0.0.0:$VNC_PORT"
    echo "  Forwarding to: $VNC_SOCKET"
else
    echo "✗ ERROR: Failed to start VNC bridge"
    exit 1
fi
echo ""

echo "========================================================================"
echo "  VNC Connection Instructions"
echo "========================================================================"
echo ""
echo "1. From your Ubuntu machine, create SSH tunnel:"
echo "   ssh -N -L $VNC_PORT:127.0.0.1:$VNC_PORT root@10.25.33.10"
echo ""
echo "2. Connect VNC client to:"
echo "   Host: 127.0.0.1"
echo "   Port: $VNC_PORT"
echo ""
echo "3. To stop the VNC bridge:"
echo "   kill $SOCAT_PID"
echo ""
echo "========================================================================"
echo "  VNC bridge is running (PID: $SOCAT_PID)"
echo "========================================================================"
