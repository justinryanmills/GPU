#!/bin/bash

XCP_HOST="10.25.33.10"
XCP_USER="root"
LOCAL_PORT="8443"  # Port on Ubuntu that Windows will connect to
REMOTE_PORT="443"  # XCP-ng API port

echo "Setting up SSH tunnel for XCP-ng Center..."
echo "XCP-ng host: $XCP_HOST"
echo "Local port (on Ubuntu): $LOCAL_PORT"
echo "Remote port (XCP-ng API): $REMOTE_PORT"
echo ""
echo "This will forward: localhost:$LOCAL_PORT -> $XCP_HOST:$REMOTE_PORT"
echo ""
echo "After this tunnel is running:"
echo "  - In XCP-ng Center on Windows, connect to: <UBUNTU_IP>:$LOCAL_PORT"
echo "  - Or if Windows can reach Ubuntu via localhost: localhost:$LOCAL_PORT"
echo ""

if lsof -Pi :$LOCAL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "WARNING: Port $LOCAL_PORT is already in use"
    echo "Killing existing process..."
    lsof -ti :$LOCAL_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo "Starting SSH tunnel (press Ctrl+C to stop)..."
echo ""

ssh -N -L $LOCAL_PORT:$XCP_HOST:$REMOTE_PORT $XCP_USER@$XCP_HOST
