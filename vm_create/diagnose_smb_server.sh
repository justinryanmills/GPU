#!/bin/bash

set -euo pipefail

SMB_SERVER="10.25.33.33"
SMB_SHARE="/test"
SMB_USER="momi"
SMB_SR_UUID="097e2b8c-af1a-d945-1432-8c0e7d0163fa"

echo "========================================================================"
echo "  SMB Server Connectivity Diagnostic"
echo "========================================================================"
echo "SMB Server: $SMB_SERVER"
echo "Share: $SMB_SHARE"
echo ""

echo "STEP 1: Testing basic network connectivity..."
echo "---------------------------------------------"

if ping -c 3 -W 2 "$SMB_SERVER" >/dev/null 2>&1; then
    echo "Ping successful - server is reachable at network level"
    PING_OK=true
else
    echo "✗ Ping failed - server is NOT reachable"
    echo "  This means:"
    echo "    - Server may be down/offline"
    echo "    - Network routing issue"
    echo "    - Firewall blocking ICMP"
    PING_OK=false
fi
echo ""

echo "STEP 2: Testing SMB port (445)..."
echo "----------------------------------"

if command -v nc >/dev/null 2>&1; then
    if timeout 3 nc -zv "$SMB_SERVER" 445 >/dev/null 2>&1; then
        echo "Port 445 is open - SMB service is listening"
        PORT_OK=true
    else
        echo "✗ Port 445 is closed/filtered"
        echo "  This means:"
        echo "    - SMB service not running on server"
        echo "    - Firewall blocking port 445"
        PORT_OK=false
    fi
else
    echo "⚠ nc (netcat) not installed - skipping port test"
    echo "  Install with: yum install -y nc"
    PORT_OK=unknown
fi
echo ""

echo "STEP 3: Checking network routing..."
echo "-----------------------------------"

echo "Dom0 IP addresses:"
ip addr show | grep "inet " | grep -v "127.0.0.1" || echo "  (no IPs found)"

echo ""
echo "Routing table (relevant entries):"
ip route | grep -E "10\.25\.33|default" || echo "  (no relevant routes found)"

echo ""
echo "Can dom0 reach 10.25.33.33 network?"
if ip route get "$SMB_SERVER" >/dev/null 2>&1; then
    ROUTE=$(ip route get "$SMB_SERVER" 2>/dev/null)
    echo "  Route: $ROUTE"
else
    echo "  ✗ No route found to $SMB_SERVER"
fi
echo ""

echo "STEP 4: Checking firewall rules..."
echo "-----------------------------------"

if systemctl is-active --quiet firewalld 2>/dev/null; then
    echo "firewalld is active"
    echo "Checking SMB/CIFS rules:"
    firewall-cmd --list-all 2>/dev/null | grep -E "services:|ports:" || echo "  (no SMB rules found)"
elif iptables -L -n >/dev/null 2>&1; then
    echo "iptables is active"
    echo "Checking for blocked connections to $SMB_SERVER:"
    iptables -L -n | grep -E "$SMB_SERVER|445" || echo "  (no specific rules found)"
else
    echo "No firewall detected (or not accessible)"
fi
echo ""

echo "STEP 5: Testing manual CIFS mount..."
echo "-------------------------------------"

if command -v mount.cifs >/dev/null 2>&1; then
    TEST_MOUNT="/tmp/test-smb-$$"
    mkdir -p "$TEST_MOUNT"
    
    echo "Attempting manual mount (this will fail if server is unreachable)..."
    
    if [ "$PING_OK" = "true" ] && [ "$PORT_OK" = "true" ]; then
        echo "  Server is reachable - mount should work if credentials are correct"
        echo "  To test manually:"
        echo "    mount -t cifs //$SMB_SERVER$SMB_SHARE $TEST_MOUNT -o username=$SMB_USER"
    else
        echo "  ✗ Cannot test mount - server is not reachable"
    fi
    
    rmdir "$TEST_MOUNT" 2>/dev/null || true
else
    echo "⚠ mount.cifs not installed"
    echo "  Install with: yum install -y cifs-utils"
fi
echo ""

echo "STEP 6: Checking XCP-ng PBD configuration..."
echo "---------------------------------------------"

PBD_UUID=$(xe pbd-list sr-uuid="$SMB_SR_UUID" params=uuid --minimal | head -1)
if [ -n "$PBD_UUID" ]; then
    echo "PBD UUID: $PBD_UUID"
    echo "Device config:"
    xe pbd-list uuid="$PBD_UUID" params=device-config --minimal
    
    echo ""
    echo "Currently attached:"
    xe pbd-list uuid="$PBD_UUID" params=currently-attached --minimal
else
    echo "✗ No PBD found for SMB SR"
fi
echo ""

echo "========================================================================"
echo "  DIAGNOSIS SUMMARY"
echo "========================================================================"
echo ""

if [ "$PING_OK" = "false" ]; then
    echo "❌ PRIMARY ISSUE: SMB server is not reachable at network level"
    echo ""
    echo "SOLUTIONS:"
    echo "----------"
    echo "1. Verify SMB server (10.25.33.33) is running and online"
    echo "2. Check network connectivity between dom0 and SMB server:"
    echo "   - Are they on the same network/subnet?"
    echo "   - Is there a router/firewall between them?"
    echo "3. Check dom0 network configuration:"
    echo "   - ip addr show"
    echo "   - ip route show"
    echo "4. If SMB server IP changed, update PBD device-config:"
    echo "   xe pbd-param-set uuid=<PBD_UUID> device-config:location=//<NEW_IP>/test"
    echo ""
elif [ "$PORT_OK" = "false" ]; then
    echo "⚠ SERVER REACHABLE but SMB port (445) is closed"
    echo ""
    echo "SOLUTIONS:"
    echo "----------"
    echo "1. Verify SMB service is running on 10.25.33.33"
    echo "2. Check firewall on SMB server (allow port 445)"
    echo "3. Check if SMB server changed ports"
    echo ""
else
    echo "Network connectivity appears OK"
    echo "  Issue may be:"
    echo "    - SMB credentials incorrect"
    echo "    - Share path changed"
    echo "    - SMB version mismatch"
    echo ""
    echo "Try updating PBD credentials or share path"
fi

echo ""
echo "ALTERNATIVE: If SMB cannot be fixed, use XCP-ng Center GUI"
echo "  (via SSH tunnel or Xen Orchestra web interface)"
