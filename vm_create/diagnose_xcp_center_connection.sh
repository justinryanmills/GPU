#!/bin/bash

echo "=== Checking XCP-ng services ==="
echo ""

echo "1. XAPI (XCP-ng API) service status:"
systemctl status xapi 2>/dev/null | head -5 || service xapi status 2>/dev/null | head -5

echo ""
echo "2. Listening ports (XAPI typically uses 443/80):"
netstat -tlnp 2>/dev/null | grep -E ":(443|80|22)" || ss -tlnp 2>/dev/null | grep -E ":(443|80|22)"

echo ""
echo "3. Firewall status (if firewalld is installed):"
systemctl status firewalld 2>/dev/null | head -3 || echo "firewalld not running"

echo ""
echo "4. iptables rules (checking for blocked connections):"
iptables -L -n 2>/dev/null | head -20 || echo "iptables not accessible"

echo ""
echo "=== Common fixes ==="
echo "If XAPI is not listening on 443/80:"
echo "  - Restart XAPI: systemctl restart xapi"
echo ""
echo "If firewall is blocking:"
echo "  - Allow HTTPS: firewall-cmd --permanent --add-service=https"
echo "  - Allow HTTP:  firewall-cmd --permanent --add-service=http"
echo "  - Reload:      firewall-cmd --reload"
echo ""
echo "If Windows machine can't reach dom0:"
echo "  - Test from Windows: ping 10.25.33.10"
echo "  - Test from Windows: telnet 10.25.33.10 443"
