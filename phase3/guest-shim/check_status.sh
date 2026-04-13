#!/bin/bash
# Status check script - shows current state of shim installation

PASS="Calvin@123"

echo "======================================================================"
echo "SHIM INSTALLATION STATUS CHECK - $(date)"
echo "======================================================================"
echo ""

echo "1. /etc/ld.so.preload:"
if [ -f /etc/ld.so.preload ]; then
    echo "   EXISTS"
    echo "$PASS" | sudo -S cat /etc/ld.so.preload 2>&1 | sed 's/^/   /'
else
    echo "   MISSING"
fi
echo ""

echo "2. LD_AUDIT interceptor:"
if [ -f /usr/lib64/libldaudit_cuda.so ]; then
    echo "   EXISTS"
    ls -la /usr/lib64/libldaudit_cuda.so 2>&1 | sed 's/^/   /'
else
    echo "   MISSING"
fi
echo ""

echo "3. force_load_shim:"
if [ -f /usr/local/bin/force_load_shim ]; then
    echo "   EXISTS"
    ls -la /usr/local/bin/force_load_shim 2>&1 | sed 's/^/   /'
else
    echo "   MISSING"
fi
echo ""

echo "4. Test shim loading:"
if [ -f ~/phase3/guest-shim/test_shim_load.c ]; then
    gcc -o /tmp/test_shim_load ~/phase3/guest-shim/test_shim_load.c -ldl 2>&1
    /tmp/test_shim_load 2>&1 | sed 's/^/   /'
else
    echo "   test_shim_load.c MISSING"
fi
echo ""

echo "5. Ollama service status:"
echo "$PASS" | sudo -S systemctl status ollama --no-pager -l 2>&1 | head -15 | sed 's/^/   /'
echo ""

echo "6. Ollama logs (shim-related):"
echo "$PASS" | sudo -S journalctl -u ollama -n 150 --no-pager 2>&1 | grep -iE "libvgpu|LOADED|cuInit|cuda|gpu" | head -25 | sed 's/^/   /'
echo ""

echo "7. Ollama GPU detection:"
ollama info 2>&1 | head -50 | sed 's/^/   /'
echo ""

echo "======================================================================"
echo "STATUS CHECK COMPLETE"
echo "======================================================================"
