set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RPM_BUILD="$HOME/vgpu-build/rpmbuild"
SOURCES="$RPM_BUILD/SOURCES"
SPECS="$RPM_BUILD/SPECS"

echo "==============================================================="
echo "  Building Enhanced QEMU with vGPU Stub v2 (MMIO)"
echo "==============================================================="
echo ""

echo "[1/7] Verifying build environment..."
if [ ! -d "$RPM_BUILD" ]; then
    echo "ERROR: $RPM_BUILD does not exist."
    echo "       Please set up the build environment first."
    echo "       See the prerequisites section in this script."
    exit 1
fi

if [ ! -f "$SPECS/qemu.spec" ]; then
    echo "ERROR: $SPECS/qemu.spec not found."
    exit 1
fi

echo "  Build directory: $RPM_BUILD"
echo "  Spec file: $SPECS/qemu.spec"
echo ""

echo "[2/7] Copying enhanced source files..."

cp "$SCRIPT_DIR/vgpu-stub-enhanced.c" "$SOURCES/vgpu-stub.c"
echo "  Copied vgpu-stub-enhanced.c → SOURCES/vgpu-stub.c"

cp "$SCRIPT_DIR/vgpu_protocol.h" "$SOURCES/vgpu_protocol.h"
echo "  Copied vgpu_protocol.h → SOURCES/vgpu_protocol.h"

echo ""

echo "[3/7] Checking spec file for vgpu-stub integration..."

if ! grep -q "Source3: vgpu-stub.c" "$SPECS/qemu.spec"; then
    echo "  Adding Source3 line for vgpu-stub.c..."
    sed -i '/^Source2:/a Source3: vgpu-stub.c' "$SPECS/qemu.spec"
    echo "  Added Source3: vgpu-stub.c"
else
    echo "  Source3 already present"
fi

if ! grep -q "Source4: vgpu_protocol.h" "$SPECS/qemu.spec"; then
    echo "  Adding Source4 line for vgpu_protocol.h..."
    sed -i '/^Source3:/a Source4: vgpu_protocol.h' "$SPECS/qemu.spec"
    echo "  Added Source4: vgpu_protocol.h"
else
    echo "  Source4 already present"
fi


if ! grep -q "cp -a %{SOURCE4}" "$SPECS/qemu.spec"; then
    echo "  Adding protocol header copy to spec..."
    if grep -q "cp -a %{SOURCE3}" "$SPECS/qemu.spec"; then
        sed -i '/cp -a %{SOURCE3} hw\/misc\/vgpu-stub.c/a \
cp -a %{SOURCE4} hw/misc/vgpu_protocol.h' "$SPECS/qemu.spec"
        echo "  Added copy of vgpu_protocol.h to hw/misc/"
    else
        sed -i '/tar xzf.*SOURCE2/a \
\
cp -a %{SOURCE3} hw/misc/vgpu-stub.c\
cp -a %{SOURCE4} hw/misc/vgpu_protocol.h\
echo "obj-y += vgpu-stub.o" >> hw/misc/Makefile.objs' \
            "$SPECS/qemu.spec"
        echo "  Added full vgpu-stub integration block"
    fi
else
    echo "  Protocol header copy already in spec"
fi

if ! grep -q "vgpu-stub.o" "$SPECS/qemu.spec"; then
    sed -i '/cp -a %{SOURCE4} hw\/misc\/vgpu_protocol.h/a \
echo "obj-y += vgpu-stub.o" >> hw/misc/Makefile.objs' \
        "$SPECS/qemu.spec"
    echo "  Added Makefile.objs integration"
else
    echo "  Makefile.objs integration present"
fi

if ! grep -q "\-\-disable-werror" "$SPECS/qemu.spec"; then
    sed -i 's/--enable-werror/--disable-werror/' "$SPECS/qemu.spec"
    echo "  Switched to --disable-werror"
else
    echo "  --disable-werror already set"
fi

echo ""

echo "[4/7] Verifying spec file..."
echo -n "  Source3: "
grep "Source3: vgpu-stub.c" "$SPECS/qemu.spec" && true || echo "MISSING!"
echo -n "  Source4: "
grep "Source4: vgpu_protocol.h" "$SPECS/qemu.spec" && true || echo "MISSING!"
echo -n "  Copy stub: "
grep "vgpu-stub.c" "$SPECS/qemu.spec" | grep "cp" | head -1 && true || echo "MISSING!"
echo -n "  Copy header: "
grep "vgpu_protocol.h" "$SPECS/qemu.spec" | grep "cp" | head -1 && true || echo "MISSING!"
echo -n "  Makefile: "
grep "vgpu-stub.o" "$SPECS/qemu.spec" | head -1 && true || echo "MISSING!"
echo -n "  Werror: "
grep "disable-werror" "$SPECS/qemu.spec" | head -1 && true || echo "MISSING!"
echo ""

echo "[5/7] Starting RPM build (this takes 30-45 minutes)..."
echo "  Build log: $RPM_BUILD/BUILD/qemu-4.2.1/build.log"
echo ""

cd "$RPM_BUILD"
rpmbuild -bb --define "_topdir $RPM_BUILD" SPECS/qemu.spec 2>&1 | tee /tmp/qemu-build.log

echo ""
echo "[6/7] Checking build result..."
RPM_FILE=$(ls -1 "$RPM_BUILD/RPMS/x86_64/qemu-4.2.1-"*.rpm 2>/dev/null | head -1)
if [ -z "$RPM_FILE" ]; then
    echo "ERROR: RPM build failed — no RPM file found."
    echo "Check /tmp/qemu-build.log for details."
    exit 1
fi

echo "  Built: $RPM_FILE"
echo ""

echo "[7/7] Installing custom QEMU..."
echo "  ⚠  Ensure all VMs using vgpu-stub are stopped!"
echo ""

rpm -Uvh --nodeps --force "$RPM_FILE"

echo ""
echo "==============================================================="
echo "  Installation complete!"
echo "==============================================================="
echo ""

echo "Post-install verification:"
echo ""

QEMU_BIN="/usr/lib64/xen/bin/qemu-system-i386"
if [ -x "$QEMU_BIN" ]; then
    echo -n "  QEMU version: "
    $QEMU_BIN --version 2>/dev/null | head -1

    echo -n "  vgpu-stub device: "
    if $QEMU_BIN -device help 2>/dev/null | grep -q "vgpu-stub"; then
        echo "AVAILABLE"
    else
        echo "✗ NOT FOUND"
    fi
else
    echo "  ⚠ QEMU binary not found at $QEMU_BIN"
fi

echo ""
echo "Next steps:"
echo "  1. Start a test VM with:"
echo "     xe vm-param-set uuid=<VM_UUID> \\"
echo "       platform:device-model-args=\"-device vgpu-stub,pool_id=B,priority=high,vm_id=200\""
echo "  2. Inside the VM, run the register test program"
echo "  3. Start the mediator daemon on dom0"
echo ""
echo "Done!"
