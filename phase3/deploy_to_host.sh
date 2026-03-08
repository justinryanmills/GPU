#!/usr/bin/env bash

set -euo pipefail

HOST="root@10.25.33.10"
GUEST="test-3@10.25.33.11"
VM_UUID="57e01e2c-ae40-f429-0a28-cf475b79e58c"
LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)"   # /home/david/Downloads/gpu

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SCP="scp $SSH_OPTS"
SSH="ssh $SSH_OPTS"

info()  { echo "[deploy] $*"; }
step()  { echo ""; echo "================================================================="; echo "  $*"; echo "================================================================="; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

for f in \
    phase3/src/vgpu-stub-enhanced.c \
    phase3/src/mediator_phase3.c \
    phase3/include/vgpu_protocol.h \
    phase3/include/cuda_protocol.h \
    phase3/guest-shim/cuda_transport.c \
    phase3/guest-shim/install.sh; do
    [[ -f "$LOCAL_BASE/$f" ]] || die "Local file missing: $LOCAL_BASE/$f"
done
info "All local source files present."

step "Step 1 — Push updated sources to XCP-NG host ($HOST)"

$SCP "$LOCAL_BASE/phase3/src/vgpu-stub-enhanced.c" \
     "$HOST:/root/phase3/src/vgpu-stub-enhanced.c"
info "vgpu-stub-enhanced.c"

$SCP "$LOCAL_BASE/phase3/src/mediator_phase3.c" \
     "$HOST:/root/phase3/src/mediator_phase3.c"
info "mediator_phase3.c"

$SCP "$LOCAL_BASE/phase3/include/vgpu_protocol.h" \
     "$HOST:/root/phase3/include/vgpu_protocol.h"
info "vgpu_protocol.h"

$SCP "$LOCAL_BASE/phase3/include/cuda_protocol.h" \
     "$HOST:/root/phase3/include/cuda_protocol.h"
info "cuda_protocol.h"

step "Step 2 — Rebuild mediator_phase3 on host"

$SSH "$HOST" bash -s << 'REMOTE_MEDIATOR'
set -euo pipefail
cd /root/phase3
echo "[host] Rebuilding mediator..."
make clean 2>&1 | tail -3
make host  2>&1 | tail -20
echo "[host] mediator_phase3 rebuilt"
REMOTE_MEDIATOR

step "Step 3 — Prepare QEMU RPM sources"

$SSH "$HOST" bash -s << 'REMOTE_PREPARE'
set -euo pipefail
cd /root/phase3
echo "[host] Running make qemu-prepare..."
make qemu-prepare
echo "[host] Sources staged in ~/vgpu-build/rpmbuild/SOURCES/"
REMOTE_PREPARE

step "Step 4 — Build QEMU RPM (30-45 min — please wait)"

info "This step takes 30-45 minutes.  Output will stream below."
info "Build log is also written to /tmp/qemu-build.log on the host."

$SSH "$HOST" bash -s << 'REMOTE_BUILD'
set -euo pipefail
cd /root/phase3
make qemu-build
echo "[host] QEMU RPM built"
REMOTE_BUILD

step "Step 5 — Stop VM, install QEMU RPM, restart VM + mediator"

VM_UUID_VAR="$VM_UUID"  # export to subshell

$SSH "$HOST" bash -s << REMOTE_INSTALL
set -euo pipefail
VM_UUID="$VM_UUID_VAR"

echo "[host] Stopping VM \$VM_UUID..."
xe vm-shutdown uuid="\$VM_UUID" force=true 2>/dev/null || true
sleep 5

echo "[host] Installing new QEMU RPM..."
RPM_FILE=\$(ls -1 ~/vgpu-build/rpmbuild/RPMS/x86_64/qemu-*.rpm 2>/dev/null | head -1)
[[ -z "\$RPM_FILE" ]] && { echo "ERROR: no RPM file found"; exit 1; }
echo "[host] RPM: \$RPM_FILE"
rpm -Uvh --nodeps --force "\$RPM_FILE"
echo "[host] QEMU RPM installed"

QEMU_BIN="/usr/lib64/xen/bin/qemu-system-i386"
if [[ -x "\$QEMU_BIN" ]]; then
    if "\$QEMU_BIN" -device help 2>/dev/null | grep -q "vgpu-cuda"; then
        echo "[host] vgpu-cuda device present in new QEMU binary"
    else
        echo "[host] WARNING: vgpu-cuda device NOT found in QEMU binary — check the build"
    fi
fi

echo "[host] Starting VM \$VM_UUID..."
xe vm-start uuid="\$VM_UUID"
echo "[host] VM started"

echo "[host] Restarting mediator..."
pkill mediator_phase3 2>/dev/null || true
sleep 2
cd /root/phase3
nohup ./mediator_phase3 > /var/log/mediator_phase3.log 2>&1 &
sleep 3
if pgrep -x mediator_phase3 > /dev/null; then
    echo "[host] mediator_phase3 running (PID=\$(pgrep -x mediator_phase3))"
else
    echo "[host] ERROR: mediator_phase3 failed to start — check /var/log/mediator_phase3.log"
fi
REMOTE_INSTALL

step "Step 6 — Push updated guest shim to VM ($GUEST)"

info "Waiting 30 s for VM to finish booting..."
sleep 30

$SCP "$LOCAL_BASE/phase3/guest-shim/cuda_transport.c" \
     "$GUEST:/home/test-3/phase3/cuda_transport.c"
info "cuda_transport.c"

$SCP "$LOCAL_BASE/phase3/guest-shim/install.sh" \
     "$GUEST:/home/test-3/phase3/install.sh"
info "install.sh"

$SCP "$LOCAL_BASE/phase3/include/vgpu_protocol.h" \
     "$GUEST:/home/test-3/phase3/include/vgpu_protocol.h"
info "vgpu_protocol.h (guest copy)"

$SCP "$LOCAL_BASE/phase3/include/cuda_protocol.h" \
     "$GUEST:/home/test-3/phase3/include/cuda_protocol.h"
info "cuda_protocol.h (guest copy)"

step "Step 7 — Run install.sh inside VM"

$SSH "$GUEST" bash -s << 'REMOTE_INSTALL_SH'
set -euo pipefail
cd ~/phase3
chmod +x install.sh
sudo ./install.sh
REMOTE_INSTALL_SH

step "Verification hints"

echo ""
echo "  Self-test should show:"
echo "    caps=0x00000071   ← bit 6 (VGPU_CAP_SHMEM) now set"
echo "    [self-test] PASS: doorbell->mediator round-trip OK"
echo ""
echo "  Run inside VM to confirm GPU is being used:"
echo "    ollama run llama3.2:1b 'Hello'"
echo ""
echo "  Run on host to watch mediator stats:"
echo "    tail -f /var/log/mediator_phase3.log"
echo "  — 'Total processed' should increment with each Ollama prompt."
echo ""
echo "==================================================================="
echo "  Deployment complete!"
echo "==================================================================="
