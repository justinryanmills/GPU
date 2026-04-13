#!/bin/bash
# ============================================================================
# Guest VM Setup Script — VGPU Shim Libraries + Ollama
#
# This script is run INSIDE the guest VM. It:
#   1. Compiles the CUDA shim (libvgpu-cuda.so) and NVML shim (libvgpu-nvml.so)
#   2. Installs them as drop-in replacements for NVIDIA libraries
#   3. Optionally installs Ollama for LLM inference testing
#   4. Creates /dev/nvidia* stub device nodes
#   5. Sets up LD_LIBRARY_PATH and environment for applications
#
# Prerequisites (on the guest VM):
#   - gcc installed (build-essential on Debian/Ubuntu)
#   - The vgpu-cuda PCI device must be visible (lspci shows it)
#   - Root access
#
# Usage:
#   sudo ./install.sh [--with-ollama] [--uninstall] [--check]
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_LIB_DIR="/usr/lib64"
INSTALL_BIN_DIR="/usr/local/bin"
SHIM_SRC_DIR="${SCRIPT_DIR}"

# Locate include directory — works whether install.sh is run from:
#   guest-shim/  (SCRIPT_DIR/../include = phase3/include)  ← canonical
#   phase3/      (SCRIPT_DIR/include    = phase3/include)  ← common mistake
if [[ -f "${SCRIPT_DIR}/../include/cuda_protocol.h" ]]; then
    INCLUDE_DIR="${SCRIPT_DIR}/../include"
elif [[ -f "${SCRIPT_DIR}/include/cuda_protocol.h" ]]; then
    INCLUDE_DIR="${SCRIPT_DIR}/include"
    # Also update SHIM_SRC_DIR: if run from phase3/, shim sources are in guest-shim/
    if [[ -d "${SCRIPT_DIR}/guest-shim" ]]; then
        SHIM_SRC_DIR="${SCRIPT_DIR}/guest-shim"
    fi
else
    # Fall through — will fail with a clear message inside build_shims()
    INCLUDE_DIR="${SCRIPT_DIR}/../include"
fi

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No colour

log()   { echo -e "${GREEN}[vgpu-shim]${NC} $*"; }
warn()  { echo -e "${YELLOW}[vgpu-shim]${NC} $*"; }
error() { echo -e "${RED}[vgpu-shim]${NC} $*" >&2; }
info()  { echo -e "${BLUE}[vgpu-shim]${NC} $*"; }

# ============================================================================
# Check: verify VGPU-STUB PCI device is present
# ============================================================================
check_vgpu_device() {
    log "Checking for VGPU-STUB PCI device..."

    local found=0
    for dev_dir in /sys/bus/pci/devices/*; do
        local vendor device class
        vendor=$(cat "${dev_dir}/vendor" 2>/dev/null || echo "0x0000")
        device=$(cat "${dev_dir}/device" 2>/dev/null || echo "0x0000")
        class=$(cat "${dev_dir}/class" 2>/dev/null || echo "0x000000")

        # vGPU device: vendor=0x10de (NVIDIA), device=0x2331 (H100 PCIe)
        if [[ "$vendor" == "0x10de" && "$device" == "0x2331" ]]; then
            local dev_name
            dev_name=$(basename "$dev_dir")
            log "  Found vGPU device at ${dev_name} (class=${class})"
            found=1

            # Check if resource files exist
            if [[ -f "${dev_dir}/resource0" ]]; then
                log "  BAR0 resource available"
            else
                warn "  BAR0 resource not found"
            fi

            if [[ -f "${dev_dir}/resource1" ]]; then
                log "  BAR1 resource available (16 MB data region)"
            else
                warn "  BAR1 resource not found (will use BAR0 fallback)"
            fi

            # Grant non-root processes read/write access to BAR0 and BAR1.
            # Without this the ollama user (and any other non-root process)
            # cannot open resource0/resource1 for MMIO mapping, causing
            # cuda_transport_init() to fail → cuInit() → CUDA_ERROR_NO_DEVICE.
            for res in resource0 resource1; do
                local rpath="${dev_dir}/${res}"
                if [[ -f "$rpath" ]]; then
                    chmod 0666 "$rpath" \
                        && log "  ✓ ${rpath} → 0666 (non-root MMIO access)" \
                        || warn "  Could not chmod ${rpath}"
                fi
            done
            break
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        error "VGPU-STUB PCI device not found!"
        error ""
        error "The device must be added to the VM configuration."
        error "On the XCP-ng host, run:"
        error '  xe vm-param-set uuid=<VM_UUID> \'
        error '    platform:device-model-args="-device vgpu-cuda,pool_id=B,priority=high,vm_id=200"'
        error ""
        error "Then restart the VM."
        return 1
    fi

    return 0
}

# ============================================================================
# Build shim libraries
# ============================================================================
build_shims() {
    log "Building shim libraries..."

    # Check for GCC
    if ! command -v gcc &>/dev/null; then
        error "gcc not found. Install build tools:"
        error "  Ubuntu/Debian: sudo apt install build-essential"
        error "  RHEL/CentOS:   sudo yum install gcc make"
        return 1
    fi

    # Check required source files
    for f in libvgpu_cuda.c libvgpu_nvml.c cuda_transport.c cuda_transport.h \
             gpu_properties.h; do
        if [[ ! -f "${SHIM_SRC_DIR}/${f}" ]]; then
            error "Missing source file: ${SHIM_SRC_DIR}/${f}"
            return 1
        fi
    done

    if [[ ! -f "${INCLUDE_DIR}/cuda_protocol.h" ]]; then
        error "Missing header: ${INCLUDE_DIR}/cuda_protocol.h"
        return 1
    fi

    # Build syscall interception library
    log "  Compiling libvgpu-syscall.so ..."
    if [[ -f "${SHIM_SRC_DIR}/libvgpu_syscall.c" ]]; then
        if gcc -shared -fPIC -o "${SHIM_SRC_DIR}/libvgpu-syscall.so" \
           "${SHIM_SRC_DIR}/libvgpu_syscall.c" \
           -ldl -lpthread \
           2>&1 | while IFS= read -r l; do warn "    $l"; done; then
            log "  ✓ libvgpu-syscall.so built"
        else
            error "  ✗ Failed to build libvgpu-syscall.so"
            return 1
        fi
    else
        warn "  libvgpu_syscall.c not found, skipping syscall interception"
    fi
    
    # Build process spawning interception library
    log "  Compiling libvgpu-exec.so ..."
    if [[ -f "${SHIM_SRC_DIR}/libvgpu_exec.c" ]]; then
        if gcc -shared -fPIC -o "${SHIM_SRC_DIR}/libvgpu-exec.so" \
           "${SHIM_SRC_DIR}/libvgpu_exec.c" \
           -ldl \
           2>&1 | while IFS= read -r l; do warn "    $l"; done; then
            log "  ✓ libvgpu-exec.so built"
        else
            error "  ✗ Failed to build libvgpu-exec.so"
            return 1
        fi
    else
        warn "  libvgpu_exec.c not found, skipping exec interception"
    fi
    
    # Build process monitor daemon
    log "  Compiling process_monitor ..."
    if [[ -f "${SHIM_SRC_DIR}/process_monitor.c" ]]; then
        if gcc -o "${SHIM_SRC_DIR}/process_monitor" \
           "${SHIM_SRC_DIR}/process_monitor.c" \
           -lpthread \
           2>&1 | while IFS= read -r l; do warn "    $l"; done; then
            log "  ✓ process_monitor built"
        else
            warn "  ✗ Failed to build process_monitor (non-fatal)"
        fi
    else
        warn "  process_monitor.c not found, skipping process monitor"
    fi
    
    # Build CUDA shim
    log "  Compiling libvgpu-cuda.so ..."
    # -Wl,-soname,libcuda.so.1  — sets the ELF SONAME so that LD_PRELOAD works
    # correctly with runtime dlopen("libcuda.so.1") calls: the dynamic linker
    # sees the already-mapped library matches the requested soname and reuses it
    # rather than loading a second, disconnected copy.
    gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
        -I"${INCLUDE_DIR}" -I"${SHIM_SRC_DIR}" \
        -Wl,-soname,libcuda.so.1 \
        -o "${SHIM_SRC_DIR}/libvgpu-cuda.so" \
        "${SHIM_SRC_DIR}/libvgpu_cuda.c" \
        "${SHIM_SRC_DIR}/cuda_transport.c" \
        -lpthread -ldl
    log "  ✓ libvgpu-cuda.so built"

    # Build NVML shim
    # Note: cuda_transport.c uses libvgpu_set_skip_interception() which is
    # defined in libvgpu-cuda.so. We allow undefined symbols here because
    # both libraries will be loaded together (via LD_PRELOAD or symlinks),
    # so the symbol will be available via RTLD_DEFAULT at runtime.
    log "  Compiling libvgpu-nvml.so ..."
    gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
        -I"${INCLUDE_DIR}" -I"${SHIM_SRC_DIR}" \
        -Wl,-soname,libnvidia-ml.so.1 \
        -Wl,--allow-shlib-undefined \
        -o "${SHIM_SRC_DIR}/libvgpu-nvml.so" \
        "${SHIM_SRC_DIR}/libvgpu_nvml.c" \
        "${SHIM_SRC_DIR}/cuda_transport.c" \
        -lpthread -ldl
    log "  ✓ libvgpu-nvml.so built"
    
    log "  Compiling libvgpu-cudart.so ..."
    # Build with version script to add @libcudart.so.12 version symbols
    # This ensures libggml-cuda.so can resolve versioned symbols like cudaGetDeviceCount@libcudart.so.12
    if [[ -f "${SHIM_SRC_DIR}/libcudart.so.12.versionscript" ]]; then
        gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
            -I"${INCLUDE_DIR}" -I"${SHIM_SRC_DIR}" \
            -Wl,--version-script="${SHIM_SRC_DIR}/libcudart.so.12.versionscript" \
            -Wl,-soname,libcudart.so.12 \
            -o "${SHIM_SRC_DIR}/libvgpu-cudart.so" \
            "${SHIM_SRC_DIR}/libvgpu_cudart.c" \
            -ldl -lpthread
    else
        # Fallback if version script doesn't exist
        gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
            -I"${INCLUDE_DIR}" -I"${SHIM_SRC_DIR}" \
            -Wl,-soname,libcudart.so.12 \
            -o "${SHIM_SRC_DIR}/libvgpu-cudart.so" \
            "${SHIM_SRC_DIR}/libvgpu_cudart.c" \
            -ldl -lpthread
    fi
    log "  ✓ libvgpu-cudart.so built"
}

# ============================================================================
# Install shim libraries
# ============================================================================
install_shims() {
    log "Installing shim libraries..."

    # Ensure install directory exists
    mkdir -p "${INSTALL_LIB_DIR}"

    # Back up existing NVIDIA libraries if they exist (and aren't already our shims)
    for lib in libcuda.so.1 libcuda.so libnvidia-ml.so.1 libnvidia-ml.so; do
        local target="${INSTALL_LIB_DIR}/${lib}"
        if [[ -f "$target" && ! -L "$target" ]]; then
            warn "  Backing up existing ${lib} → ${lib}.real"
            mv "$target" "${target}.real"
        elif [[ -L "$target" ]]; then
            local link_target
            link_target=$(readlink -f "$target" 2>/dev/null || true)
            if [[ "$link_target" == *"libvgpu"* ]]; then
                log "  ${lib} already points to vgpu shim"
            fi
        fi
    done

    # Install shim libraries
    install -m 755 "${SHIM_SRC_DIR}/libvgpu-cuda.so" "${INSTALL_LIB_DIR}/"
    log "  ✓ Installed libvgpu-cuda.so → ${INSTALL_LIB_DIR}/"

    install -m 755 "${SHIM_SRC_DIR}/libvgpu-nvml.so" "${INSTALL_LIB_DIR}/"
    log "  ✓ Installed libvgpu-nvml.so → ${INSTALL_LIB_DIR}/"
    
    install -m 755 "${SHIM_SRC_DIR}/libvgpu-cudart.so" "${INSTALL_LIB_DIR}/"
    log "  ✓ Installed libvgpu-cudart.so → ${INSTALL_LIB_DIR}/"

    # Create symlinks
    ln -sf "${INSTALL_LIB_DIR}/libvgpu-cuda.so" "${INSTALL_LIB_DIR}/libcuda.so.1"
    ln -sf "${INSTALL_LIB_DIR}/libvgpu-cuda.so" "${INSTALL_LIB_DIR}/libcuda.so"
    log "  ✓ libcuda.so.1 → libvgpu-cuda.so"

    ln -sf "${INSTALL_LIB_DIR}/libvgpu-nvml.so" "${INSTALL_LIB_DIR}/libnvidia-ml.so.1"
    ln -sf "${INSTALL_LIB_DIR}/libvgpu-nvml.so" "${INSTALL_LIB_DIR}/libnvidia-ml.so"
    log "  ✓ libnvidia-ml.so.1 → libvgpu-nvml.so"

    # Also install in standard system library paths for better compatibility
    for syslib_dir in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
        if [[ -d "$syslib_dir" ]]; then
            # Backup existing libraries if they exist
            for lib in libcuda.so.1 libcuda.so libnvidia-ml.so.1 libnvidia-ml.so; do
                local target="${syslib_dir}/${lib}"
                if [[ -f "$target" && ! -L "$target" ]] && [[ "$target" != *"libvgpu"* ]]; then
                    mv "$target" "${target}.real" 2>/dev/null || true
                fi
            done
            # Install shims
            cp -f "${INSTALL_LIB_DIR}/libvgpu-cuda.so" "${syslib_dir}/libcuda.so.1" 2>/dev/null && \
            cp -f "${INSTALL_LIB_DIR}/libvgpu-nvml.so" "${syslib_dir}/libnvidia-ml.so.1" 2>/dev/null && \
            ln -sf "libcuda.so.1" "${syslib_dir}/libcuda.so" 2>/dev/null && \
            ln -sf "libnvidia-ml.so.1" "${syslib_dir}/libnvidia-ml.so" 2>/dev/null && \
            log "  ✓ Installed shims in ${syslib_dir}" || true
        fi
    done

    # Blacklist real NVIDIA kernel modules to prevent them from binding
    # to the vGPU PCI device (which now presents NVIDIA vendor/device IDs)
    local blacklist_file="/etc/modprobe.d/blacklist-nvidia-real.conf"
    cat > "${blacklist_file}" <<'BLACKLIST_EOF'
# Prevent real NVIDIA kernel drivers from binding to the vGPU device.
# GPU compute is handled by the vGPU CUDA remoting layer, not a local driver.
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidia_uvm
BLACKLIST_EOF
    log "  ✓ NVIDIA kernel modules blacklisted (${blacklist_file})"

    # Update library cache
    ldconfig 2>/dev/null || true
    log "  ✓ Library cache updated"
}

# ============================================================================
# Create NVIDIA device nodes (stubs)
# ============================================================================
create_device_nodes() {
    log "Creating NVIDIA device nodes..."

    # nvidia0 — primary GPU
    if [[ ! -e /dev/nvidia0 ]]; then
        mknod /dev/nvidia0 c 195 0 2>/dev/null || true
        chmod 666 /dev/nvidia0 2>/dev/null || true
        log "  ✓ /dev/nvidia0 created"
    else
        log "  /dev/nvidia0 already exists"
    fi

    # nvidiactl — control device
    if [[ ! -e /dev/nvidiactl ]]; then
        mknod /dev/nvidiactl c 195 255 2>/dev/null || true
        chmod 666 /dev/nvidiactl 2>/dev/null || true
        log "  ✓ /dev/nvidiactl created"
    else
        log "  /dev/nvidiactl already exists"
    fi

    # nvidia-uvm — unified virtual memory
    if [[ ! -e /dev/nvidia-uvm ]]; then
        mknod /dev/nvidia-uvm c 510 0 2>/dev/null || true
        chmod 666 /dev/nvidia-uvm 2>/dev/null || true
        log "  ✓ /dev/nvidia-uvm created"
    else
        log "  /dev/nvidia-uvm already exists"
    fi

    # nvidia-uvm-tools
    if [[ ! -e /dev/nvidia-uvm-tools ]]; then
        mknod /dev/nvidia-uvm-tools c 510 1 2>/dev/null || true
        chmod 666 /dev/nvidia-uvm-tools 2>/dev/null || true
        log "  ✓ /dev/nvidia-uvm-tools created"
    else
        log "  /dev/nvidia-uvm-tools already exists"
    fi
}

# ============================================================================
# Create udev rules for persistent device nodes
# ============================================================================
create_udev_rules() {
    log "Creating udev rules..."

    cat > /etc/udev/rules.d/99-vgpu-nvidia.rules <<'EOF'
# VGPU shim — create NVIDIA device nodes on boot
KERNEL=="nvidia[0-9]*", RUN+="/bin/chmod 666 /dev/%k"
KERNEL=="nvidiactl", RUN+="/bin/chmod 666 /dev/%k"
KERNEL=="nvidia-uvm", RUN+="/bin/chmod 666 /dev/%k"
KERNEL=="nvidia-uvm-tools", RUN+="/bin/chmod 666 /dev/%k"

# Grant non-root processes (e.g. the 'ollama' user) read/write access to the
# vGPU PCI BAR resource files.  Without this, cuda_transport_init() fails with
# EACCES when called from an unprivileged process, causing cuInit() to return
# CUDA_ERROR_NO_DEVICE and Ollama to fall back to CPU inference.
# Vendor 0x10de = NVIDIA, Device 0x2331 = H100 PCIe (vgpu-cuda stub ID)
SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{device}=="0x2331", \
    RUN+="/bin/chmod 0666 /sys%p/resource0 /sys%p/resource1"
EOF

    log "  ✓ udev rules written to /etc/udev/rules.d/99-vgpu-nvidia.rules"
}

# ============================================================================
# Ensure /usr/lib64 is in the dynamic linker search path (ldconfig)
#
# On Ubuntu/Debian, /usr/lib64 is NOT in the default ld.so.conf search paths.
# Without this, dlopen("libcuda.so.1") from libggml-cuda.so would fail even
# though the file exists at /usr/lib64/libcuda.so.1.
# ============================================================================
ensure_ldconfig_path() {
    local conf="/etc/ld.so.conf.d/vgpu-lib64.conf"
    local needs_lib64=0
    local needs_ollama=0
    
    if ! ldconfig -p 2>/dev/null | grep -q '/usr/lib64'; then
        needs_lib64=1
    fi
    
    # Also add /usr/local/lib/ollama so libggml-base.so.0 is found
    # even if LD_LIBRARY_PATH isn't applied correctly
    if [[ -d "/usr/local/lib/ollama" ]] && \
       ! ldconfig -p 2>/dev/null | grep -q '/usr/local/lib/ollama'; then
        needs_ollama=1
    fi
    
    if [[ "$needs_lib64" -eq 1 ]] || [[ "$needs_ollama" -eq 1 ]]; then
        {
            [[ "$needs_lib64" -eq 1 ]] && echo "/usr/lib64"
            [[ "$needs_ollama" -eq 1 ]] && echo "/usr/local/lib/ollama"
        } > "$conf"
        ldconfig 2>/dev/null || true
        if [[ "$needs_lib64" -eq 1 ]]; then
            log "  ✓ Added /usr/lib64 to ldconfig search path"
        fi
        if [[ "$needs_ollama" -eq 1 ]]; then
            log "  ✓ Added /usr/local/lib/ollama to ldconfig search path"
        fi
    fi
}

# ============================================================================
# Link the CUDA ggml backend into Ollama's top-level lib directory
#
# Ollama 0.16+ uses ggml backend plugins.  CPU backends (libggml-cpu-*.so)
# live in /usr/local/lib/ollama/.  The CUDA backend (libggml-cuda.so) is
# shipped in /usr/local/lib/ollama/cuda_v12/ but is NOT automatically
# loaded because the ggml backend scanner only looks in the top-level dir.
#
# Fix: create a symlink so the backend scanner finds it.
# Also symlink the cuBLAS dependencies that libggml-cuda.so expects to
# dlopen at runtime.
# ============================================================================
link_ggml_cuda_backend() {
    local base="/usr/local/lib/ollama"
    local cuda_dir="$base/cuda_v12"

    if [[ ! -f "$cuda_dir/libggml-cuda.so" ]]; then
        # Try cuda_v13
        cuda_dir="$base/cuda_v13"
    fi

    if [[ ! -f "$cuda_dir/libggml-cuda.so" ]]; then
        warn "  No libggml-cuda.so found in Ollama cuda directories — skipping"
        return 0
    fi

    log "Linking ggml-cuda backend into Ollama's backend search path..."

    # Symlink the CUDA ggml backend to the top-level directory
    ln -sf "$cuda_dir/libggml-cuda.so" "$base/libggml-cuda.so" 2>/dev/null
    log "  ✓ $base/libggml-cuda.so → $cuda_dir/libggml-cuda.so"

    # Symlink cuBLAS and cuDNN dependencies too (libggml-cuda.so dlopen's these)
    for lib in libcublas.so libcublasLt.so libcudart.so; do
        local found
        found=$(find "$cuda_dir" -maxdepth 1 -name "${lib}*" -type f -o -name "${lib}*" -type l 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            local basename_f
            basename_f=$(basename "$found")
            ln -sf "$found" "$base/$basename_f" 2>/dev/null || true
        fi
    done

    log "  ✓ CUDA backend dependencies symlinked"
}

# ============================================================================
# Create environment profile for CUDA/NVIDIA paths
# ============================================================================
create_env_profile() {
    log "Setting up environment..."

    cat > /etc/profile.d/vgpu-cuda.sh <<'EOF'
# VGPU CUDA shim environment
# NOTE: CUDA_VISIBLE_DEVICES must NOT be set here.  Ollama uses it to select
# an alternate GPU-discovery path that bypasses our NVML shim, causing
# library=cpu even when the VGPU-STUB is present and functional.
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Ensure our shim libraries are found
if [ -d /usr/lib64 ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/lib64"
fi
EOF

    chmod 644 /etc/profile.d/vgpu-cuda.sh
    log "  ✓ Environment profile created: /etc/profile.d/vgpu-cuda.sh"
}

# ============================================================================
# Create a systemd service for device node creation at boot
# ============================================================================
create_boot_service() {
    log "Creating boot service for device nodes..."

    cat > /etc/systemd/system/vgpu-devices.service <<'EOF'
[Unit]
Description=Create NVIDIA device nodes and grant VGPU BAR access
After=systemd-udev-settle.service
Before=ollama.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c '\
    [ -e /dev/nvidia0 ] || mknod /dev/nvidia0 c 195 0; \
    [ -e /dev/nvidiactl ] || mknod /dev/nvidiactl c 195 255; \
    [ -e /dev/nvidia-uvm ] || mknod /dev/nvidia-uvm c 510 0; \
    [ -e /dev/nvidia-uvm-tools ] || mknod /dev/nvidia-uvm-tools c 510 1; \
    chmod 666 /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true; \
    for dev in /sys/bus/pci/devices/*/; do \
        v=$(cat "$dev/vendor" 2>/dev/null); \
        d=$(cat "$dev/device" 2>/dev/null); \
        if [ "$v" = "0x10de" ] && [ "$d" = "0x2331" ]; then \
            chmod 0666 "${dev}resource0" "${dev}resource1" 2>/dev/null || true; \
        fi; \
    done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload 2>/dev/null || true
    systemctl enable vgpu-devices.service 2>/dev/null || true
    log "  ✓ vgpu-devices.service created and enabled"
}

# ============================================================================
# Configure the Ollama systemd service to use the VGPU CUDA shim.
# Called after install_ollama (whether Ollama was just installed or was
# already present).  Always rewrites the drop-in and restarts the service.
# ============================================================================
configure_ollama_service() {
    if ! command -v ollama &>/dev/null; then
        return 0  # Ollama not installed — nothing to configure
    fi

    log "Configuring Ollama service for VGPU shim..."

    mkdir -p /etc/systemd/system/ollama.service.d

    # Discover the active sandboxing settings from the base Ollama unit.
    # We need to know which restrictions are enabled so we can override only
    # the ones that would block /sys/bus/pci/devices access.
    local protect_kt
    protect_kt=$(systemctl show ollama --property=ProtectKernelTunables 2>/dev/null \
                 | sed 's/ProtectKernelTunables=//')
    local private_dev
    private_dev=$(systemctl show ollama --property=PrivateDevices 2>/dev/null \
                 | sed 's/PrivateDevices=//')

    # Wrapper: install under /usr/local/lib/vgpu so the ollama service user can
    # read it (ProtectHome hides /home/... from the service).
    local wrapper_src="${SCRIPT_DIR}/ollama_wrapper.sh"
    local wrapper_install="/usr/local/lib/vgpu/ollama_wrapper.sh"

    if [ -f "$wrapper_src" ]; then
        mkdir -p /usr/local/lib/vgpu
        cp -f "$wrapper_src" "$wrapper_install"
        chmod 755 "$wrapper_install"
        cat > /etc/systemd/system/ollama.service.d/vgpu.conf <<EOF
[Service]
# ── Use wrapper script for comprehensive shim injection ─────────────────────
# The wrapper script ensures LD_PRELOAD is set and propagated to all
# subprocesses, including those spawned by Go's runtime.
# Installed copy: ${wrapper_install} (not \$SCRIPT_DIR — ollama user cannot read /home).
ExecStart=
ExecStart=/bin/bash ${wrapper_install}

# Same resource/sandbox overrides as the non-wrapper drop-in (mlock shmem, PCI BAR).
LimitMEMLOCK=infinity
ProtectKernelTunables=no
PrivateDevices=no
ReadWritePaths=/sys/bus/pci/devices/

# Shims: set here too so systemd exports to runner children (not only the bash wrapper).
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cudart.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"
# Ollama / ggml CUDA path (wrapper also exports LD_PRELOAD; these match phase3/ollama.service.d_vgpu.conf)
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"
Environment="OLLAMA_NUM_GPU=999"
Environment="OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama"
Environment="LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
Environment="OLLAMA_LOAD_TIMEOUT=4h"
Environment="OLLAMA_FLASH_ATTENTION=0"
Environment="CUDA_TRANSPORT_TIMEOUT_SEC=14400"
Environment="VGPU_SHMEM_MIN_SPAN_KB=32"
Environment="VGPU_ALLOW_MULTI_PROCESS_SHMEM=1"
Environment="VGPU_HTOD_BAR1=0"
Environment="VGPU_MODULE_BAR1=0"
Environment="VGPU_HTOD_BAR1_SHADOW=0"
Environment="GGML_CUDA_DISABLE_GRAPHS=1"
Environment="GGML_CUDA_DISABLE_GRAPH_RESERVE=1"
Environment="GGML_CUDA_DISABLE_BATCHED_CUBLAS=1"
Environment="OLLAMA_NO_MMAP=1"
Environment="NVIDIA_VISIBLE_DEVICES=all"
CapabilityBoundingSet=CAP_SYS_ADMIN CAP_IPC_LOCK
AmbientCapabilities=CAP_SYS_ADMIN CAP_IPC_LOCK
NoNewPrivileges=no
EOF
    else
        # Fallback to direct environment variables if wrapper not available
        cat > /etc/systemd/system/ollama.service.d/vgpu.conf <<EOF
[Service]
# ── Library preloading ─────────────────────────────────────────────────────
# Preload BOTH the CUDA Driver shim and the NVML shim.
#
# ELF SONAME of each library matches the name that callers dlopen():
#   libvgpu-cuda.so  → SONAME libcuda.so.1       (CUDA Driver API)
#   libvgpu-nvml.so  → SONAME libnvidia-ml.so.1  (NVML – GPU enumeration)
#
# With the correct SONAME, any dlopen("libcuda.so.1") call finds the
# already-mapped preloaded library instead of loading a separate copy.
# This intercepts Ollama's NVML-based GPU discovery as well as all CUDA
# Driver calls from the bundled libcudart / libggml-cuda runners.
#
# IMPORTANT: Do NOT set CUDA_VISIBLE_DEVICES.  Ollama 0.16+ logs
# "if GPUs are not correctly discovered, unset and try again" when that
# variable is set because it bypasses normal NVML discovery in favour of
# a fast validation path that requires a real NVIDIA kernel driver.
Environment="LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
Environment="NVIDIA_VISIBLE_DEVICES=all"
Environment="OLLAMA_DEBUG=1"

# ── Error capture ─────────────────────────────────────────────────────────────
# Redirect stderr to file for comprehensive error capture
StandardError=file:/tmp/ollama_stderr.log

# ── CUDA backend override ───────────────────────────────────────────────────
# Force the cuda_v12 backend regardless of NVML GPU detection.
# Our shim (libcuda.so.1) is installed directly into the cuda_v12 runner
# directory via install_into_ollama_runner_dirs(), so the bundled
# libcudart.so.12 and libggml-cuda.so will pick it up via RPATH \$ORIGIN.
# This bypasses NVML-based GPU detection entirely and ensures cuInit()
# reaches our shim even if LD_PRELOAD is not honoured for some reason.
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"

# Belt-and-suspenders: force all models to use GPU layers in the server
# environment.  When library=cpu is still reported by older detection paths
# this alone cannot help, but when library=cuda_v12 is selected it ensures
# the first model load request uses GPU layers without waiting for a client
# to supply num_gpu > 0.
Environment="OLLAMA_NUM_GPU=999"
Environment="VGPU_SHMEM_MIN_SPAN_KB=32"
Environment="VGPU_ALLOW_MULTI_PROCESS_SHMEM=1"
Environment="VGPU_HTOD_BAR1=0"
Environment="VGPU_MODULE_BAR1=0"
Environment="VGPU_HTOD_BAR1_SHADOW=0"
Environment="OLLAMA_NO_MMAP=1"

# ── Resource limits ────────────────────────────────────────────────────────
# Allow mlock() of the 256 MB shared-memory region used by the VGPU transport.
LimitMEMLOCK=infinity

# ── Sandbox overrides ──────────────────────────────────────────────────────
# The VGPU CUDA shim accesses PCI BAR registers through:
#   /sys/bus/pci/devices/<bdf>/resource0   (open O_RDWR + mmap MAP_SHARED)
#   /sys/bus/pci/devices/<bdf>/resource1   (BAR1 data path)
#
# Two common Ollama service hardening settings break this:
#
#   ProtectKernelTunables=yes
#     Mounts /proc and /sys read-only inside the service's mount namespace.
#     Even with chmod 0666, open(resource0, O_RDWR) returns EACCES,
#     so cuda_transport_init() fails before any CUDA call reaches the mediator.
#
#   PrivateDevices=yes
#     Creates a private /dev with only standard device nodes, hiding
#     /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm from the service.
#
# Both restrictions are overridden here so the VGPU shim can operate.
# All other Ollama hardening settings remain in effect.
ProtectKernelTunables=no
PrivateDevices=no

# ── PCI BAR access for runner subprocesses ─────────────────────────────────
# Even with ProtectKernelTunables=no the systemd service may mount /sys
# read-only in a private namespace.  ReadWritePaths= grants explicit
# read-write access to the PCI device tree so that the inference runner
# subprocess (spawned by ollama server) can open resource0 with O_RDWR
# when ensure_connected() fires on the first real CUDA call.
ReadWritePaths=/sys/bus/pci/devices/

EOF
    fi

    if [[ "$protect_kt" == "yes" ]]; then
        log "  ✓ Overrode ProtectKernelTunables=yes → no (needed for /sys PCI access)"
    fi
    if [[ "$private_dev" == "yes" ]]; then
        log "  ✓ Overrode PrivateDevices=yes → no (needed for /dev/nvidia*)"
    fi

    systemctl daemon-reload 2>/dev/null || true

    log "  ✓ Ollama service drop-in written (/etc/systemd/system/ollama.service.d/vgpu.conf)"

    # Restart so the running server picks up the new library and env vars.
    if systemctl is-active --quiet ollama 2>/dev/null; then
        log "  Restarting Ollama service to load VGPU CUDA shim..."
        systemctl restart ollama 2>/dev/null || warn "  Could not restart ollama service (may not be using systemd)"
        log "  ✓ Ollama restarted — shim is now active"
    else
        log "  Ollama service is not running; starting it now..."
        systemctl start ollama 2>/dev/null || warn "  Could not start ollama service"
        log "  ✓ Ollama started with VGPU shim"
    fi

    # Wait for the server to initialise and write its startup log, then verify
    # that the key environment variables were actually applied by systemd and
    # that GPU discovery did not fall back to library=cpu.
    sleep 4

    log ""
    log "  ── Verifying Ollama environment and GPU discovery ──────────────"

    local llm_lib gpu_status
    llm_lib=$(journalctl -u ollama --no-pager 2>/dev/null \
              | grep "OLLAMA_LLM_LIBRARY" | tail -1 \
              | sed 's/.*OLLAMA_LLM_LIBRARY:\([^ ]*\).*/\1/')
    gpu_status=$(journalctl -u ollama --no-pager 2>/dev/null \
                 | grep "inference compute" | tail -1 \
                 | sed 's/.*library=\([^ ]*\).*/\1/')

    if [[ -z "$llm_lib" || "$llm_lib" == "OLLAMA_LLM_LIBRARY" ]]; then
        warn "  ✗ OLLAMA_LLM_LIBRARY is not visible in the journal."
        warn "    Verify the drop-in: cat /etc/systemd/system/ollama.service.d/vgpu.conf"
    else
        log "  ✓ OLLAMA_LLM_LIBRARY=$llm_lib (from journal)"
    fi

    if [[ "$gpu_status" == "cpu" ]]; then
        warn "  ✗ Ollama still reports library=cpu — GPU discovery may still be failing."
        warn "    Check: journalctl -u ollama --no-pager | grep 'inference compute' | tail -3"
    elif [[ -n "$gpu_status" ]]; then
        log "  ✓ Ollama GPU discovery: library=$gpu_status"
    else
        log "  (No inference compute line yet — will appear on first model load)"
    fi
    log "  ────────────────────────────────────────────────────────────────"
}

# ============================================================================
# Install shim into Ollama's bundled CUDA runner directories.
#
# Ollama bundles GPU runner binaries under /usr/local/lib/ollama/ and their
# RPATH prioritises that directory tree over the system library path.
# LD_PRELOAD propagates to Ollama child processes but the runners may also
# use dlopen with absolute paths rooted in /usr/local/lib/ollama/.
# Placing our shim there (as libcuda.so.1 / libnvidia-ml.so.1) ensures the
# runners load our intercept library regardless of how they call dlopen.
# ============================================================================
install_into_ollama_runner_dirs() {
    local base="/usr/local/lib/ollama"
    if [[ ! -d "$base" ]]; then
        return 0  # Ollama not installed or uses a different layout
    fi

    log "Installing shim into Ollama runner directories..."

    # Build the list of candidate directories using three strategies:
    #
    # 1. Named sub-directories that Ollama versions are known to use.
    # 2. Any sub-directory (depth≤3) that already contains a CUDA-related
    #    shared library — these are the runner's own library bundles.
    # 3. Any sub-directory (depth≤3) that contains a known runner binary
    #    (ollama_llama_runner, llama-server, llama_runner, ollama, …).
    #    The runner binary's RPATH usually points to its own directory, so
    #    placing libcuda.so.1 there makes dlopen("libcuda.so.1") pick up
    #    our shim before anything else.

    declare -A seen_dirs
    local -a dirs=("$base")
    seen_dirs["$base"]=1

    _add_dir() {
        local d
        d="$(dirname "$1")"
        if [[ -d "$d" ]] && [[ -z "${seen_dirs[$d]+_}" ]]; then
            dirs+=("$d")
            seen_dirs["$d"]=1
        fi
    }

    # Strategy 1: well-known sub-directory names
    for d in "$base/runners" "$base/cuda_v11" "$base/cuda_v12" \
              "$base/cuda_v12_2" "$base/cuda_v12_3" "$base/cuda_v12_4" \
              "$base/rocm_v6" "$base/metal"; do
        if [[ -d "$d" ]] && [[ -z "${seen_dirs[$d]+_}" ]]; then
            dirs+=("$d"); seen_dirs["$d"]=1
        fi
    done

    # Strategy 2: dirs containing CUDA/NVML shared libraries
    while IFS= read -r -d '' f; do
        _add_dir "$f"
    done < <(find "$base" -maxdepth 3 -type f \( \
        -name 'libcuda*.so*' -o -name 'libnvidia-ml*.so*' \
        -o -name 'libcublas*.so*' -o -name 'libcudart*.so*' \) \
        -print0 2>/dev/null)

    # Strategy 3: dirs containing runner executables
    while IFS= read -r -d '' f; do
        _add_dir "$f"
    done < <(find "$base" -maxdepth 3 -type f \( \
        -name 'ollama_llama_runner' -o -name 'llama-server' \
        -o -name 'llama_runner'    -o -name 'ollama'        \
        -o -name 'ollama-cuda*'    -o -name '*_runner*'     \) \
        -print0 2>/dev/null)

    local installed=0
    for d in "${dirs[@]}"; do
        [[ -d "$d" ]] || continue
        cp -f "${INSTALL_LIB_DIR}/libvgpu-cuda.so" "$d/libcuda.so.1" \
            && cp -f "${INSTALL_LIB_DIR}/libvgpu-nvml.so" "$d/libnvidia-ml.so.1" \
            || { warn "  Failed to copy shim to $d"; continue; }
        ln -sf "libcuda.so.1"      "$d/libcuda.so"      2>/dev/null || true
        ln -sf "libnvidia-ml.so.1" "$d/libnvidia-ml.so" 2>/dev/null || true
        log "  ✓ Shim injected into $d"
        installed=$((installed + 1))
    done

    log "  ✓ CUDA shim installed in ${installed:-0} Ollama location(s)"
    return 0
}

# ============================================================================
# Install Ollama
# ============================================================================
install_ollama() {
    log "Installing Ollama..."

    if command -v ollama &>/dev/null; then
        local ver
        ver=$(ollama --version 2>/dev/null || echo "unknown")
        log "  Ollama already installed: ${ver}"
        return 0
    fi

    # Check for curl
    if ! command -v curl &>/dev/null; then
        error "curl not found. Install it first:"
        error "  Ubuntu/Debian: sudo apt install curl"
        error "  RHEL/CentOS:   sudo yum install curl"
        return 1
    fi

    log "  Downloading Ollama installer..."
    curl -fsSL https://ollama.com/install.sh | sh

    if command -v ollama &>/dev/null; then
        log "  ✓ Ollama installed successfully"
    else
        error "Ollama installation failed"
        return 1
    fi
}

# ============================================================================
# Shim self-test: compile a minimal cuInit probe and run it.
#
# The probe opens BAR0 via /sys, rings the CUDA doorbell (CUDA_CALL_INIT),
# and polls STATUS for DONE/ERROR.  If the mediator is running it will
# receive the call and print "[cuda-executor] CUDA_CALL_INIT vm=..." on the
# host side.  On the guest side we just confirm the device responds.
# ============================================================================
test_shim() {
    log ""
    log "═══════════════════════════════════════════════"
    log "  VGPU Shim Self-Test"
    log "═══════════════════════════════════════════════"

    if ! command -v gcc &>/dev/null; then
        warn "  gcc not found — skipping shim self-test"
        return 0
    fi

    # Write inline C probe source to a temp file
    local probe_src
    probe_src=$(mktemp /tmp/vgpu_probe_XXXXXX.c)
    local probe_bin
    probe_bin=$(mktemp /tmp/vgpu_probe_XXXXXX)

    cat > "$probe_src" <<'PROBE_EOF'
/* vgpu shim self-test: ring CUDA_CALL_INIT doorbell, check STATUS */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>
#include <time.h>

#define VGPU_VENDOR_ID  0x10DE
#define VGPU_DEVICE_ID  0x2331
#define VGPU_CLASS      0x030200
#define VGPU_CLASS_MASK 0xFFFF00
#define BAR0_SIZE       4096

/* MMIO register offsets */
#define REG_STATUS          0x004
#define REG_VM_ID           0x010
#define REG_PROTOCOL_VER    0x020
#define REG_CAPABILITIES    0x024
#define REG_CUDA_OP         0x080
#define REG_CUDA_SEQ        0x084
#define REG_CUDA_NUM_ARGS   0x088
#define REG_CUDA_DATA_LEN   0x08C
#define REG_CUDA_DOORBELL   0x0A8
#define REG_CUDA_RESULT_STATUS 0x0F0

#define STATUS_DONE  0x02
#define STATUS_ERROR 0x03
#define CUDA_CALL_INIT 0x0001

#define REG32(base, off) (*(volatile uint32_t *)((volatile char *)(base) + (off)))

static int find_bar0(char *out, size_t sz) {
    DIR *dir = opendir("/sys/bus/pci/devices");
    if (!dir) return -1;
    struct dirent *e;
    while ((e = readdir(dir)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char p[512];
        unsigned vendor = 0, device = 0, cls = 0;
        FILE *f;
        snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/vendor", e->d_name);
        f = fopen(p, "r"); if (!f) continue;
        fscanf(f, "%x", &vendor); fclose(f);
        snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/device", e->d_name);
        f = fopen(p, "r"); if (!f) continue;
        fscanf(f, "%x", &device); fclose(f);
        snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/class", e->d_name);
        f = fopen(p, "r"); if (!f) continue;
        fscanf(f, "%x", &cls); fclose(f);
        int class_ok = ((cls & VGPU_CLASS_MASK) == VGPU_CLASS);
        int exact  = class_ok && (vendor == VGPU_VENDOR_ID) && (device == VGPU_DEVICE_ID);
        int legacy = class_ok && (vendor == 0x1234 || vendor == 0x1AF4);
        if (!exact && !legacy) continue;
        fprintf(stderr, "[self-test] PCI: %s vendor=0x%04x device=0x%04x class=0x%06x\n",
                e->d_name, vendor, device, cls);
        snprintf(out, sz, "/sys/bus/pci/devices/%s/resource0", e->d_name);
        closedir(dir);
        return 0;
    }
    closedir(dir);
    return -1;
}

int main(void) {
    char res0[512];
    if (find_bar0(res0, sizeof(res0)) != 0) {
        fprintf(stderr, "[self-test] FAIL: VGPU-STUB PCI device not found\n");
        return 1;
    }
    fprintf(stderr, "[self-test] Found BAR0: %s\n", res0);

    int fd = open(res0, O_RDWR | O_SYNC);
    if (fd < 0) { perror("[self-test] open BAR0"); return 1; }

    volatile void *bar0 = mmap(NULL, BAR0_SIZE, PROT_READ|PROT_WRITE,
                                MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("[self-test] mmap BAR0"); close(fd); return 1; }

    uint32_t ver  = REG32(bar0, REG_PROTOCOL_VER);
    uint32_t caps = REG32(bar0, REG_CAPABILITIES);
    uint32_t vm_id= REG32(bar0, REG_VM_ID);
    fprintf(stderr, "[self-test] Protocol ver=0x%08x caps=0x%08x vm_id=%u\n",
            ver, caps, vm_id);

    /* Send CUDA_CALL_INIT via MMIO doorbell */
    REG32(bar0, REG_CUDA_OP)       = CUDA_CALL_INIT;
    REG32(bar0, REG_CUDA_SEQ)      = 0xDEAD;
    REG32(bar0, REG_CUDA_NUM_ARGS) = 0;
    REG32(bar0, REG_CUDA_DATA_LEN) = 0;
    REG32(bar0, REG_CUDA_DOORBELL) = 1;

    /* Poll STATUS (max 5 s).
     * We keep the last real register value so the error message is precise. */
    time_t start = time(NULL);
    uint32_t st = 0xFF;
    int timed_out = 0;
    while (1) {
        st = REG32(bar0, REG_STATUS);
        if (st == STATUS_DONE || st == STATUS_ERROR) break;
        if (time(NULL) - start >= 5) { timed_out = 1; break; }
        usleep(50000);
    }

    uint32_t res = REG32(bar0, REG_CUDA_RESULT_STATUS);
    munmap((void *)bar0, BAR0_SIZE);
    close(fd);

    if (!timed_out && st == STATUS_DONE) {
        fprintf(stderr, "[self-test] PASS: doorbell→mediator round-trip OK "
                "(cuda_result=%u)\n", res);
        return 0;
    } else if (!timed_out && st == STATUS_ERROR) {
        fprintf(stderr, "[self-test] WARN: doorbell reached vgpu-stub but mediator "
                "returned ERROR (cuda_result=%u) — mediator may not be running\n", res);
        return 2;
    } else if (timed_out && st == 0x00) {
        /* STATUS never left IDLE: the doorbell write was not dispatched to
         * vgpu_process_cuda_doorbell.  This is the MMIO-ordering bug —
         * the CUDA register block (0x080-0x0FF) was captured by the legacy
         * request-buffer handler (0x040-0x43F) before the CUDA handler
         * could see it.  The QEMU vgpu-stub binary must be rebuilt with
         * the MMIO-ordering fix and redeployed to the host. */
        fprintf(stderr, "[self-test] FAIL: STATUS stayed IDLE (0x00) for 5 s "
                "— doorbell not dispatched.  "
                "QEMU binary likely lacks the MMIO-ordering fix; rebuild QEMU.\n");
        return 1;
    } else if (timed_out && st == 0x01) {
        /* STATUS went BUSY but never DONE/ERROR: doorbell reached the stub
         * and was sent to the mediator socket, but the mediator did not
         * reply within 5 s.  Check that mediator_phase3 is running on the
         * host and that the socket at /tmp/vgpu-mediator.sock is present
         * inside QEMU's chroot. */
        fprintf(stderr, "[self-test] FAIL: STATUS stuck BUSY (0x01) for 5 s "
                "— stub sent the call but mediator did not reply.  "
                "Is mediator_phase3 running on the host?\n");
        return 1;
    } else {
        fprintf(stderr, "[self-test] FAIL: timeout after 5 s "
                "(last STATUS=0x%02x, expected DONE=0x02 or ERROR=0x03)\n", st);
        return 1;
    }
}
PROBE_EOF

    # Compile
    if ! gcc -O1 -o "$probe_bin" "$probe_src" 2>/tmp/vgpu_probe_cc_err; then
        warn "  gcc failed to compile self-test probe:"
        cat /tmp/vgpu_probe_cc_err | while IFS= read -r l; do warn "    $l"; done
        rm -f "$probe_src" "$probe_bin" /tmp/vgpu_probe_cc_err
        return 0
    fi
    rm -f /tmp/vgpu_probe_cc_err

    # Run (capture stderr which holds all our messages)
    local rc=0
    "$probe_bin" 2>&1 | while IFS= read -r l; do log "  $l"; done
    rc=${PIPESTATUS[0]}
    rm -f "$probe_src" "$probe_bin"

    case "$rc" in
        0) log "  ✓ Shim self-test PASSED — end-to-end doorbell→mediator confirmed" ;;
        2) warn "  ○ Shim self-test: device responds but mediator returned an error" ;;
        *) warn "  ○ Shim self-test: no mediator response (is the mediator running on the host?)" ;;
    esac

    # ── Ollama-user permission test ──────────────────────────────────────────
    # The Ollama service runs as a non-root user.  Confirm that user can open
    # resource0 with O_RDWR and mmap it (both needed by cuda_transport_init).
    if id ollama &>/dev/null && command -v gcc &>/dev/null; then
        log ""
        log "  Running self-test as the 'ollama' user..."

        # Re-compile the probe (it may have been deleted above; rebuild it here)
        local probe_u_src probe_u_bin
        probe_u_src=$(mktemp /tmp/vgpu_probe_XXXXXX.c)
        probe_u_bin=$(mktemp /tmp/vgpu_probe_XXXXXX)
        # Use the same probe source (re-written above)
        cat > "$probe_u_src" <<'UPROBE_EOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>
#include <time.h>
#define VGPU_CLASS      0x030200
#define VGPU_CLASS_MASK 0xFFFF00
#define VGPU_VENDOR_ID  0x10DE
#define VGPU_DEVICE_ID  0x2331
#define BAR0_SIZE       4096
#define REG_STATUS          0x004
#define REG_CUDA_OP         0x080
#define REG_CUDA_SEQ        0x084
#define REG_CUDA_NUM_ARGS   0x088
#define REG_CUDA_DATA_LEN   0x08C
#define REG_CUDA_DOORBELL   0x0A8
#define REG_CUDA_RESULT_STATUS 0x0F0
#define STATUS_DONE  0x02
#define STATUS_ERROR 0x03
#define CUDA_CALL_INIT 0x0001
#define REG32(base,off) (*(volatile uint32_t*)((volatile char*)(base)+(off)))
static int find_bar0(char *out, size_t sz) {
    DIR *d = opendir("/sys/bus/pci/devices"); if (!d) return -1;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0]=='.') continue;
        char p[512]; unsigned v=0,dev=0,cls=0; FILE *f;
        snprintf(p,sizeof(p),"/sys/bus/pci/devices/%s/vendor",e->d_name);
        f=fopen(p,"r"); if(!f) continue; fscanf(f,"%x",&v); fclose(f);
        snprintf(p,sizeof(p),"/sys/bus/pci/devices/%s/device",e->d_name);
        f=fopen(p,"r"); if(!f) continue; fscanf(f,"%x",&dev); fclose(f);
        snprintf(p,sizeof(p),"/sys/bus/pci/devices/%s/class",e->d_name);
        f=fopen(p,"r"); if(!f) continue; fscanf(f,"%x",&cls); fclose(f);
        int cok=((cls&VGPU_CLASS_MASK)==VGPU_CLASS);
        if(!(cok&&((v==VGPU_VENDOR_ID&&dev==VGPU_DEVICE_ID)||(v==0x1234||v==0x1AF4)))) continue;
        snprintf(out,sz,"/sys/bus/pci/devices/%s/resource0",e->d_name);
        closedir(d); return 0;
    }
    closedir(d); return -1;
}
int main(void) {
    char res0[512];
    if (find_bar0(res0,sizeof(res0))!=0) {
        fprintf(stderr,"[ollama-user-test] FAIL: VGPU device not found\n"); return 1;
    }
    int fd=open(res0,O_RDWR|O_SYNC);
    if (fd<0) { fprintf(stderr,"[ollama-user-test] FAIL: open %s: %m\n",res0); return 1; }
    volatile void *bar0=mmap(NULL,BAR0_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if (bar0==MAP_FAILED) { fprintf(stderr,"[ollama-user-test] FAIL: mmap: %m\n"); close(fd); return 1; }
    REG32(bar0,REG_CUDA_OP)=CUDA_CALL_INIT;
    REG32(bar0,REG_CUDA_SEQ)=0xBEEF;
    REG32(bar0,REG_CUDA_NUM_ARGS)=0;
    REG32(bar0,REG_CUDA_DATA_LEN)=0;
    REG32(bar0,REG_CUDA_DOORBELL)=1;
    time_t s=time(NULL); uint32_t st=0xFF; int to=0;
    while(1){ st=REG32(bar0,REG_STATUS);
              if(st==STATUS_DONE||st==STATUS_ERROR) break;
              if(time(NULL)-s>=5){to=1;break;} usleep(50000); }
    munmap((void*)bar0,BAR0_SIZE); close(fd);
    if(!to&&st==STATUS_DONE){ fprintf(stderr,"[ollama-user-test] PASS\n"); return 0; }
    fprintf(stderr,"[ollama-user-test] FAIL: status=0x%02x timed_out=%d\n",st,to); return 1;
}
UPROBE_EOF
        if gcc -O1 -o "$probe_u_bin" "$probe_u_src" 2>/dev/null; then
            chmod 0755 "$probe_u_bin"
            local u_rc=0
            sudo -u ollama "$probe_u_bin" 2>&1 | while IFS= read -r l; do log "  $l"; done
            u_rc=${PIPESTATUS[0]}
            if [[ "$u_rc" -eq 0 ]]; then
                log "  ✓ ollama-user self-test PASSED — BAR0 accessible as 'ollama' user"
            else
                warn "  ✗ ollama-user self-test FAILED (rc=$u_rc)"
                warn "    This means the 'ollama' service cannot access BAR0."
                warn "    Check: ls -la /sys/bus/pci/devices/*/resource0"
                # Show actual permissions for diagnosis
                for r0 in /sys/bus/pci/devices/*/resource0; do
                    local perm; perm=$(stat -c '%a' "$r0" 2>/dev/null)
                    warn "    $r0 → $perm"
                done
            fi
        fi
        rm -f "$probe_u_src" "$probe_u_bin"
    fi

    # ── Ollama process map check ─────────────────────────────────────────────
    # After Ollama restarts, check whether our shim appears in its memory map.
    # This confirms LD_PRELOAD is being honoured by the systemd service.
    local ollama_pid
    ollama_pid=$(systemctl show ollama --property=MainPID --value 2>/dev/null)
    if [[ -n "$ollama_pid" && "$ollama_pid" != "0" ]]; then
        log ""
        log "  Checking if shim is mapped in Ollama process (pid=$ollama_pid)..."
        local maps_out
        maps_out=$(grep -i "vgpu\|libcuda\|libnvidia" /proc/"$ollama_pid"/maps 2>/dev/null)
        if [[ -n "$maps_out" ]]; then
            log "  ✓ VGPU shim mapped in Ollama process:"
            echo "$maps_out" | while IFS= read -r l; do log "    $l"; done
        else
            warn "  ✗ VGPU shim NOT found in Ollama process maps!"
            warn "    LD_PRELOAD is not being applied to the service."
            warn "    Dropping environment from the drop-in:"
            grep -A2 'LD_PRELOAD\|OLLAMA_LLM' \
                /etc/systemd/system/ollama.service.d/vgpu.conf 2>/dev/null \
                | while IFS= read -r l; do warn "    $l"; done
        fi
    fi

    return 0
}

# ============================================================================
# Verify installation
# ============================================================================
verify_installation() {
    log ""
    log "═══════════════════════════════════════════════"
    log "  VGPU Shim Installation Verification"
    log "═══════════════════════════════════════════════"

    local ok=1

    # Check shim libraries
    if [[ -f "${INSTALL_LIB_DIR}/libvgpu-cuda.so" ]]; then
        log "  ✓ libvgpu-cuda.so installed"
    else
        error "  ✗ libvgpu-cuda.so NOT found"
        ok=0
    fi

    if [[ -f "${INSTALL_LIB_DIR}/libvgpu-nvml.so" ]]; then
        log "  ✓ libvgpu-nvml.so installed"
    else
        error "  ✗ libvgpu-nvml.so NOT found"
        ok=0
    fi

    # Check symlinks
    for lib in libcuda.so.1 libnvidia-ml.so.1; do
        if [[ -L "${INSTALL_LIB_DIR}/${lib}" ]]; then
            local target
            target=$(readlink "${INSTALL_LIB_DIR}/${lib}")
            log "  ✓ ${lib} → ${target}"
        else
            error "  ✗ ${lib} symlink missing"
            ok=0
        fi
    done

    # Check device nodes
    for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
        if [[ -e "$dev" ]]; then
            log "  ✓ ${dev} exists"
        else
            warn "  ○ ${dev} not found (will be created at next boot)"
        fi
    done

    # Check PCI device
    if check_vgpu_device 2>/dev/null; then
        log "  ✓ VGPU-STUB PCI device visible"
    else
        error "  ✗ VGPU-STUB PCI device not found"
        ok=0
    fi

    # Check Ollama
    if command -v ollama &>/dev/null; then
        log "  ✓ Ollama installed"
    else
        info "  ○ Ollama not installed (use --with-ollama to install)"
    fi

    log ""
    if [[ "$ok" -eq 1 ]]; then
        log "═══════════════════════════════════════════════"
        log "  Installation looks good!"
        log ""
        log "  Next steps:"
        log "    1. Source the environment:"
        log "       source /etc/profile.d/vgpu-cuda.sh"
        log ""
        log "    2. Verify GPU detection:"
        log "       python3 -c 'import ctypes; cuda=ctypes.CDLL(\"libcuda.so.1\"); print(\"CUDA loaded\")'"
        log ""
        if command -v ollama &>/dev/null; then
            log "    3. Test with Ollama:"
            log "       ollama run llama3.2:1b \"Hello\""
        fi
        log "═══════════════════════════════════════════════"
    else
        error "═══════════════════════════════════════════════"
        error "  Some checks failed — see above for details."
        error "═══════════════════════════════════════════════"
    fi

    return 0
}

# ============================================================================
# Uninstall
# ============================================================================
uninstall() {
    log "Uninstalling VGPU shim libraries..."

    # Remove shim libraries
    for f in libvgpu-cuda.so libvgpu-nvml.so; do
        if [[ -f "${INSTALL_LIB_DIR}/${f}" ]]; then
            rm -f "${INSTALL_LIB_DIR}/${f}"
            log "  Removed ${f}"
        fi
    done

    # Remove symlinks
    for link in libcuda.so.1 libcuda.so libnvidia-ml.so.1 libnvidia-ml.so; do
        local target="${INSTALL_LIB_DIR}/${link}"
        if [[ -L "$target" ]]; then
            local dest
            dest=$(readlink "$target")
            if [[ "$dest" == *"libvgpu"* ]]; then
                rm -f "$target"
                log "  Removed symlink ${link}"

                # Restore backup if it exists
                if [[ -f "${target}.real" ]]; then
                    mv "${target}.real" "$target"
                    log "  Restored ${link} from backup"
                fi
            fi
        fi
    done

    # Remove device nodes (optional — they'll be recreated by real NVIDIA driver)
    # Don't remove them as they might belong to a real driver now

    # Remove NVIDIA kernel module blacklist
    rm -f /etc/modprobe.d/blacklist-nvidia-real.conf
    log "  Removed NVIDIA kernel module blacklist"

    # Remove environment profile
    rm -f /etc/profile.d/vgpu-cuda.sh
    log "  Removed environment profile"

    # Remove udev rules
    rm -f /etc/udev/rules.d/99-vgpu-nvidia.rules
    log "  Removed udev rules"

    # Remove boot service
    if [[ -f /etc/systemd/system/vgpu-devices.service ]]; then
        systemctl disable vgpu-devices.service 2>/dev/null || true
        rm -f /etc/systemd/system/vgpu-devices.service
        systemctl daemon-reload 2>/dev/null || true
        log "  Removed boot service"
    fi

    # Remove Ollama overrides (but don't uninstall Ollama itself)
    rm -f /etc/systemd/system/ollama.service.d/vgpu.conf
    rmdir /etc/systemd/system/ollama.service.d 2>/dev/null || true

    ldconfig 2>/dev/null || true

    log ""
    log "VGPU shim uninstalled. Run 'ldconfig' to refresh library cache."
}

# ============================================================================
# Main
# ============================================================================
main() {
    local with_ollama=0
    local do_uninstall=0
    local do_check=0

    for arg in "$@"; do
        case "$arg" in
            --with-ollama)  with_ollama=1 ;;
            --uninstall)    do_uninstall=1 ;;
            --check)        do_check=1 ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --with-ollama    Also install Ollama for LLM testing"
                echo "  --uninstall      Remove shim libraries and symlinks"
                echo "  --check          Verify installation only"
                echo "  --help           Show this help"
                echo ""
                echo "This script installs VGPU CUDA/NVML shim libraries into"
                echo "the guest VM so that GPU-accelerated applications like"
                echo "Ollama can run using the host GPU via the VGPU-STUB device."
                exit 0
                ;;
            *)
                error "Unknown option: $arg"
                error "Use --help for usage information."
                exit 1
                ;;
        esac
    done

    # Check-only mode
    if [[ "$do_check" -eq 1 ]]; then
        verify_installation
        exit $?
    fi

    # Uninstall mode
    if [[ "$do_uninstall" -eq 1 ]]; then
        uninstall
        exit $?
    fi

    # Must be root for install
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (or with sudo)."
        exit 1
    fi

    log ""
    log "═══════════════════════════════════════════════"
    log "  VGPU Guest VM Setup"
    log "═══════════════════════════════════════════════"
    log ""

    # Step 1: Check device
    check_vgpu_device || exit 1

    # Step 2: Build shims
    build_shims || exit 1

    # Step 3: Install shims
    install_shims || exit 1

    # Step 4: Create device nodes
    create_device_nodes || exit 1

    # Step 5: udev rules
    create_udev_rules || exit 1

    # Step 6: Environment
    create_env_profile || exit 1

    # Step 6b: Ensure /usr/lib64 is in ldconfig (Ubuntu/Debian fix)
    ensure_ldconfig_path

    # Step 7: Boot service
    create_boot_service || exit 1

    # Step 8: (Optional) Install Ollama
    if [[ "$with_ollama" -eq 1 ]]; then
        install_ollama || warn "Ollama installation failed (non-fatal)"
    fi

    # Step 8b: Inject shim into Ollama's own runner directories (bypasses RPATH)
    install_into_ollama_runner_dirs

    # Step 8c: Link ggml-cuda backend to the top-level Ollama lib directory
    #          so the ggml backend scanner finds and loads it.
    link_ggml_cuda_backend

    # Step 8d: Configure Ollama service + restart (always, if Ollama is present)
    configure_ollama_service

    # Step 9: Shim self-test (confirms doorbell → mediator round-trip)
    test_shim

    # Step 10: Verify
    verify_installation
}

main "$@"
