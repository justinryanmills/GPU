# ============================================================================
# Phase 3 — Self-Contained Build System
#
# All source files live under phase3/ — no cross-directory dependencies.
#
# Builds:
#   mediator_phase3    — host-side GPU mediation daemon with WFQ, rate limiter,
#                        watchdog, metrics, NVML, CUDA executor, and admin socket
#   vgpu-admin         — admin CLI (C binary, uses libvgpu_config + sqlite)
#   vm_client_enhanced — guest-side MMIO client with Phase 3 retry logic
#
# Guest shim libraries (built separately, installed inside VM):
#   libvgpu-cuda.so    — CUDA Driver API shim (replaces libcuda.so)
#   libvgpu-nvml.so    — NVML shim (replaces libnvidia-ml.so)
#
# Usage:
#   make              — build all host binaries
#   make guest        — build guest shim libraries (libvgpu-cuda.so, libvgpu-nvml.so)
#   make client       — build vm_client_enhanced only (for guest deployment)
#   make install      — install binaries and DB schema
#   make clean        — remove build artefacts
# ============================================================================

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDFLAGS  = -lpthread -lsqlite3 -ldl -lm

# CUDA paths (host only; guest builds skip this)
CUDA_PATH  ?= /usr/local/cuda
CUDA_LIB   = $(CUDA_PATH)/lib64
CUDA_INC   = $(CUDA_PATH)/include
NVCC       ?= nvcc
NVCC_FLAGS = -O2 -arch=sm_90 -x cu      # sm_90 = H100

# Local directories (everything is inside phase3/)
INC_DIR    = include
SRC_DIR    = src
SHIM_DIR   = guest-shim

# Include path — single local directory + CUDA
INCLUDES = -I$(INC_DIR) -I$(CUDA_INC)

# ---- QEMU build variables (RPM-based build system) ------------------------
# Based on build_enhanced_qemu.sh - uses RPM build system for XCP-ng
RPM_BUILD      ?= $(HOME)/vgpu-build/rpmbuild
RPM_SOURCES    = $(RPM_BUILD)/SOURCES
RPM_SPECS      = $(RPM_BUILD)/SPECS
QEMU_SPEC      = $(RPM_SPECS)/qemu.spec
QEMU_STUB_SRC  = $(SRC_DIR)/vgpu-stub-enhanced.c
QEMU_STUB_RPM  = $(RPM_SOURCES)/vgpu-stub.c
QEMU_PROTO_RPM = $(RPM_SOURCES)/vgpu_protocol.h
QEMU_CUDA_PROTO_RPM = $(RPM_SOURCES)/cuda_protocol.h

# ---- Phase 3 module sources ------------------------------------------------
P3_SRCS = $(SRC_DIR)/scheduler_wfq.c \
          $(SRC_DIR)/rate_limiter.c   \
          $(SRC_DIR)/metrics.c        \
          $(SRC_DIR)/nvml_monitor.c   \
          $(SRC_DIR)/watchdog.c

P3_OBJS = $(P3_SRCS:.c=.o)

# ---- CUDA executor (host-side CUDA API replay engine) ----------------------
CUDA_EXEC_SRC = $(SRC_DIR)/cuda_executor.c
CUDA_EXEC_OBJ = $(SRC_DIR)/cuda_executor.o

# ---- Mediator (host) -------------------------------------------------------
MEDIATOR_SRC  = $(SRC_DIR)/mediator_phase3.c
MEDIATOR_OBJ  = $(MEDIATOR_SRC:.c=.o)
MEDIATOR_BIN  = mediator_phase3

# CUDA helper (compiled with nvcc)
CUDA_SRC = $(SRC_DIR)/cuda_vector_add.c
CUDA_OBJ = $(SRC_DIR)/cuda_vector_add.o

# vgpu_config library
CONFIG_SRC = $(SRC_DIR)/vgpu_config.c
CONFIG_OBJ = $(SRC_DIR)/vgpu_config.o

# ---- vgpu-admin CLI --------------------------------------------------------
ADMIN_SRC = $(SRC_DIR)/vgpu-admin.c
ADMIN_OBJ = $(SRC_DIR)/vgpu-admin.o
ADMIN_BIN = vgpu-admin

# ---- VM client (guest) -----------------------------------------------------
CLIENT_SRC = $(SRC_DIR)/vm_client_enhanced.c
CLIENT_BIN = vm_client_enhanced

# ---- Guest shim libraries --------------------------------------------------
SHIM_CUDA_SRC      = $(SHIM_DIR)/libvgpu_cuda.c
SHIM_CUDART_SRC    = $(SHIM_DIR)/libvgpu_cudart.c
SHIM_NVML_SRC      = $(SHIM_DIR)/libvgpu_nvml.c
SHIM_CUBLAS_SRC    = $(SHIM_DIR)/libvgpu_cublas.c
SHIM_CUBLASLT_SRC  = $(SHIM_DIR)/libvgpu_cublasLt.c
SHIM_TRANSPORT_SRC = $(SHIM_DIR)/cuda_transport.c
SHIM_CUDA_LIB      = $(SHIM_DIR)/libvgpu-cuda.so.1
SHIM_CUDART_LIB    = $(SHIM_DIR)/libvgpu-cudart.so
SHIM_NVML_LIB      = $(SHIM_DIR)/libvgpu-nvml.so
SHIM_CUBLAS_LIB    = $(SHIM_DIR)/libvgpu-cublas.so.12
SHIM_CUBLASLT_LIB  = $(SHIM_DIR)/libvgpu-cublasLt.so.12
SHIM_EXEC_INJECT_SRC = $(SHIM_DIR)/libvgpu_exec_inject.c
SHIM_EXEC_INJECT_LIB = $(SHIM_DIR)/libvgpu-exec-inject.so
SHIM_INCLUDES      = -I$(INC_DIR) -I$(SHIM_DIR)
OLLAMA_WRAPPER_BIN = ollama_vgpu_wrapper

# ============================================================================
# Targets
# ============================================================================
.PHONY: all host guest client test-cublas-vm clean install migrate help ollama-wrapper

all: host

host: $(MEDIATOR_BIN) $(ADMIN_BIN)
	@echo ""
	@echo "Host binaries built:"
	@echo "  $(MEDIATOR_BIN)  — mediator daemon"
	@echo "  $(ADMIN_BIN)        — admin CLI"
	@echo ""

guest: $(SHIM_EXEC_INJECT_LIB) $(SHIM_CUDA_LIB) $(SHIM_CUDART_LIB) $(SHIM_NVML_LIB) $(SHIM_CUBLAS_LIB) $(SHIM_CUBLASLT_LIB)
	@echo ""
	@echo "Guest shim libraries built:"
	@echo "  $(SHIM_CUDA_LIB)    — CUDA Driver API shim (replaces libcuda.so.1)"
	@echo "  $(SHIM_CUDART_LIB)  — CUDA Runtime API shim (replaces libcudart.so.12)"
	@echo "  $(SHIM_NVML_LIB)    — NVML shim (replaces libnvidia-ml.so)"
	@echo "  $(SHIM_CUBLAS_LIB)  — CUBLAS shim (replaces libcublas.so.12)"
	@echo "  $(SHIM_CUBLASLT_LIB) — CUBLAS LT shim (replaces libcublasLt.so.12)"
	@echo ""
	@echo "Deploy to guest VM:"
	@echo "  scp $(SHIM_DIR)/libvgpu-cuda.so.1 $(SHIM_DIR)/libvgpu-cudart.so \\"
	@echo "      $(SHIM_DIR)/libvgpu-cublas.so.12 $(SHIM_DIR)/libvgpu-cublasLt.so.12 \\"
	@echo "      $(SHIM_DIR)/libvgpu-nvml.so $(SHIM_DIR)/install.sh guest-vm:/tmp/vgpu/"
	@echo "  ssh guest-vm 'sudo /tmp/vgpu/install.sh'"
	@echo ""

client: $(CLIENT_BIN)
	@echo "Guest binary built: $(CLIENT_BIN)"

# ---- VM-local CUBLAS test (no host transfer) ------------
TEST_CUBLAS_VM = $(SHIM_DIR)/test_cublas_vm
test-cublas-vm: $(TEST_CUBLAS_VM)
	@echo "Run on VM: LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:\$$LD_LIBRARY_PATH ./guest-shim/test_cublas_vm"

$(TEST_CUBLAS_VM): $(SHIM_DIR)/test_cublas_vm.c
	$(CC) $(CFLAGS) -o $@ $< -ldl
	@echo "[TEST] $@ — VM-local, no host"

# ---- Ollama vGPU wrapper (C binary, no shell) --------------------------------
$(OLLAMA_WRAPPER_BIN): ollama_vgpu_wrapper.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Built $@ — install as /usr/local/bin/ollama on guest VM (back up ollama.real first)."

OLLAMA_REAL_WRAPPER_BIN = ollama_real_wrapper
$(OLLAMA_REAL_WRAPPER_BIN): ollama_real_wrapper.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Built $@ — install as /usr/local/bin/ollama.real on guest VM (real binary must be at ollama.real.bin)."

ollama-wrapper: $(OLLAMA_WRAPPER_BIN)
ollama-real-wrapper: $(OLLAMA_REAL_WRAPPER_BIN)

# ---- Mediator ---------------------------------------------------------------
$(MEDIATOR_BIN): $(MEDIATOR_OBJ) $(P3_OBJS) $(CONFIG_OBJ) $(CUDA_OBJ) $(CUDA_EXEC_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -L$(CUDA_LIB) -lcudart -lcuda -lcublas -lcublasLt -lnvidia-ml
	@echo "[LINK] $@"

# ---- vgpu-admin CLI --------------------------------------------------------
$(ADMIN_BIN): $(ADMIN_OBJ) $(CONFIG_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lsqlite3
	@echo "[LINK] $@"

# ---- VM client (no CUDA, no sqlite — runs inside guest) --------------------
$(CLIENT_BIN): $(CLIENT_SRC) $(INC_DIR)/vgpu_protocol.h
	$(CC) $(CFLAGS) -I$(INC_DIR) -o $@ $(CLIENT_SRC)
	@echo "[LINK] $@"

# ---- Guest shim libraries (no CUDA SDK needed — pure userspace) ------------
$(SHIM_CUDA_LIB): $(SHIM_CUDA_SRC) $(SHIM_TRANSPORT_SRC) $(SHIM_DIR)/cuda_transport.h \
                  $(SHIM_DIR)/gpu_properties.h $(INC_DIR)/cuda_protocol.h
	$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
		-o $@ $(SHIM_CUDA_SRC) $(SHIM_TRANSPORT_SRC) -ldl -lpthread
	@echo "[SHIM] $@"

$(SHIM_CUDART_LIB): $(SHIM_CUDART_SRC) $(SHIM_DIR)/gpu_properties.h $(INC_DIR)/cuda_protocol.h
	$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
		-o $@ $(SHIM_CUDART_SRC) -ldl
	@echo "[SHIM] $@"

$(SHIM_EXEC_INJECT_LIB): $(SHIM_EXEC_INJECT_SRC)
	$(CC) -shared -fPIC $(CFLAGS) -I$(SHIM_DIR) -o $@ $(SHIM_EXEC_INJECT_SRC) -ldl
	@echo "[SHIM] $@"

$(SHIM_NVML_LIB): $(SHIM_NVML_SRC) $(SHIM_TRANSPORT_SRC) $(SHIM_DIR)/cuda_transport.h \
                  $(SHIM_DIR)/gpu_properties.h $(INC_DIR)/cuda_protocol.h
	$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
		-o $@ $(SHIM_NVML_SRC) $(SHIM_TRANSPORT_SRC) -lpthread
	@echo "[SHIM] $@"

$(SHIM_CUBLAS_LIB): $(SHIM_CUBLAS_SRC) $(SHIM_TRANSPORT_SRC) $(SHIM_DIR)/cuda_transport.h $(SHIM_DIR)/gpu_properties.h $(INC_DIR)/cuda_protocol.h
	@if [ -f "$(SHIM_DIR)/cublas_version.lds" ]; then \
		$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
			-Wl,--version-script=$(SHIM_DIR)/cublas_version.lds \
			-o $@ $(SHIM_CUBLAS_SRC) $(SHIM_TRANSPORT_SRC) -ldl -lpthread; \
		echo "[SHIM] $@ (with version script)"; \
	else \
		$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
			-o $@ $(SHIM_CUBLAS_SRC) $(SHIM_TRANSPORT_SRC) -ldl -lpthread; \
		echo "[SHIM] $@ (without version script - may need symbol versioning)"; \
	fi

$(SHIM_CUBLASLT_LIB): $(SHIM_CUBLASLT_SRC) $(SHIM_DIR)/gpu_properties.h
	$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
		-o $@ $(SHIM_CUBLASLT_SRC) -ldl
	@echo "[SHIM] $@"

# ---- Compile rules ----------------------------------------------------------

# All .c files under src/ use the same include path
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[CC]   $<"

# CUDA executor (uses CUDA Driver API — needs CUDA SDK)
$(CUDA_EXEC_OBJ): $(CUDA_EXEC_SRC) $(INC_DIR)/cuda_executor.h $(INC_DIR)/cuda_protocol.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[CC]   $< (CUDA executor)"

# CUDA helper (nvcc — extension is .c but contains CUDA kernels)
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) -c $< -o $@
	@echo "[NVCC] $<"

# ---- Install ----------------------------------------------------------------
PREFIX     ?= /usr/local
ETC_PREFIX ?= /etc/vgpu
DB_PATH    = $(ETC_PREFIX)/vgpu_config.db

install: host
	@echo "Installing Phase 3 binaries and schema..."
	install -d $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(ETC_PREFIX)
	install -m 755 $(MEDIATOR_BIN) $(DESTDIR)$(PREFIX)/bin/
	install -m 755 $(ADMIN_BIN)    $(DESTDIR)$(PREFIX)/bin/
	install -m 644 schema/migrate_phase3.sql $(DESTDIR)$(ETC_PREFIX)/
	install -m 644 schema/init_db.sql        $(DESTDIR)$(ETC_PREFIX)/
	install -d $(DESTDIR)/var/vgpu
	@echo ""
	@echo "Installed to $(DESTDIR)$(PREFIX)/bin/"
	@echo "Schema files in $(DESTDIR)$(ETC_PREFIX)/"
	@echo ""
	@echo "To initialise or migrate the database:"
	@echo "  sqlite3 $(DB_PATH) < $(ETC_PREFIX)/init_db.sql"
	@echo "  sqlite3 $(DB_PATH) < $(ETC_PREFIX)/migrate_phase3.sql"
	@echo ""
	@echo "To deploy guest shim libraries into a VM:"
	@echo "  make guest"
	@echo "  scp guest-shim/libvgpu-cuda.so.1 guest-shim/libvgpu-cudart.so \\"
	@echo "      guest-shim/libvgpu-cublas.so.12 guest-shim/libvgpu-cublasLt.so.12 \\"
	@echo "      guest-shim/libvgpu-nvml.so guest-shim/install.sh <guest-vm>:/tmp/vgpu/"
	@echo "  ssh <guest-vm> 'sudo /tmp/vgpu/install.sh --with-ollama'"

# ---- Database migration shortcut -------------------------------------------
migrate:
	@echo "Running Phase 3 schema migration..."
	sqlite3 $(DB_PATH) < schema/migrate_phase3.sql
	@echo "Migration complete."

# ---- Clean ------------------------------------------------------------------
clean:
	rm -f $(P3_OBJS) $(MEDIATOR_OBJ) $(CONFIG_OBJ) $(ADMIN_OBJ) $(CUDA_OBJ) $(CUDA_EXEC_OBJ)
	rm -f $(MEDIATOR_BIN) $(ADMIN_BIN) $(CLIENT_BIN)
	rm -f $(SHIM_EXEC_INJECT_LIB) $(SHIM_CUDA_LIB) $(SHIM_CUDART_LIB) $(SHIM_NVML_LIB) $(SHIM_CUBLAS_LIB) $(SHIM_CUBLASLT_LIB)
	@echo "Clean."

# ---- Help -------------------------------------------------------------------
help:
	@echo "Phase 3 Build Targets:"
	@echo "  make          — build mediator_phase3 + vgpu-admin (host)"
	@echo "  make guest    — build guest shim libraries (libvgpu-cuda.so, libvgpu-nvml.so)"
	@echo "  make client   — build vm_client_enhanced (guest)"
	@echo "  make install  — install binaries + schema to system paths"
	@echo "  make migrate  — run Phase 3 DB schema migration"
	@echo "  make clean    — remove build artefacts"
	@echo "  make qemu     — build QEMU with vgpu-cuda device"
	@echo ""
	@echo "Guest Shim:"
	@echo "  The 'guest' target builds CUDA/NVML shim libraries that are installed"
	@echo "  inside the guest VM. These replace libcuda.so and libnvidia-ml.so to"
	@echo "  intercept CUDA/NVML calls and forward them to the host via the VGPU-STUB"
	@echo "  PCI device. Use guest-shim/install.sh to deploy inside the VM."
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH=/usr/local/cuda   — CUDA toolkit location"
	@echo "  PREFIX=/usr/local            — binary install prefix"
	@echo "  ETC_PREFIX=/etc/vgpu         — config/schema install prefix"
	@echo "  RPM_BUILD=$$HOME/vgpu-build/rpmbuild — RPM build directory (default)"

# ============================================================================
# QEMU Build Integration
# ============================================================================
.PHONY: qemu qemu-prepare qemu-integrate qemu-build qemu-install qemu-check qemu-clean

# ---- Safety checks ---------------------------------------------------------
qemu-check:
	@if [ ! -d "$(RPM_BUILD)" ]; then \
		echo "ERROR: RPM build directory does not exist: $(RPM_BUILD)"; \
		echo "       Please set up the build environment first."; \
		echo "       Create: mkdir -p $(RPM_BUILD)/{SOURCES,SPECS,BUILD,RPMS,SRPMS}"; \
		exit 1; \
	fi
	@if [ ! -f "$(QEMU_SPEC)" ]; then \
		echo "ERROR: QEMU spec file not found: $(QEMU_SPEC)"; \
		echo "       Please copy qemu.spec to $(RPM_SPECS)/"; \
		exit 1; \
	fi
	@if [ ! -f "$(QEMU_STUB_SRC)" ]; then \
		echo "ERROR: vgpu-stub source file not found: $(QEMU_STUB_SRC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(INC_DIR)/vgpu_protocol.h" ]; then \
		echo "ERROR: vgpu_protocol.h not found: $(INC_DIR)/vgpu_protocol.h"; \
		exit 1; \
	fi
	@if [ ! -f "$(INC_DIR)/cuda_protocol.h" ]; then \
		echo "ERROR: cuda_protocol.h not found: $(INC_DIR)/cuda_protocol.h"; \
		exit 1; \
	fi
	@echo "  Verifying source file integrity..."
	@if ! grep -q 'connect(fd,' "$(QEMU_STUB_SRC)"; then \
		echo "ERROR: Source file $(QEMU_STUB_SRC) appears corrupted"; \
		echo "       The 'connect(fd,' function call is missing."; \
		echo "       Please check the source file."; \
		exit 1; \
	fi
	@if ! grep -q "vgpu_try_connect_mediator" "$(QEMU_STUB_SRC)"; then \
		echo "ERROR: Source file $(QEMU_STUB_SRC) appears corrupted"; \
		echo "       The 'vgpu_try_connect_mediator' function is missing."; \
		exit 1; \
	fi
	@if ! grep -q 'VGPU_SOCKET_HDR_SIZE' "$(QEMU_STUB_SRC)"; then \
		echo "ERROR: Source file $(QEMU_STUB_SRC) appears corrupted"; \
		echo "       The 'VGPU_SOCKET_HDR_SIZE' constant is missing."; \
		exit 1; \
	fi
	@if ! grep -qE 'sock_rx_len.*VGPU_SOCKET_HDR_SIZE' "$(QEMU_STUB_SRC)"; then \
		echo "ERROR: Source file $(QEMU_STUB_SRC) appears corrupted"; \
		echo "       The critical check 'sock_rx_len < VGPU_SOCKET_HDR_SIZE' is missing."; \
		echo ""; \
		echo "       Expected code at line 430:"; \
		echo "         if (s->sock_rx_len < VGPU_SOCKET_HDR_SIZE) {"; \
		echo ""; \
		echo "       Debug info - searching for pattern in file:"; \
		grep -n "sock_rx_len" "$(QEMU_STUB_SRC)" | head -3 || echo "       (no 'sock_rx_len' found at all)"; \
		grep -n "VGPU_SOCKET_HDR_SIZE" "$(QEMU_STUB_SRC)" | head -3 || echo "       (no 'VGPU_SOCKET_HDR_SIZE' found at all)"; \
		echo ""; \
		echo "       Checking lines around 430:"; \
		sed -n '428,432p' "$(QEMU_STUB_SRC)" 2>/dev/null | sed 's/^/       /' || echo "       (could not read lines 428-432)"; \
		echo ""; \
		echo "       Action: Copy the correct file from your local machine to the XCP-NG host."; \
		exit 1; \
	fi
	@if ! grep -q 'VGPU_MSG_REQUEST' "$(QEMU_STUB_SRC)"; then \
		echo "ERROR: Source file $(QEMU_STUB_SRC) appears corrupted"; \
		echo "       The 'VGPU_MSG_REQUEST' constant is missing."; \
		exit 1; \
	fi
	@if ! grep -q 'PCI_BASE_ADDRESS_SPACE_MEMORY' "$(QEMU_STUB_SRC)"; then \
		echo "ERROR: Source file $(QEMU_STUB_SRC) appears corrupted"; \
		echo "       The 'PCI_BASE_ADDRESS_SPACE_MEMORY' constant is missing."; \
		exit 1; \
	fi
	@if ! grep -q 'pci_register_bar.*&s->mmio' "$(QEMU_STUB_SRC)"; then \
		echo "ERROR: Source file $(QEMU_STUB_SRC) appears corrupted"; \
		echo "       The 'pci_register_bar' call is missing the memory region parameter."; \
		exit 1; \
	fi
	@echo "✓ RPM build environment verified"
	@echo "  Build directory: $(RPM_BUILD)"
	@echo "  Spec file: $(QEMU_SPEC)"
	@echo "  Source file integrity: OK"

# ---- Prepare: Copy files to RPM SOURCES directory -------------------------
qemu-prepare: qemu-check
	@echo "Preparing RPM build sources..."
	@mkdir -p "$(RPM_SOURCES)"
	@if [ -f "$(QEMU_STUB_RPM)" ]; then \
		echo "  ⚠  Warning: $(QEMU_STUB_RPM) already exists (will be overwritten)"; \
	fi
	@if [ -f "$(QEMU_PROTO_RPM)" ]; then \
		echo "  ⚠  Warning: $(QEMU_PROTO_RPM) already exists (will be overwritten)"; \
	fi
	@echo "  Copying source files..."
	@cp "$(QEMU_STUB_SRC)" "$(QEMU_STUB_RPM)"
	@echo "  ✓ Copied $(QEMU_STUB_SRC) → $(QEMU_STUB_RPM)"
	@cp "$(INC_DIR)/vgpu_protocol.h" "$(QEMU_PROTO_RPM)"
	@echo "  ✓ Copied $(INC_DIR)/vgpu_protocol.h → $(QEMU_PROTO_RPM)"
	@cp "$(INC_DIR)/cuda_protocol.h" "$(QEMU_CUDA_PROTO_RPM)"
	@echo "  ✓ Copied $(INC_DIR)/cuda_protocol.h → $(QEMU_CUDA_PROTO_RPM)"
	@echo "  Verifying file integrity..."
	@if ! grep -q 'connect(fd,' "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     The 'connect(fd,' function call is missing or corrupted."; \
		echo "     This usually indicates the file was corrupted during copy."; \
		echo "     Please check the source file and try again."; \
		exit 1; \
	fi
	@if ! grep -q "vgpu_try_connect_mediator" "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     The 'vgpu_try_connect_mediator' function is missing."; \
		exit 1; \
	fi
	@if ! grep -q "struct sockaddr_un" "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     Socket structures are missing."; \
		exit 1; \
	fi
	@if ! grep -q 'VGPU_SOCKET_HDR_SIZE' "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     The 'VGPU_SOCKET_HDR_SIZE' constant is missing or corrupted."; \
		echo "     This usually indicates the file was corrupted during copy."; \
		exit 1; \
	fi
	@if grep -q 'VGPU{' "$(QEMU_STUB_RPM)" 2>/dev/null; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     Found corrupted pattern 'VGPU{' (should be 'VGPU_SOCKET_HDR_SIZE')."; \
		echo "     This indicates truncation during file copy."; \
		exit 1; \
	fi
	@if ! grep -qE 'sock_rx_len.*VGPU_SOCKET_HDR_SIZE' "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     The critical check 'sock_rx_len < VGPU_SOCKET_HDR_SIZE' is missing or corrupted."; \
		echo ""; \
		echo "     Expected code at line 430:"; \
		echo "       if (s->sock_rx_len < VGPU_SOCKET_HDR_SIZE) {"; \
		echo ""; \
		echo "     Debug info - searching for pattern in copied file:"; \
		grep -n "sock_rx_len" "$(QEMU_STUB_RPM)" | head -3 || echo "     (no 'sock_rx_len' found at all)"; \
		grep -n "VGPU_SOCKET_HDR_SIZE" "$(QEMU_STUB_RPM)" | head -3 || echo "     (no 'VGPU_SOCKET_HDR_SIZE' found at all)"; \
		echo ""; \
		echo "     Checking lines around 430:"; \
		sed -n '428,432p' "$(QEMU_STUB_RPM)" 2>/dev/null | sed 's/^/     /' || echo "     (could not read lines 428-432)"; \
		echo ""; \
		echo "     Action: The file was corrupted during copy. Check the source file and try again."; \
		exit 1; \
	fi
	@if ! grep -q 'VGPU_MSG_REQUEST' "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     The 'VGPU_MSG_REQUEST' constant is missing or corrupted."; \
		exit 1; \
	fi
	@if grep -qE 'msg_type\s*=\s*ST[^A-Za-z_]' "$(QEMU_STUB_RPM)" 2>/dev/null; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     Found corrupted pattern 'msg_type = ST' (should be 'VGPU_MSG_REQUEST')."; \
		echo "     This indicates the file on the host is corrupted."; \
		echo "     Please copy the correct file from your local machine."; \
		exit 1; \
	fi
	@if ! grep -q 'PCI_BASE_ADDRESS_SPACE_MEMORY' "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     The 'PCI_BASE_ADDRESS_SPACE_MEMORY' constant is missing or corrupted."; \
		exit 1; \
	fi
	@if grep -q 'PCI_BASE_ADDRESS_SPACE_Mmio' "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     Found corrupted pattern 'PCI_BASE_ADDRESS_SPACE_Mmio' (should be 'PCI_BASE_ADDRESS_SPACE_MEMORY')."; \
		exit 1; \
	fi
	@if ! grep -q 'pci_register_bar.*&s->mmio' "$(QEMU_STUB_RPM)"; then \
		echo "  ✗ ERROR: File corruption detected in $(QEMU_STUB_RPM)"; \
		echo "     The 'pci_register_bar' call is missing the memory region parameter."; \
		exit 1; \
	fi
	@echo "  ✓ File integrity verified"

# ---- Spec file integration ------------------------------------------------
# Modify qemu.spec to include vgpu-stub sources and build integration
qemu-integrate: qemu-prepare
	@echo "Integrating vgpu-stub into QEMU spec file..."
	@if ! grep -q "Source3: vgpu-stub.c" "$(QEMU_SPEC)"; then \
		echo "  Adding Source3 line for vgpu-stub.c..."; \
		sed -i '/^Source2:/a Source3: vgpu-stub.c' "$(QEMU_SPEC)"; \
		echo "  ✓ Added Source3: vgpu-stub.c"; \
	else \
		echo "  ✓ Source3 already present"; \
	fi
	@if ! grep -q "Source4: vgpu_protocol.h" "$(QEMU_SPEC)"; then \
		echo "  Adding Source4 line for vgpu_protocol.h..."; \
		sed -i '/^Source3:/a Source4: vgpu_protocol.h' "$(QEMU_SPEC)"; \
		echo "  ✓ Added Source4: vgpu_protocol.h"; \
	else \
		echo "  ✓ Source4 already present"; \
	fi
	@if ! grep -q "Source5: cuda_protocol.h" "$(QEMU_SPEC)"; then \
		echo "  Adding Source5 line for cuda_protocol.h..."; \
		sed -i '/^Source4:/a Source5: cuda_protocol.h' "$(QEMU_SPEC)"; \
		echo "  ✓ Added Source5: cuda_protocol.h"; \
	else \
		echo "  ✓ Source5 already present"; \
	fi
	@if ! grep -q "cp -a %{SOURCE4}" "$(QEMU_SPEC)"; then \
		echo "  Adding protocol header copy to spec..."; \
		if grep -q "cp -a %{SOURCE3}" "$(QEMU_SPEC)"; then \
			sed -i '/cp -a %{SOURCE3} hw\/misc\/vgpu-stub.c/a \
cp -a %{SOURCE4} hw/misc/vgpu_protocol.h\
cp -a %{SOURCE5} hw/misc/cuda_protocol.h' "$(QEMU_SPEC)"; \
			echo "  ✓ Added copy of vgpu_protocol.h and cuda_protocol.h to hw/misc/"; \
		else \
			sed -i '/tar xzf.*SOURCE2/a \
\
# Add vgpu device (MMIO + CUDA remoting)\
cp -a %{SOURCE3} hw/misc/vgpu-stub.c\
cp -a %{SOURCE4} hw/misc/vgpu_protocol.h\
cp -a %{SOURCE5} hw/misc/cuda_protocol.h\
echo "obj-y += vgpu-stub.o" >> hw/misc/Makefile.objs' "$(QEMU_SPEC)"; \
			echo "  ✓ Added full vgpu integration block"; \
		fi; \
	else \
		echo "  ✓ Protocol header copy already in spec"; \
	fi
	@if ! grep -q "cp -a %{SOURCE5}" "$(QEMU_SPEC)"; then \
		echo "  Adding cuda_protocol.h copy to spec..."; \
		sed -i '/cp -a %{SOURCE4} hw\/misc\/vgpu_protocol.h/a \
cp -a %{SOURCE5} hw/misc/cuda_protocol.h' "$(QEMU_SPEC)"; \
		echo "  ✓ Added copy of cuda_protocol.h to hw/misc/"; \
	else \
		echo "  ✓ cuda_protocol.h copy already in spec"; \
	fi
	@if ! grep -q "vgpu-stub.o" "$(QEMU_SPEC)"; then \
		sed -i '/cp -a %{SOURCE4} hw\/misc\/vgpu_protocol.h/a \
echo "obj-y += vgpu-stub.o" >> hw/misc/Makefile.objs' "$(QEMU_SPEC)"; \
		echo "  ✓ Added Makefile.objs integration"; \
	else \
		echo "  ✓ Makefile.objs integration present"; \
	fi
	@if ! grep -q "\-\-disable-werror" "$(QEMU_SPEC)"; then \
		sed -i 's/--enable-werror/--disable-werror/' "$(QEMU_SPEC)"; \
		echo "  ✓ Switched to --disable-werror"; \
	else \
		echo "  ✓ --disable-werror already set"; \
	fi
	@echo ""
	@echo "Verifying spec file..."
	@echo -n "  Source3: "; { grep "Source3: vgpu-stub.c" "$(QEMU_SPEC)" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }
	@echo -n "  Source4: "; { grep "Source4: vgpu_protocol.h" "$(QEMU_SPEC)" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }
	@echo -n "  Source5: "; { grep "Source5: cuda_protocol.h" "$(QEMU_SPEC)" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }
	@echo -n "  Copy stub: "; { grep "vgpu-stub.c" "$(QEMU_SPEC)" | grep "cp" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }
	@echo -n "  Copy vgpu_protocol: "; { grep "vgpu_protocol.h" "$(QEMU_SPEC)" | grep "cp" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }
	@echo -n "  Copy cuda_protocol: "; { grep "cuda_protocol.h" "$(QEMU_SPEC)" | grep "cp" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }
	@echo -n "  Makefile: "; { grep "vgpu-stub.o" "$(QEMU_SPEC)" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }
	@echo -n "  Werror: "; { grep "disable-werror" "$(QEMU_SPEC)" | head -1 | sed 's/[\\]$$//' || echo "MISSING!"; }

# ---- Build QEMU RPM --------------------------------------------------------
qemu-build: qemu-integrate
	@echo "Building QEMU RPM (this takes 30-45 minutes)..."
	@echo "  Build log: $(RPM_BUILD)/BUILD/qemu-*/build.log"
	@echo ""
	@cd "$(RPM_BUILD)" && \
	rpmbuild -bb --define "_topdir $(RPM_BUILD)" SPECS/qemu.spec 2>&1 | tee /tmp/qemu-build.log || \
		(echo "ERROR: RPM build failed. Check /tmp/qemu-build.log for details." && exit 1)
	@echo ""
	@echo "Checking build result..."
	@RPM_FILE=$$(ls -1 "$(RPM_BUILD)/RPMS/x86_64/qemu-"*.rpm 2>/dev/null | head -1); \
	if [ -z "$$RPM_FILE" ]; then \
		echo "ERROR: RPM build failed — no RPM file found."; \
		echo "Check /tmp/qemu-build.log for details."; \
		exit 1; \
	else \
		echo "  ✓ Built: $$RPM_FILE"; \
	fi
	@echo "  ✓ QEMU RPM build complete"

# ---- Install QEMU RPM (optional) -------------------------------------------
qemu-install: qemu-build
	@echo "Installing QEMU RPM..."
	@echo "  ⚠  This step requires root privileges and will install QEMU system-wide"
	@echo "  ⚠  Ensure all VMs using the vgpu device are stopped before installing"
	@RPM_FILE=$$(ls -1 "$(RPM_BUILD)/RPMS/x86_64/qemu-"*.rpm 2>/dev/null | head -1); \
	if [ -z "$$RPM_FILE" ]; then \
		echo "ERROR: No RPM file found. Build may have failed."; \
		exit 1; \
	fi; \
	echo "  RPM to install: $$RPM_FILE"; \
	read -p "  Continue with installation? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		sudo rpm -Uvh --nodeps --force "$$RPM_FILE"; \
		echo "  ✓ QEMU installed"; \
		echo ""; \
		echo "  Verification:"; \
		QEMU_BIN="/usr/lib64/xen/bin/qemu-system-i386"; \
		if [ -x "$$QEMU_BIN" ]; then \
			echo -n "    QEMU version: "; $$QEMU_BIN --version 2>/dev/null | head -1; \
			echo -n "    vgpu-cuda device: "; \
			if $$QEMU_BIN -device help 2>/dev/null | grep -q "vgpu-cuda"; then \
				echo "✓ AVAILABLE"; \
			else \
				echo "✗ NOT FOUND"; \
			fi; \
		else \
			echo "    ⚠ QEMU binary not found at $$QEMU_BIN"; \
		fi; \
	else \
		echo "  Installation skipped"; \
	fi

# ---- Main QEMU target -------------------------------------------------------
qemu: qemu-build
	@echo ""
	@echo "QEMU build complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Install QEMU RPM: make qemu-install"
	@echo "  2. Configure a test VM with vgpu-cuda:"
	@echo "     xe vm-param-set uuid=<VM_UUID> \\"
	@echo "       platform:device-model-args=\"-device vgpu-cuda,pool_id=B,priority=high,vm_id=200\""
	@echo "  3. Start the VM and verify the device appears in lspci"
	@echo "  4. Start the mediator daemon on dom0"

# ---- Clean QEMU build artifacts ------------------------------------------------
qemu-clean:
	@echo "Cleaning QEMU build artifacts..."
	@if [ -d "$(RPM_BUILD)/BUILD" ]; then \
		echo "  Removing QEMU build directory..."; \
		rm -rf "$(RPM_BUILD)/BUILD"/*; \
	fi
	@if [ -f "$(QEMU_STUB_RPM)" ]; then \
		echo "  Removing copied vgpu-stub.c..."; \
		rm -f "$(QEMU_STUB_RPM)"; \
	fi
	@if [ -f "$(QEMU_PROTO_RPM)" ]; then \
		echo "  Removing copied vgpu_protocol.h..."; \
		rm -f "$(QEMU_PROTO_RPM)"; \
	fi
	@if [ -f "$(QEMU_CUDA_PROTO_RPM)" ]; then \
		echo "  Removing copied cuda_protocol.h..."; \
		rm -f "$(QEMU_CUDA_PROTO_RPM)"; \
	fi
	@if [ -d "$(RPM_BUILD)/RPMS" ]; then \
		echo "  Removing built RPM files..."; \
		rm -f "$(RPM_BUILD)/RPMS"/x86_64/qemu-*.rpm; \
	fi
	@if [ -f "/tmp/qemu-build.log" ]; then \
		echo "  Removing build log..."; \
		rm -f /tmp/qemu-build.log; \
	fi
	@echo "  ✓ QEMU build artifacts cleaned"
	@echo ""
	@echo "Note: This does NOT remove:"
	@echo "  - RPM build directory structure ($(RPM_BUILD))"
	@echo "  - QEMU spec file ($(QEMU_SPEC))"
	@echo "  - Source files in $(RPM_SOURCES) (except vgpu-stub files)"
	@echo ""
	@echo "To remove everything including spec file modifications, manually delete:"
	@echo "  $(RPM_BUILD)"
