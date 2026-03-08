# Makefile for CUDA Vector Addition System
# 
# IMPORTANT: Different components build on different machines!
# 
# Components:
#   - mediator_async: MEDIATOR daemon (build on Dom0/host - needs CUDA)
#   - vm_client_vector: VM client (build on VM/guest - NO CUDA needed)
#
# Usage:
#   On Dom0:  make dom0    (builds MEDIATOR)
#   On VM:     make vm      (builds VM_CLIENT)

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -g
NVCCFLAGS = -arch=sm_80 -O2 -g  # sm_80 for H100
LDFLAGS = -lpthread
CUDA_LDFLAGS = -lcudart -L/usr/local/cuda/lib64

# Directories
SRC_DIR = .
BUILD_DIR = build
DOM0_BUILD_DIR = build-dom0
VM_BUILD_DIR = build-vm
INSTALL_DIR = /usr/local

# Source files
CUDA_SRC = cuda_vector_add.c
MEDIATOR_SRC = mediator_async.c
VM_CLIENT_SRC = vm_client_vector.c
TEST_CLIENT_SRC = test_mediator_client.c

# Executables
CUDA_EXEC = $(DOM0_BUILD_DIR)/cuda_vector_add
MEDIATOR_EXEC = $(DOM0_BUILD_DIR)/mediator_async
VM_CLIENT_EXEC = $(VM_BUILD_DIR)/vm_client_vector
TEST_CLIENT_EXEC = $(VM_BUILD_DIR)/test_mediator_client

# Default target (shows help)
all: help

# ============================================================================
# DOM0 (Host) Builds - Requires CUDA
# ============================================================================

# Build everything for Dom0 (MEDIATOR + CUDA test)
dom0: $(DOM0_BUILD_DIR) $(MEDIATOR_EXEC) $(CUDA_EXEC)
	@echo ""
	@echo "=========================================="
	@echo "Dom0 build complete!"
	@echo "  MEDIATOR: $(MEDIATOR_EXEC)"
	@echo "  CUDA test: $(CUDA_EXEC)"
	@echo "=========================================="

# Create Dom0 build directory
$(DOM0_BUILD_DIR):
	mkdir -p $(DOM0_BUILD_DIR)

# CUDA vector addition test (for Dom0)
$(CUDA_EXEC): $(CUDA_SRC) | $(DOM0_BUILD_DIR)
	@echo "Building CUDA test executable (Dom0)..."
	$(NVCC) $(NVCCFLAGS) -x cu -o $@ $< $(CUDA_LDFLAGS) -DCUDA_VECTOR_ADD_TEST
	@echo "Built: $@"

# MEDIATOR daemon (for Dom0 - links with CUDA)
# Compile CUDA code with nvcc, C code with gcc, then link
$(MEDIATOR_EXEC): $(MEDIATOR_SRC) $(CUDA_SRC) | $(DOM0_BUILD_DIR)
	@echo "Building MEDIATOR daemon (Dom0)..."
	@echo "  Compiling CUDA code with nvcc (force CUDA mode)..."
	$(NVCC) $(NVCCFLAGS) -x cu -c $(CUDA_SRC) -o $(DOM0_BUILD_DIR)/cuda_vector_add.o
	@echo "  Compiling MEDIATOR code with gcc..."
	$(CC) $(CFLAGS) -c $(MEDIATOR_SRC) -o $(DOM0_BUILD_DIR)/mediator_async.o -I$(SRC_DIR)
	@echo "  Linking..."
	$(NVCC) $(NVCCFLAGS) -o $@ $(DOM0_BUILD_DIR)/mediator_async.o $(DOM0_BUILD_DIR)/cuda_vector_add.o $(LDFLAGS) $(CUDA_LDFLAGS)
	@echo "Built: $@"

# ============================================================================
# VM (Guest) Builds - NO CUDA needed
# ============================================================================

# Build VM client (no CUDA dependencies)
vm: $(VM_BUILD_DIR) $(VM_CLIENT_EXEC) $(TEST_CLIENT_EXEC)
	@echo ""
	@echo "=========================================="
	@echo "VM build complete!"
	@echo "  VM Client: $(VM_CLIENT_EXEC)"
	@echo "  Test Client: $(TEST_CLIENT_EXEC)"
	@echo "=========================================="

# Create VM build directory
$(VM_BUILD_DIR):
	mkdir -p $(VM_BUILD_DIR)

# VM client (standalone, no CUDA - for VM/guest)
$(VM_CLIENT_EXEC): $(VM_CLIENT_SRC) | $(VM_BUILD_DIR)
	@echo "Building VM client (VM/guest)..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

# Test MEDIATOR client (for VM/guest - testing tool)
$(TEST_CLIENT_EXEC): $(TEST_CLIENT_SRC) | $(VM_BUILD_DIR)
	@echo "Building test MEDIATOR client (VM/guest)..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lm
	@echo "Built: $@"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(DOM0_BUILD_DIR) $(VM_BUILD_DIR)
	@echo "Cleaned all build directories"

clean-dom0:
	rm -rf $(DOM0_BUILD_DIR)
	@echo "Cleaned Dom0 build directory (including object files)"

clean-vm:
	rm -rf $(VM_BUILD_DIR)
	@echo "Cleaned VM build directory"

# Install on Dom0 (MEDIATOR)
install-dom0: $(MEDIATOR_EXEC)
	install -d $(INSTALL_DIR)/bin
	install -m 755 $(MEDIATOR_EXEC) $(INSTALL_DIR)/bin/
	@echo "Installed MEDIATOR to $(INSTALL_DIR)/bin/mediator_async"

# Install on VM (VM client)
install-vm: $(VM_CLIENT_EXEC)
	install -d $(INSTALL_DIR)/bin
	install -m 755 $(VM_CLIENT_EXEC) $(INSTALL_DIR)/bin/
	@echo "Installed VM client to $(INSTALL_DIR)/bin/vm_client_vector"

# Test CUDA component (Dom0 only)
test-cuda: $(CUDA_EXEC)
	@echo "Testing CUDA component (Dom0)..."
	$(CUDA_EXEC)

# Help
help:
	@echo "=================================================================================="
	@echo "                    CUDA Vector Addition System - Build Guide"
	@echo "=================================================================================="
	@echo ""
	@echo "IMPORTANT: Different components build on different machines!"
	@echo ""
	@echo "ON DOM0 (Host):"
	@echo "  make dom0          - Build MEDIATOR daemon (requires CUDA)"
	@echo "  make install-dom0  - Install MEDIATOR to /usr/local/bin"
	@echo "  make test-cuda     - Test CUDA component"
	@echo "  make clean-dom0    - Clean Dom0 build directory"
	@echo ""
	@echo "ON VM (Guest):"
	@echo "  make vm            - Build VM client (NO CUDA needed)"
	@echo "  make install-vm    - Install VM client to /usr/local/bin"
	@echo "  make clean-vm      - Clean VM build directory"
	@echo ""
	@echo "General:"
	@echo "  make clean         - Clean all build directories"
	@echo ""
	@echo "Build Outputs:"
	@echo "  Dom0: $(MEDIATOR_EXEC)"
	@echo "  VM:   $(VM_CLIENT_EXEC)"
	@echo "  Test: $(TEST_CLIENT_EXEC)"
	@echo ""
	@echo "=================================================================================="

.PHONY: all dom0 vm clean clean-dom0 clean-vm install-dom0 install-vm test-cuda help
