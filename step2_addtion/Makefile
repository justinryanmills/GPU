# Makefile for Enhanced vGPU Mediator and Test Programs

CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lpthread

# CUDA paths (adjust as needed)
CUDA_PATH ?= /usr/local/cuda
CUDA_LIBDIR = $(CUDA_PATH)/lib64
CUDA_INCDIR = $(CUDA_PATH)/include
NVCC ?= nvcc
NVCC_FLAGS = -O2 -arch=sm_50 -I. -I$(CUDA_INCDIR) -x cu

# Source files
MEDIATOR_SRC = mediator_enhanced.c
MEDIATOR_OBJ = $(MEDIATOR_SRC:.c=.o)
MEDIATOR_BIN = mediator_enhanced

TEST_SRC = test_vgpu_enhanced.c
TEST_BIN = test_vgpu_enhanced

VM_CLIENT_SRC = vm_client_enhanced.c
VM_CLIENT_BIN = vm_client_enhanced

# CUDA source (use current directory if present, otherwise step2_test)
CUDA_SRC = cuda_vector_add.c
CUDA_OBJ = cuda_vector_add.o

# Headers
PROTOCOL_H = vgpu_protocol.h

# Default target
all: $(MEDIATOR_BIN) $(TEST_BIN) $(VM_CLIENT_BIN)

# Enhanced mediator (requires CUDA)
$(MEDIATOR_BIN): $(MEDIATOR_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -L$(CUDA_LIBDIR) -lcudart -lcuda
	@echo "Built: $@"

# Test program (no CUDA needed)
$(TEST_BIN): $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ $<

# Enhanced VM client (no CUDA needed)
$(VM_CLIENT_BIN): $(VM_CLIENT_SRC)
	$(CC) $(CFLAGS) -o $@ $<

# Object files
%.o: %.c $(PROTOCOL_H)
	$(CC) $(CFLAGS) -I. -I$(CUDA_INCDIR) -c $< -o $@

# CUDA object file (must use nvcc, not gcc)
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(MEDIATOR_BIN) $(TEST_BIN) $(VM_CLIENT_BIN) $(MEDIATOR_OBJ) $(CUDA_OBJ)

# Install mediator (copy to /usr/local/bin)
install: $(MEDIATOR_BIN)
	sudo cp $(MEDIATOR_BIN) /usr/local/bin/
	sudo chmod +x /usr/local/bin/$(MEDIATOR_BIN)
	@echo "Installed $(MEDIATOR_BIN) to /usr/local/bin/"

.PHONY: all clean install
