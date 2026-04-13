#!/bin/bash
# create_symlinks.sh - Create symlinks for libcuda.so and libnvidia-ml.so redirection
#
# This script creates symlinks in standard library paths to redirect
# libcuda.so and libnvidia-ml.so requests to our shim libraries. This works
# at the filesystem level and doesn't require interception, making it 100%
# reliable even when Go uses direct syscalls.
#
# Usage: sudo ./create_symlinks.sh

set -e

SHIM_CUDA="/usr/lib64/libvgpu-cuda.so"
SHIM_NVML="/usr/lib64/libvgpu-nvml.so"

# Standard library paths for CUDA
CUDA_SYMLINK_PATHS=(
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    "/usr/lib/x86_64-linux-gnu/libcuda.so"
    "/usr/lib64/libcuda.so.1"
    "/usr/lib64/libcuda.so"
    "/usr/local/lib/libcuda.so.1"
    "/usr/local/lib/libcuda.so"
    "/lib/x86_64-linux-gnu/libcuda.so.1"
    "/lib/x86_64-linux-gnu/libcuda.so"
)

# Ollama-specific paths for CUDA
OLLAMA_CUDA_PATHS=(
    "/usr/local/lib/ollama/libcuda.so.1"
    "/usr/local/lib/ollama/libcuda.so"
    "/usr/local/lib/ollama/cuda_v12/libcuda.so.1"
    "/usr/local/lib/ollama/cuda_v12/libcuda.so"
    "/usr/local/lib/ollama/cuda_v13/libcuda.so.1"
    "/usr/local/lib/ollama/cuda_v13/libcuda.so"
)

# Standard library paths for NVML
NVML_SYMLINK_PATHS=(
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so"
    "/usr/lib64/libnvidia-ml.so.1"
    "/usr/lib64/libnvidia-ml.so"
    "/usr/local/lib/libnvidia-ml.so.1"
    "/usr/local/lib/libnvidia-ml.so"
    "/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    "/lib/x86_64-linux-gnu/libnvidia-ml.so"
)

# Ollama-specific paths for NVML
OLLAMA_NVML_PATHS=(
    "/usr/local/lib/ollama/libnvidia-ml.so.1"
    "/usr/local/lib/ollama/libnvidia-ml.so"
    "/usr/local/lib/ollama/cuda_v12/libnvidia-ml.so.1"
    "/usr/local/lib/ollama/cuda_v12/libnvidia-ml.so"
    "/usr/local/lib/ollama/cuda_v13/libnvidia-ml.so.1"
    "/usr/local/lib/ollama/cuda_v13/libnvidia-ml.so"
)

# Combine all paths
ALL_CUDA_PATHS=("${CUDA_SYMLINK_PATHS[@]}" "${OLLAMA_CUDA_PATHS[@]}")
ALL_NVML_PATHS=("${NVML_SYMLINK_PATHS[@]}" "${OLLAMA_NVML_PATHS[@]}")

# Check if shim libraries exist
if [ ! -f "$SHIM_CUDA" ]; then
    echo "ERROR: CUDA shim library not found: $SHIM_CUDA"
    echo "Please build the shim library first."
    exit 1
fi

if [ ! -f "$SHIM_NVML" ]; then
    echo "WARNING: NVML shim library not found: $SHIM_NVML"
    echo "NVML symlinks will be skipped. CUDA symlinks will still be created."
    NVML_AVAILABLE=0
else
    NVML_AVAILABLE=1
fi

echo "Creating symlinks for library redirection..."
echo "CUDA shim: $SHIM_CUDA"
[ "$NVML_AVAILABLE" = "1" ] && echo "NVML shim: $SHIM_NVML"
echo ""

# Function to create symlink
# Returns: 0=created, 1=skipped, 2=created_with_backup
create_symlink() {
    local symlink_path="$1"
    local target_lib="$2"
    local lib_name="$3"
    local backed_up=0
    
    # Create directory structure if needed
    local symlink_dir=$(dirname "$symlink_path")
    if [ ! -d "$symlink_dir" ]; then
        echo "Creating directory: $symlink_dir"
        mkdir -p "$symlink_dir"
    fi
    
    if [ -L "$symlink_path" ]; then
        # Symlink already exists - check if it points to our shim
        local current_target=$(readlink -f "$symlink_path" 2>/dev/null || echo "")
        if [ "$current_target" = "$target_lib" ]; then
            echo "✓ Already exists and correct: $symlink_path"
            return 1  # Skipped
        else
            # Backup existing symlink
            local backup_path="${symlink_path}.backup.$(date +%Y%m%d_%H%M%S)"
            echo "Backing up existing symlink: $symlink_path -> $backup_path"
            mv "$symlink_path" "$backup_path"
            ln -sf "$target_lib" "$symlink_path"
            echo "✓ Created: $symlink_path -> $target_lib"
            return 2  # Created with backup
        fi
    elif [ -f "$symlink_path" ]; then
        # Regular file exists - backup it
        local backup_path="${symlink_path}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "WARNING: Regular file exists, backing up: $symlink_path -> $backup_path"
        mv "$symlink_path" "$backup_path"
        ln -sf "$target_lib" "$symlink_path"
        echo "✓ Created: $symlink_path -> $target_lib"
        return 2  # Created with backup
    else
        # No file exists - create symlink
        ln -sf "$target_lib" "$symlink_path"
        echo "✓ Created: $symlink_path -> $target_lib"
        return 0  # Created
    fi
}

# Create CUDA symlinks
echo "Creating CUDA symlinks..."
cuda_created=0
cuda_skipped=0
cuda_backed_up=0

for symlink_path in "${ALL_CUDA_PATHS[@]}"; do
    result=$(create_symlink "$symlink_path" "$SHIM_CUDA" "CUDA")
    case $result in
        0) cuda_created=$((cuda_created + 1)) ;;
        1) cuda_skipped=$((cuda_skipped + 1)) ;;
        2) cuda_created=$((cuda_created + 1))
           cuda_backed_up=$((cuda_backed_up + 1)) ;;
    esac
done

# Create NVML symlinks (if available)
nvml_created=0
nvml_skipped=0
nvml_backed_up=0

if [ "$NVML_AVAILABLE" = "1" ]; then
    echo ""
    echo "Creating NVML symlinks..."
    for symlink_path in "${ALL_NVML_PATHS[@]}"; do
        result=$(create_symlink "$symlink_path" "$SHIM_NVML" "NVML")
        case $result in
            0) nvml_created=$((nvml_created + 1)) ;;
            1) nvml_skipped=$((nvml_skipped + 1)) ;;
            2) nvml_created=$((nvml_created + 1))
               nvml_backed_up=$((nvml_backed_up + 1)) ;;
        esac
    done
fi

echo ""
echo "Summary:"
echo "  CUDA symlinks:"
echo "    Created: $cuda_created"
echo "    Skipped: $cuda_skipped (already correct)"
echo "    Backed up: $cuda_backed_up existing files/symlinks"
if [ "$NVML_AVAILABLE" = "1" ]; then
    echo "  NVML symlinks:"
    echo "    Created: $nvml_created"
    echo "    Skipped: $nvml_skipped (already correct)"
    echo "    Backed up: $nvml_backed_up existing files/symlinks"
fi
echo ""
echo "✓ Symlink creation complete"
