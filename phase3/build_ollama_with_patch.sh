#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/patches/ollama_runner_ld_preload.patch"
DEST="${1:-${SCRIPT_DIR}/ollama-build}"
BUILD_DIR="${DEST}/ollama"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch not found: $PATCH_FILE"
  exit 1
fi

if ! command -v go &>/dev/null; then
  echo "Go is not installed. Install Go (e.g. sudo apt install golang-go) and run this script again."
  exit 1
fi

mkdir -p "$DEST"
cd "$DEST"

if [[ ! -d ollama/.git ]]; then
  echo "Cloning ollama..."
  git clone --depth 1 https://github.com/ollama/ollama.git
fi

cd ollama
echo "Applying patch..."
if ! patch -p1 < "$PATCH_FILE"; then
  echo "Patch failed. Apply manually - see OLLAMA_RUNNER_LD_PRELOAD_PATCH.md"
  exit 1
fi

echo "Building..."
go build -o ollama .

echo "Done. Binary: $(pwd)/ollama"
echo "Copy to VM: /usr/local/bin/ollama.bin; sudo systemctl restart ollama.service"
