#!/bin/bash
set -e
COREDIR="${1:-/tmp/cores}"
CORE="${2:-}"
if [ -z "$CORE" ]; then
  CORE=$(ls -t "$COREDIR"/core.ollama.* 2>/dev/null | head -1)
fi
[ -n "$CORE" ] || { echo "No core in $COREDIR"; exit 1; }
EXEC=/bin/bash
echo "=== Core: $CORE ==="
cd "$(dirname "$CORE")"
gdb -batch \
  -ex "set pagination off" \
  -ex "bt" \
  -ex "bt full" \
  -ex "info reg rip rsp rbp" \
  -ex "x/i \$rip" \
  -ex "info sharedlibrary" \
  -ex "quit" \
  "$EXEC" "$(basename "$CORE")" 2>&1
