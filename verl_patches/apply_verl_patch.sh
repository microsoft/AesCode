#!/bin/bash
# Apply custom verl patches (Qwen3 VL support, GDPO fixes, generate-only training)
# Usage: bash apply_verl_patch.sh /path/to/site-packages/verl
#
# The patch is generated against verl commit 4b9c14f2c306d43f84eccf47315323e3f1ce1270
# and must be applied after `pip install git+https://github.com/volcengine/verl.git`
# at that same commit.

set -euo pipefail

VERL_DIR="${1:?Usage: bash apply_verl_patch.sh /path/to/site-packages/verl}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/verl_custom.patch"

if [ ! -d "$VERL_DIR" ]; then
    echo "ERROR: verl directory not found: $VERL_DIR"
    exit 1
fi

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: patch file not found: $PATCH_FILE"
    exit 1
fi

echo "Applying verl custom patches to: $VERL_DIR"
echo "  Patch file: $PATCH_FILE"
echo "  Modified files: $(grep -c '^--- ' "$PATCH_FILE")"

if patch -p1 -N --dry-run -d "$VERL_DIR" < "$PATCH_FILE" > /dev/null 2>&1; then
    patch -p1 -N -d "$VERL_DIR" < "$PATCH_FILE"
    echo "Done. All hunks applied."
    exit 0
fi

if patch -p1 -R --dry-run -d "$VERL_DIR" < "$PATCH_FILE" > /dev/null 2>&1; then
    echo "Already applied; nothing to do."
    exit 0
fi

echo "ERROR: the patch does not apply cleanly to $VERL_DIR" >&2
echo "Install verl at commit 4b9c14f2c306d43f84eccf47315323e3f1ce1270 first." >&2
patch -p1 -N --dry-run -d "$VERL_DIR" < "$PATCH_FILE" >&2 || true
exit 1
