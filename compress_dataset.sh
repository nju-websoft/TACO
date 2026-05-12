#!/bin/bash
# Compress dataset directory: only include json, profile.txt, index, pkl files
# Exclude filtered subdirectories (miwv, superfiltering, selectit, alpagasus, deita, etc.)
#
# Usage:
#   ./compress_dataset.sh <dataset_name> [dataset_name2 ...]
#   ./compress_dataset.sh open_r1
#   ./compress_dataset.sh open_r1 openhermes ultramedical
#   ./compress_dataset.sh all          # compress all datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_BASE="$SCRIPT_DIR/dataset"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name> [dataset_name2 ...]"
    echo "       $0 all"
    echo ""
    echo "Available datasets:"
    ls -1 "$DATASET_BASE" | while read d; do
        [ -d "$DATASET_BASE/$d" ] && echo "  $d"
    done
    exit 1
fi

# Resolve target list
if [ "$1" = "all" ]; then
    targets=()
    for d in "$DATASET_BASE"/*/; do
        targets+=("$(basename "$d")")
    done
else
    targets=("$@")
fi

for name in "${targets[@]}"; do
    src="$DATASET_BASE/$name"
    if [ ! -d "$src" ]; then
        echo "[SKIP] $src does not exist"
        continue
    fi

    output="$DATASET_BASE/${name}.tar.gz"

    # Collect files: json, profile.txt, index, pkl — top-level only (no subdirs)
    file_list=$(find "$src" -maxdepth 1 -type f \
        \( -name "*.json" -o -name "profile.txt" -o -name "*.index" -o -name "*.pkl" \) \
        | sort)

    count=$(echo "$file_list" | grep -c . || true)
    if [ "$count" -eq 0 ]; then
        echo "[SKIP] $name: no matching files"
        continue
    fi

    echo "[PACK] $name: $count files -> $output"

    # Use tar with -C to get relative paths inside the archive
    echo "$file_list" | sed "s|^$src/||" | \
        tar -czf "$output" -C "$src" -T -

    size=$(du -h "$output" | cut -f1)
    echo "[DONE] $name: $size"
    echo ""
done
