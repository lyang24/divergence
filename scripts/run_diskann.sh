#!/usr/bin/env bash
#
# Run DiskANN Rust disk-index benchmark on Cohere 100K.
#
# Uses the Rust diskann-benchmark binary (NOT C++ build_disk_index/search_disk_index).
# Input: JSON config file specifying build + search parameters.
#
# Prerequisites:
#   - DiskANN Rust repo built: cargo build --release -p diskann-benchmark --features disk-index
#   - Data exported via: python3 scripts/export_diskann_format.py --verify
#
# Usage:
#   DISKANN_BIN=/mnt/nvme/DiskANN/target/release/diskann-benchmark \
#   INPUT_JSON=scripts/diskann_cohere100k.json \
#     bash scripts/run_diskann.sh
#
set -euo pipefail

DISKANN_BIN="${DISKANN_BIN:-/mnt/nvme/DiskANN/target/release/diskann-benchmark}"
INPUT_JSON="${INPUT_JSON:-scripts/diskann_cohere100k.json}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/diskann_output.json}"

[ -x "$DISKANN_BIN" ] || { echo "ERROR: $DISKANN_BIN not found. Build DiskANN first." >&2; exit 1; }
[ -f "$INPUT_JSON" ] || { echo "ERROR: $INPUT_JSON not found." >&2; exit 1; }

echo "=== DiskANN Rust benchmark ===" >&2
echo "  Binary: $DISKANN_BIN" >&2
echo "  Input:  $INPUT_JSON" >&2
echo "  Output: $OUTPUT_JSON" >&2

# Optional: drop OS caches (needs root)
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "  WARNING: Cannot drop caches (need root)" >&2

"$DISKANN_BIN" run \
    --input-file "$INPUT_JSON" \
    --output-file "$OUTPUT_JSON" \
    2>&1

echo "" >&2
echo "Results written to: $OUTPUT_JSON" >&2
echo "=== Done ===" >&2
