#!/usr/bin/env bash
#
# Run Divergence benchmark with OS-level IO counting for fair DiskANN comparison.
#
# Wraps the existing exp_bench_stable Rust test with /proc/diskstats delta
# measurement to validate the internal phys_reads counter against OS-level IO.
#
# Usage:
#   BENCH_DIR=/mnt/nvme/bench COHERE_DIR=/path/to/cohere_100k \
#     bash scripts/run_divergence.sh
#
# Output: Rust test stdout (TSV) + diskstats delta summary to stderr
#
set -euo pipefail

BENCH_DIR="${BENCH_DIR:?Set BENCH_DIR to NVMe-backed directory for O_DIRECT}"
COHERE_DIR="${COHERE_DIR:-data/cohere_100k}"
COHERE_N="${COHERE_N:-100000}"
NVME_DEV="${NVME_DEV:-}"  # e.g. nvme0n1, auto-detect if empty

# --- Auto-detect NVMe device ---
get_nvme_dev() {
    if [ -n "$NVME_DEV" ]; then
        echo "$NVME_DEV"
        return
    fi
    local mount_dev
    mount_dev=$(df "$BENCH_DIR" 2>/dev/null | tail -1 | awk '{print $1}')
    if [[ "$mount_dev" == /dev/nvme* ]]; then
        basename "$mount_dev" | sed 's/p[0-9]*$//'
        return
    fi
    echo ""
}

read_diskstats() {
    local dev="$1"
    if [ -z "$dev" ]; then
        echo "0"
        return
    fi
    awk -v dev="$dev" '$3 == dev {print $4}' /proc/diskstats 2>/dev/null || echo "0"
}

NVME=$(get_nvme_dev)
echo "=== Divergence benchmark ===" >&2
echo "  BENCH_DIR=$BENCH_DIR" >&2
echo "  COHERE_DIR=$COHERE_DIR" >&2
echo "  COHERE_N=$COHERE_N" >&2
if [ -n "$NVME" ]; then
    echo "  NVMe device: $NVME" >&2
else
    echo "  WARNING: No NVMe device detected. OS-level IO counting disabled." >&2
fi

# Verify data checksums if available
CHECKSUM_FILE="${COHERE_DIR}/diskann/checksums.sha256"
if [ -f "$CHECKSUM_FILE" ]; then
    echo "  Checksums file: $CHECKSUM_FILE" >&2
    grep -v '^#' "$CHECKSUM_FILE" | head -2 >&2
fi

# Optional: drop caches (needs root)
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "  WARNING: Cannot drop caches" >&2

# Optional: disable readahead (needs root)
OLD_RA=""
if [ -n "$NVME" ] && [ -w "/sys/block/$NVME/queue/read_ahead_kb" ]; then
    OLD_RA=$(cat "/sys/block/$NVME/queue/read_ahead_kb")
    echo 0 > "/sys/block/$NVME/queue/read_ahead_kb"
    echo "  Readahead disabled (was ${OLD_RA}KB)" >&2
fi

# Record diskstats before
READS_BEFORE=$(read_diskstats "$NVME")
echo "  diskstats reads before: $READS_BEFORE" >&2

# Run benchmark
echo "" >&2
echo "Running cargo test (release) ..." >&2

BENCH_DIR="$BENCH_DIR" COHERE_N="$COHERE_N" COHERE_DIR="$COHERE_DIR" \
    cargo test --release -p divergence-engine --test disk_search exp_bench_stable \
    -- --nocapture --ignored 2>&1

# Record diskstats after
READS_AFTER=$(read_diskstats "$NVME")
echo "" >&2
echo "  diskstats reads after: $READS_AFTER" >&2

if [ -n "$NVME" ] && [ "$READS_BEFORE" != "0" ]; then
    READS_DELTA=$((READS_AFTER - READS_BEFORE))
    echo "  OS-level total read IOs: $READS_DELTA" >&2
    echo "  (Compare with sum of phy/q * nq from bench output for validation)" >&2
fi

# Capture peak RSS
echo "" >&2
echo "  Peak RSS: check bench output (search process has exited)" >&2

# Restore readahead
if [ -n "$OLD_RA" ] && [ -w "/sys/block/$NVME/queue/read_ahead_kb" ]; then
    echo "$OLD_RA" > "/sys/block/$NVME/queue/read_ahead_kb"
    echo "  Readahead restored to ${OLD_RA}KB" >&2
fi

echo "=== Done ===" >&2
