#!/usr/bin/env bash
# ==============================================================================
# scp_frames.sh — Pull v4 frame CSVs from GHC and render GIFs locally.
#
# Usage:
#   bash scripts/scp_frames.sh [user@host]
#
# Default host: mnguyen3@ghc28.ghc.andrew.cmu.edu
# ==============================================================================
set -euo pipefail

HOST="${1:-mnguyen3@ghc28.ghc.andrew.cmu.edu}"
REMOTE_DIR="~/private/project/Parallel-Sheet-Simulator/results/outputs"
LOCAL_DIR="results/outputs"

mkdir -p "$LOCAL_DIR"

SIZES=("25 25" "100 100" "250 250")

echo "=== Pulling frame CSVs from $HOST ==="
for size in "${SIZES[@]}"; do
    read -r W H <<< "$size"
    FNAME="frames_v4_${W}x${H}.csv"
    echo -n "  $FNAME ... "
    if scp "${HOST}:${REMOTE_DIR}/${FNAME}" "${LOCAL_DIR}/${FNAME}" 2>/dev/null; then
        ROWS=$(wc -l < "${LOCAL_DIR}/${FNAME}")
        echo "ok  (${ROWS} rows)"
    else
        echo "SKIP (not found on remote)"
    fi
done

echo ""
echo "=== Rendering GIFs ==="
for size in "${SIZES[@]}"; do
    read -r W H <<< "$size"
    CSV="${LOCAL_DIR}/frames_v4_${W}x${H}.csv"
    GIF="${LOCAL_DIR}/cloth_sim_v4_${W}x${H}.gif"
    if [ -f "$CSV" ]; then
        echo -n "  ${W}x${H} -> $GIF ... "
        python3 scripts/visualize.py --frames "$CSV" --gif "$GIF"
    else
        echo "  ${W}x${H}: CSV not found, skipping"
    fi
done

echo ""
echo "=== Done ==="
ls -lh "$LOCAL_DIR"/cloth_sim_v4_*.gif 2>/dev/null || true
