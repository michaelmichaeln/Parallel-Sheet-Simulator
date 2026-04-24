#!/usr/bin/env bash
# ==============================================================================
# generate_frames.sh — Run cloth_sim_v4 with frame output for visualization.
#
# Run this on GHC after building v4:
#   bash scripts/generate_frames.sh
#
# Frame CSVs are written to results/outputs/frames_v4_WxH.csv
# Practical size limit: 250x250 (~300 MB CSV). 500x500+ omitted — too large.
# ==============================================================================
set -euo pipefail

BINDIR="bin"
OUTDIR="results/outputs"
NUM_STEPS="${1:-1200}"
CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v g++-11 2>/dev/null || command -v g++)}"

mkdir -p "$OUTDIR"

# ── Always rebuild v4 (ensures --frames support from updated source) ───────────
echo "=== Building v4 (with --frames support) ==="
mkdir -p "$BINDIR"
nvcc -O2 -std=c++17 -arch=sm_75 -ccbin "$CUDAHOSTCXX" \
    -o "$BINDIR/cloth_sim_v4" simCode/cloth_sim_v4.cu && echo "  OK" || { echo "  FAIL"; exit 1; }

# ── Sizes for frame generation ─────────────────────────────────────────────────
# 500x500 / 1000x1000 omitted: ~800 MB / ~3 GB CSV, impractical for GIF
SIZES=("25 25" "100 100" "250 250")

echo "=== Generating frames with v4 (${NUM_STEPS} steps, every 10) ==="
for size in "${SIZES[@]}"; do
    read -r W H <<< "$size"
    OUTFILE="$OUTDIR/frames_v4_${W}x${H}.csv"
    echo -n "  ${W}x${H} -> $OUTFILE ... "
    "$BINDIR/cloth_sim_v4" "$W" "$H" "$NUM_STEPS" 16 16 --frames "$OUTFILE" \
        > /dev/null 2>&1
    ROWS=$(wc -l < "$OUTFILE")
    echo "done  (${ROWS} rows)"
done

echo ""
echo "=== Frame files ==="
ls -lh "$OUTDIR"/frames_v4_*.csv 2>/dev/null || echo "  (none found)"
echo ""
echo "Pull to local with:"
echo "  bash scripts/scp_frames.sh mnguyen3@ghc28.ghc.andrew.cmu.edu"
