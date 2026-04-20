#!/usr/bin/env bash
# ==============================================================================
# benchmark.sh — Run all simulator versions at multiple grid sizes
# Collects timing data to results/metrics/benchmark_results.csv
#
# Usage:  bash scripts/benchmark.sh [NUM_STEPS]
# ==============================================================================
set -euo pipefail

BINDIR="bin"
OUTFILE="results/metrics/benchmark_results.csv"
NUM_STEPS="${1:-1200}"
REPEATS=5

SIZES=(
    "100 100"
    "250 250"
    "500 500"
    "1000 1000"
)

VERSIONS=("seq" "v1" "v2" "v3" "v4")

mkdir -p results/metrics results/outputs

echo "version,grid_w,grid_h,num_particles,num_springs,num_steps,elapsed_ms" > "$OUTFILE"

# ── Phase 1: Generate CPU reference files ────────────────────────────────────
echo "=== Generating CPU reference files ==="
for size in "${SIZES[@]}"; do
    read -r W H <<< "$size"
    REF="results/outputs/ref_${W}x${H}.bin"
    if [ ! -f "$REF" ]; then
        echo "  Generating reference for ${W}x${H}..."
        "${BINDIR}/cloth_sim_seq" "$W" "$H" "$NUM_STEPS" 2>/dev/null >> /dev/null
    else
        echo "  Reference for ${W}x${H} already exists."
    fi
done
echo ""

# ── Phase 2: Benchmark all versions ─────────────────────────────────────────
for version in "${VERSIONS[@]}"; do
    BIN="${BINDIR}/cloth_sim_${version}"
    if [ ! -x "$BIN" ]; then
        echo "WARNING: $BIN not found or not executable, skipping."
        continue
    fi

    echo "=== Benchmarking $version ==="
    for size in "${SIZES[@]}"; do
        read -r W H <<< "$size"
        echo -n "  ${W}x${H}: "

        TIMES_FILE=$(mktemp)

        # Warm-up run (discarded)
        "$BIN" "$W" "$H" "$NUM_STEPS" > /dev/null 2>/dev/null || true

        for r in $(seq 1 "$REPEATS"); do
            LINE=$("$BIN" "$W" "$H" "$NUM_STEPS" 2>/dev/null)
            echo "$LINE" >> "$OUTFILE"
            # Extract elapsed_ms (last field)
            MS=$(echo "$LINE" | cut -d',' -f7)
            echo "$MS" >> "$TIMES_FILE"
            echo -n "${MS}ms "
        done

        # Report median
        MEDIAN=$(sort -n "$TIMES_FILE" | awk "NR==$(( (REPEATS+1)/2 )){print}")
        echo " (median: ${MEDIAN}ms)"
        rm -f "$TIMES_FILE"
    done
    echo ""
done

echo "Results written to $OUTFILE"
echo "Run 'python3 scripts/plot_results.py' to generate plots."
