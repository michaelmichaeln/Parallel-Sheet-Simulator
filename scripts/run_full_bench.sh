#!/usr/bin/env bash
# ==============================================================================
# run_full_bench.sh — One-shot: pull, build, benchmark, report
# Run this on GHC after checking out the parallel branch.
#
# Usage:  bash scripts/run_full_bench.sh
# ==============================================================================
set -euo pipefail

echo "=== System Info ==="
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
g++ --version | head -1
nvcc --version | tail -1
echo ""

BINDIR="bin"
NUM_STEPS=1200
REPEATS=5

SIZES=("100 100" "250 250" "500 500" "1000 1000")
VERSIONS=("seq" "v1" "v2" "v3" "v4")
BENCH_VERSIONS=("v1" "v2" "v3" "v4")

# Hardcoded CPU sequential medians (ms) from GHC reference run.
declare -A HARDCODED_SEQ_MEDIANS=(
    ["100x100"]="436.2"
    ["250x250"]="2752.5"
    ["500x500"]="12208.8"
    ["1000x1000"]="55321.5"
)

mkdir -p "$BINDIR" results/outputs results/metrics results/plots results/journal

# ── Build ────────────────────────────────────────────────────────────────────
echo "=== Building All Versions ==="
echo -n "  seq... "
g++ -O2 -std=c++17 -march=native -o "$BINDIR/cloth_sim_seq" cloth_sim_seq.cpp && echo "OK" || echo "FAIL"

for v in v1 v2 v3 v4; do
    echo -n "  $v... "
    nvcc -O2 -std=c++17 -arch=sm_75 -o "$BINDIR/cloth_sim_$v" "simCode/cloth_sim_$v.cu" 2>&1 && echo "OK" || echo "FAIL"
done
echo ""

# ── Generate CPU references ──────────────────────────────────────────────────
echo "=== Generating CPU Reference Files ==="
for size in "${SIZES[@]}"; do
    read -r W H <<< "$size"
    REF="results/outputs/ref_${W}x${H}.bin"
    echo -n "  ${W}x${H}... "
    if [ -f "$REF" ]; then
        echo "SKIP (exists)"
    else
        "$BINDIR/cloth_sim_seq" "$W" "$H" "$NUM_STEPS" > /dev/null 2>&1 && echo "OK" || echo "FAIL"
    fi
done
echo ""

# ── Benchmark ────────────────────────────────────────────────────────────────
OUTFILE="results/metrics/benchmark_results.csv"
echo "version,grid_w,grid_h,num_particles,num_springs,num_steps,elapsed_ms" > "$OUTFILE"

for version in "${BENCH_VERSIONS[@]}"; do
    BIN="${BINDIR}/cloth_sim_${version}"
    if [ ! -x "$BIN" ]; then
        echo "WARNING: $BIN not found, skipping."
        continue
    fi

    echo "=== Benchmarking $version ==="
    for size in "${SIZES[@]}"; do
        read -r W H <<< "$size"
        echo -n "  ${W}x${H}: "

        TIMES_FILE=$(mktemp)

        # Warm-up
        "$BIN" "$W" "$H" "$NUM_STEPS" > /dev/null 2>/dev/null || true

        for r in $(seq 1 "$REPEATS"); do
            LINE=$("$BIN" "$W" "$H" "$NUM_STEPS" 2>/dev/null)
            echo "$LINE" >> "$OUTFILE"
            MS=$(echo "$LINE" | cut -d',' -f7)
            echo "$MS" >> "$TIMES_FILE"
            echo -n "${MS}ms "
        done

        MEDIAN=$(sort -n "$TIMES_FILE" | awk "NR==$(( (REPEATS+1)/2 )){print}")
        echo " -> median: ${MEDIAN}ms"
        rm -f "$TIMES_FILE"
    done
    echo ""
done

echo "=== Results saved to $OUTFILE ==="
echo ""

# ── Print summary table ─────────────────────────────────────────────────────
echo "=== SUMMARY TABLE (median ms) ==="
echo "version,100x100,250x250,500x500,1000x1000"

for version in "${VERSIONS[@]}"; do
    echo -n "$version"
    if [ "$version" = "seq" ]; then
        for size in "${SIZES[@]}"; do
            read -r W H <<< "$size"
            echo -n ",${HARDCODED_SEQ_MEDIANS[${W}x${H}]}"
        done
    else
        for size in "${SIZES[@]}"; do
            read -r W H <<< "$size"
            # Get all elapsed_ms for this version+size, find median
            VALS=$(grep "^${version},${W},${H}," "$OUTFILE" | cut -d',' -f7 | sort -n)
            MEDIAN=$(echo "$VALS" | awk "NR==$(( (REPEATS+1)/2 )){print}")
            echo -n ",$MEDIAN"
        done
    fi
    echo ""
done

echo ""
echo "=== SPEEDUP TABLE (vs sequential) ==="

# Collect sequential medians
declare -A SEQ_MEDIANS
for size in "${SIZES[@]}"; do
    read -r W H <<< "$size"
    SEQ_MEDIANS["${W}x${H}"]="${HARDCODED_SEQ_MEDIANS[${W}x${H}]}"
done

echo "version,100x100,250x250,500x500,1000x1000"
for version in "${VERSIONS[@]}"; do
    if [ "$version" = "seq" ]; then continue; fi
    echo -n "$version"
    for size in "${SIZES[@]}"; do
        read -r W H <<< "$size"
        VALS=$(grep "^${version},${W},${H}," "$OUTFILE" | cut -d',' -f7 | sort -n)
        GPU_MEDIAN=$(echo "$VALS" | awk "NR==$(( (REPEATS+1)/2 )){print}")
        SEQ_MEDIAN="${SEQ_MEDIANS[${W}x${H}]}"
        if [ -n "$GPU_MEDIAN" ] && [ -n "$SEQ_MEDIAN" ]; then
            SPEEDUP=$(echo "scale=2; $SEQ_MEDIAN / $GPU_MEDIAN" | bc 2>/dev/null || echo "N/A")
            echo -n ",${SPEEDUP}x"
        else
            echo -n ",N/A"
        fi
    done
    echo ""
done

echo ""
echo "=== VALIDATION (from stderr of last run) ==="
for version in v1 v2 v3 v4; do
    BIN="${BINDIR}/cloth_sim_${version}"
    [ ! -x "$BIN" ] && continue
    echo "--- $version ---"
    "$BIN" 100 100 "$NUM_STEPS" > /dev/null 2>&1 || true
    "$BIN" 100 100 "$NUM_STEPS" 2>&1 1>/dev/null | grep -E "(Validation|ERROR)" || echo "  (no validation output)"
done

echo ""
echo "=== DONE ==="
echo "CSV at: $OUTFILE"
echo "Run 'python3 scripts/plot_results.py' to generate plots."
