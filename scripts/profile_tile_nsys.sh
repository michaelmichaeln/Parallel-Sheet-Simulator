#!/bin/bash

SIZE=500
TIMESTEPS=1200
OUTFILE="results/nsys_tile_profile.csv"
LOGDIR="results/nsys_logs"

mkdir -p results "$LOGDIR"

echo "tile_w,tile_h,threads_per_block,status,total_ms,force_kernel_found,integrate_kernel_found,log_file" > "$OUTFILE"

run_profile() {
    TW=$1
    TH=$2
    THREADS=$((TW * TH))
    NAME="tile_${TW}x${TH}"
    LOGFILE="$LOGDIR/${NAME}.txt"

    echo "Profiling tile ${TW}x${TH}..."

    nsys profile --stats=true --force-overwrite=true \
        -o "$LOGDIR/${NAME}" \
        ./bin/cloth_sim_v4 "$SIZE" "$SIZE" "$TIMESTEPS" \
        --tile_w "$TW" --tile_h "$TH" \
        > "$LOGFILE" 2>&1

    if grep -q "ERROR: NaN/Inf detected" "$LOGFILE"; then
        echo "  NaN/Inf detected, marking as FAIL"
        echo "$TW,$TH,$THREADS,FAIL,,,,${LOGFILE}" >> "$OUTFILE"
        return
    fi

    total_ms=$(grep "Done in" "$LOGFILE" | tail -1 | awk '{print $3}')

    force_found=$(grep -c "fused_force_kernel" "$LOGFILE")
    integrate_found=$(grep -c "fused_integrate_collision_kernel" "$LOGFILE")

    echo "  total_ms=$total_ms"
    echo "$TW,$TH,$THREADS,PASS,$total_ms,$force_found,$integrate_found,$LOGFILE" >> "$OUTFILE"
}

run_profile 8 8
run_profile 16 16
run_profile 32 8
run_profile 32 16

echo "Saved CSV to $OUTFILE"
echo "Logs saved in $LOGDIR"