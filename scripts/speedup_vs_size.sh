#!/bin/bash

SIZES=(100 250 500 750 1000)
TIMESTEPS=1200
OUTFILE="results/speedup_vs_size.csv"

mkdir -p results results/plots

echo "size,v0_ms,v1_ms,v2_ms,v3_ms,v4_ms" > "$OUTFILE"

run_and_get_time() {
    exe=$1
    size=$2

    "$exe" "$size" "$size" "$TIMESTEPS" 2>&1 | grep "Done in" | tail -1 | awk '{print $3}'
}

for N in "${SIZES[@]}"; do
    echo "Running ${N}x${N}..."

    t0=$(run_and_get_time ./bin/cloth_sim_seq "$N")
    t1=$(run_and_get_time ./bin/cloth_sim_v1  "$N")
    t2=$(run_and_get_time ./bin/cloth_sim_v2  "$N")
    t3=$(run_and_get_time ./bin/cloth_sim_v3  "$N")
    t4=$(run_and_get_time ./bin/cloth_sim_v4  "$N")

    echo "Times ms: v0=$t0 v1=$t1 v2=$t2 v3=$t3 v4=$t4"

    echo "$N,$t0,$t1,$t2,$t3,$t4" >> "$OUTFILE"
done

echo "Saved results to $OUTFILE"
