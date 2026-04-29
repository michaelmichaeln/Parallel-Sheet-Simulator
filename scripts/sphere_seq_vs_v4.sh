#!/bin/bash

SIZE=500
TIMESTEPS=1200
OUTFILE="results/sphere_seq_vs_v4.csv"

mkdir -p results

echo "sphere_r,seq_ms,v4_ms,speedup" > "$OUTFILE"

get_time() {
    "$@" 2>&1 | grep "Done in" | tail -1 | awk '{print $3}'
}

for R in 0.2 0.4 0.6 0.8 1.0; do
    echo "Running sphere radius $R..."

    seq_t=$(get_time ./bin/cloth_sim_seq "$SIZE" "$SIZE" "$TIMESTEPS" --sphere_r "$R")
    v4_t=$(get_time ./bin/cloth_sim_v4 "$SIZE" "$SIZE" "$TIMESTEPS" --sphere_r "$R")

    speedup=$(echo "$seq_t / $v4_t" | bc -l)

    echo "$R,$seq_t,$v4_t,$speedup" >> "$OUTFILE"
    echo "seq=$seq_t ms, v4=$v4_t ms, speedup=$speedup"
done

echo "Saved to $OUTFILE"