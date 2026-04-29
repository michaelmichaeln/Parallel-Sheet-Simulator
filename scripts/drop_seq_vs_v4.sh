#!/bin/bash

SIZE=500
TIMESTEPS=1200
OUTFILE="results/drop_height_seq_vs_v4.csv"

mkdir -p results

echo "drop_y,seq_ms,v4_ms,speedup" > "$OUTFILE"

get_time() {
    "$@" 2>&1 | grep "Done in" | tail -1 | awk '{print $3}'
}

for H in 1.5 2.0 2.5 3.0 3.5 4.0 5.0; do
    echo "Running drop height $H..."

    seq_t=$(get_time ./bin/cloth_sim_seq "$SIZE" "$SIZE" "$TIMESTEPS" --drop_y "$H")
    v4_t=$(get_time ./bin/cloth_sim_v4 "$SIZE" "$SIZE" "$TIMESTEPS" --drop_y "$H")

    speedup=$(echo "$seq_t / $v4_t" | bc -l)

    echo "$H,$seq_t,$v4_t,$speedup" >> "$OUTFILE"
    echo "seq=$seq_t ms, v4=$v4_t ms, speedup=$speedup"
done

echo "Saved to $OUTFILE"