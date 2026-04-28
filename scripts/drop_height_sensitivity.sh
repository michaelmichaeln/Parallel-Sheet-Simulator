#!/bin/bash

SIZE=1000
TIMESTEPS=1200
OUTFILE="results/drop_height_sensitivity.csv"

mkdir -p results

echo "drop_y,time_ms" > "$OUTFILE"

run_time() {
    ./bin/cloth_sim_v4 "$SIZE" "$SIZE" "$TIMESTEPS" \
        --drop_y "$1" 2>&1 \
    | grep "Done in" | awk '{print $3}'
}

HEIGHTS=(1.5 2.0 2.5 3.0 3.5 4.0 5.0)

for H in "${HEIGHTS[@]}"; do
    echo "Running drop height $H..."

    t=$(run_time "$H")

    echo "Time: $t ms"

    echo "$H,$t" >> "$OUTFILE"
done

echo "Saved to $OUTFILE"