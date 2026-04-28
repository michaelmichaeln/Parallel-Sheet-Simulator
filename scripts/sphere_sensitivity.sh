#!/bin/bash

SIZE=500
TIMESTEPS=1200
OUTFILE="results/sphere_sensitivity.csv"

mkdir -p results

echo "sphere_r,time_ms" > "$OUTFILE"

run_time() {
    ./bin/cloth_sim_v4 "$SIZE" "$SIZE" "$TIMESTEPS" \
        --sphere_r "$1" 2>&1 \
    | grep "Done in" | awk '{print $3}'
}

# sweep radii
RADII=(0.2 0.4 0.6 0.8 1.0)

for R in "${RADII[@]}"; do
    echo "Running sphere radius $R..."

    t=$(run_time "$R")

    echo "Time: $t ms"

    echo "$R,$t" >> "$OUTFILE"
done

echo "Saved to $OUTFILE"