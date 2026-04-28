#!/bin/bash

SIZE=1000
TIMESTEPS=1200
OUTFILE="results/tile_sensitivity.csv"

mkdir -p results

echo "tile_w,tile_h,time_ms" > "$OUTFILE"

run_time() {
    ./bin/cloth_sim_v4 "$SIZE" "$SIZE" "$TIMESTEPS" \
        --tile_w "$1" --tile_h "$2" 2>&1 \
    | grep "Done in" | awk '{print $3}'
}

for TX in 8 16 32; do
    for TY in 8 16 32; do
        echo "Running ${TX}x${TY}..."

        t=$(run_time $TX $TY)

        echo "Time: $t ms"

        echo "$TX,$TY,$t" >> "$OUTFILE"
    done
done

echo "Saved to $OUTFILE"