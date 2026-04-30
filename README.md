# Parallel Sheet Simulator
**Ankita Kundu & Michael Nguyen — 15-418/618 Final Project**

CUDA-accelerated mass-spring cloth simulation.

## Versions

| Version | File | Key Optimization |
|---------|------|-----------------|
| V0 | `cloth_sim_seq.cpp` | Sequential CPU baseline |
| V1 | `simCode/cloth_sim_v1.cu` | Naive CUDA port (AoS, node-parallel, CSR adjacency) |
| V2 | `simCode/cloth_sim_v2.cu` | SoA memory layout + partial kernel fusion |
| V3 | `simCode/cloth_sim_v3.cu` | 2D shared-memory tiled spring kernel |
| V4 | `simCode/cloth_sim_v4.cu` | Fused kernels + constant memory + `__restrict__` |

## Requirements

- CUDA Toolkit 
- Python
- Python packages in `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Build

Build all versions:

```bash
make all
```

Build specific targets:

```bash
make seq    # CPU baseline (works locally and on GHC)
make v1     # CUDA V1 (requires CUDA toolkit)
make v2     # CUDA V2
make v3     # CUDA V3
make v4     # CUDA V4
```

**For GHC (with CUDA 11.7):** Set the host compiler explicitly:

```bash
CUDAHOSTCXX=/usr/bin/g++-11 make all
```

## Run

```bash
# Sequential baseline (runs anywhere with g++)
bin/cloth_sim_seq 500 500 1200

# CUDA versions (require GPU)
bin/cloth_sim_v1 500 500 1200
bin/cloth_sim_v2 500 500 1200
bin/cloth_sim_v3 500 500 1200

# V4 with optional tile dimensions (default: 32x8)
bin/cloth_sim_v4 500 500 1200 32 8
```

## Benchmarking

Run experiments using the Python script:

```bash
python3 scripts/run_experiment.py --help
```

For manual benchmarking, run multiple trials and capture CSV output:

```bash
# Example: benchmark sequential version
for i in {1..5}; do
    bin/cloth_sim_seq 100 100 1200 >> results/metrics/seq_bench.csv
done
```

Output format: `version,grid_w,grid_h,num_particles,num_springs,num_steps,elapsed_ms`

## Plotting

Generate plots from collected data:

```bash
# Use plotting scripts in plots/
python3 plots/plot_speedup.py
python3 plots/plot_runtime.py
python3 plots/plot_all_speedup.py
python3 plots/profiling.py
```

View visualization tools:

```bash
python3 scripts/visualize.py
```

## Repository Layout

```text
cloth_sim_seq.cpp                Sequential CPU reference
simCode/                         CUDA implementations (V1-V4)
  cloth_sim_v1.cu                V1: Naive CUDA port
  cloth_sim_v2.cu                V2: SoA + kernel fusion
  cloth_sim_v3.cu                V3: Shared memory tiling
  cloth_sim_v4.cu                V4: Fully optimized
plots/                           Result plots and plotting scripts
  *.png                          Generated performance plots
  *.py                           Python plotting utilities
scripts/
  generate_frames.sh             Generate visualization frames
  run_experiment.py              Config-based scene runner
  visualize.py                   Data visualization tool
results/                         Generated outputs (gitignored)
  metrics/                       Benchmark CSV/log outputs
  plots/                         Additional plot outputs
  outputs/                       Simulation outputs and references
  journal/                       Analysis notes
Makefile                         Build/benchmark targets
README.md                        This file
requirements.txt                 Python dependencies
.gitignore                       Git ignore rules
```
