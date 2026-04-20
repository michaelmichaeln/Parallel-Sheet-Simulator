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

```bash
make all
```

Build specific targets:

```bash
make seq
make v1
make v2
make v3
make v4
```

## Run

```bash
# Sequential baseline
bin/cloth_sim_seq 500 500 1200

# CUDA V4 (optional tile dimensions for V4)
bin/cloth_sim_v4 500 500 1200 32 8
```

## Benchmark Workflows

### 1) Standard benchmark (Makefile)

```bash
make bench
```

This runs:
- `scripts/benchmark.sh` to write `results/metrics/benchmark_results.csv`
- `scripts/plot_results.py` to generate plots in `results/plots/`

### 2) Full GHC benchmark workflow

Use the full script when running your report pipeline on GHC:

```bash
CUDAHOSTCXX=/usr/bin/g++-11 bash scripts/run_full_bench.sh
```

This script:
- builds seq + all CUDA versions
- generates CPU references if missing
- benchmarks V1-V4 across target sizes
- prints summary and speedup tables
- runs a correctness check against CPU references

Note: CUDA 11.7 requires an older host compiler on GHC, so set `CUDAHOSTCXX=/usr/bin/g++-11` as shown.

## Plotting

Generate plots from benchmark CSV:

```bash
python3 scripts/plot_results.py
```

Output files:
- `results/plots/speedup.png`
- `results/plots/absolute_time.png`
- `results/plots/version_progression.png`

## Repository Layout

```text
cloth_sim_seq.cpp                Sequential CPU reference
simCode/                         CUDA implementations (V1-V4)
scripts/
  benchmark.sh                   Median timing benchmark (5 repeats)
  run_full_bench.sh              Full GHC benchmark/report pipeline
  plot_results.py                Plot generation from benchmark CSV
  run_experiment.py              Config-based scene runner
results/
  metrics/                       Benchmark CSV/log outputs
  plots/                         Generated plot PNG files
  outputs/                       CPU reference binaries for validation
  journal/                       Report journal and analysis notes
Makefile                         Build/benchmark targets
```
