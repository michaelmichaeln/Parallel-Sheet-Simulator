# Parallel Sheet Simulator
**Ankita Kundu & Michael Nguyen — 15-418/618 Final Project**

CUDA-accelerated mass-spring cloth simulator with iterative optimization from naive GPU port to fully optimized kernels. Benchmarked across 4 grid sizes (10K to 1M particles) on RTX 2080.

## Versions

| Version | File | Key Optimization |
|---------|------|-----------------|
| V0 | `cloth_sim_seq.cpp` | Sequential CPU baseline |
| V1 | `simCode/cloth_sim_v1.cu` | Naive CUDA port (AoS, node-parallel, CSR adjacency) |
| V2 | `simCode/cloth_sim_v2.cu` | SoA memory layout for coalesced access |
| V3 | `simCode/cloth_sim_v3.cu` | 2D shared memory tiling with halo |
| V4 | `simCode/cloth_sim_v4.cu` | Kernel fusion, constant memory, `__restrict__` |

## Quick Start (GHC machines)

```bash
make all                    # build sequential + all CUDA versions
make bench                  # build + run full benchmark + generate plots
```

### Run individually
```bash
bin/cloth_sim_seq 500 500 1200          # sequential: 500x500 grid, 1200 steps
bin/cloth_sim_v4  500 500 1200 16 16    # V4: with 16x16 tile size
```

### Generate plots from existing results
```bash
python3 scripts/plot_results.py
```

## Project Structure

```
cloth_sim_seq.cpp              Sequential CPU reference
simCode/
  cloth_sim_v1.cu              V1: Naive CUDA (AoS)
  cloth_sim_v2.cu              V2: SoA layout
  cloth_sim_v3.cu              V3: Shared memory tiling
  cloth_sim_v4.cu              V4: Fully optimized
scripts/
  benchmark.sh                 Automated benchmark suite (5 repeats, median)
  plot_results.py              Generate speedup/timing PNG plots
results/
  metrics/                     Raw timing CSVs
  plots/                       Generated charts (speedup, absolute time, progression)
  journal/optimization_journal.md   Version-by-version decision log
  outputs/                     CPU reference binaries for validation
Makefile                       Build targets: seq, v1-v4, all, bench, clean
visualize.py                   3D cloth animation from frame data
```

## Requirements
- CUDA Toolkit (nvcc) with SM 7.5 support
- g++ with C++17 support
- Python 3 with `pandas`, `numpy`, `matplotlib` (for plots)
