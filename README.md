# Parallel-Sheet-Simulator

Minimal cloth simulation project structure:

## Folders

- `simCode/` - C++ simulation implementations and dependencies.
  - `cloth_sim_seq.cpp` - sequential simulator (config-driven, metrics + output writing).
  - `cloth_sim_parallel.cpp` - parallel simulator entry point (scaffold file).
  - `include/json.hpp` - JSON parser header used by simulator configs.
- `configs/` - experiment and validation configuration.
  - `experiments/*.json` - scene presets.
  - `thresholds.json` - pass/fail validation thresholds.
- `scripts/` - tooling scripts.
  - `run_experiment.py` - run one/all experiments and validate metrics.
  - `visualize.py` - convert frame CSV into animated GIF.
- `results/` - simulation outputs only.
  - `metrics/` - per-step CSV metrics + per-run summary JSON files.
  - `outputs/` - rendered artifacts (`cloth_frames.csv`, `cloth_meta.csv`, `cloth_sim.gif`).

## Quick Start

1. Build simulator:
   - `cd simCode`
   - `g++ -O2 -std=c++17 -I include -o cloth_sim cloth_sim_seq.cpp`
   - `cd ..`
2. Run a scene:
   - `./simCode/cloth_sim configs/experiments/drop_on_sphere.json`
3. Run validation harness:
   - `python scripts/run_experiment.py --all`
4. Render GIF from latest frames:
   - `python scripts/visualize.py`
